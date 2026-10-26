"""
R: 「数える模型の残差」を学ぶTransformer。 logits = TF(文脈) + a * log P_bigram(次語 | 直前語)  を全体でsoftmax。
 Transformerは、直前語だけで当たる分(数えれば済む分)を学ばなくてよい。aは学習する1つの数(初期値1)。
 学習時の事前分布は「その語自身を引いた」数(leave-one-out)にして、自分の答えを見て当てる漏れを防ぐ。
 prior=none なら同じコードで普通のTransformer(比較用)。FLOP=T4と同じ式(事前分布の加算は無視できる大きさ。数えは2N回)。
使い方: LADDER_CORPUS=... python3 exp_r.py '{"N":16000000,"d":128,"nl":4,"seq":64,"epochs":3,"lr":0.002,"prior":"bi"}'
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, bits, log
import exp_scale as es
from exp_t3 import split_windows
from exp_t4 import GT, macs_per_token
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


class Prior:
    def __init__(s, stream, delta=0.9):
        prev = stream[:-1].astype(np.int64); nxt = stream[1:].astype(np.int64); s.delta = delta
        C = np.bincount(prev * V + nxt, minlength=V * V).reshape(V, V).astype(np.int32)
        R = C.sum(1).astype(np.float64); T = (C > 0).sum(1).astype(np.float64)
        pu = np.bincount(nxt, minlength=V) + 1.0; pu /= pu.sum()
        s.C = torch.from_numpy(C); s.R = torch.from_numpy(R.astype(np.float32)); s.T = torch.from_numpy(T.astype(np.float32))
        s.pu = torch.from_numpy(pu.astype(np.float32))
        s.NUM = torch.from_numpy((np.maximum(C - delta, 0) + delta * T[:, None] * pu[None, :]).astype(np.float32))

    def logp(s, prev, nxt=None):
        n = len(prev); num = s.NUM[prev]; r = s.R[prev]; ar = torch.arange(n)
        if nxt is not None:      # leave-one-out
            c = s.C[prev, nxt].float(); t = s.T[prev] - (c == 1).float()
            num[ar, nxt] = torch.clamp(c - 1 - s.delta, min=0) + s.delta * t * s.pu[nxt]
            r = r - 1
        p = num / r.clamp_min(1).unsqueeze(1); p = p / p.sum(1, keepdim=True)
        p = torch.where((r <= 0).unsqueeze(1), s.pu.unsqueeze(0).expand_as(p), p)
        return torch.log(p.clamp_min(1e-9))


def run(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    ctx = lab.load(cfg['N']); d, nl, seq = cfg['d'], cfg['nl'], cfg.get('seq', 64); S = seq + 1
    bs = cfg.get('bs', 128); lr = cfg.get('lr', 2e-3); use = cfg.get('prior', 'none') == 'bi'
    stream = np.concatenate(ctx.tr_i)[:cfg['N']]
    M = len(stream) // S; chunks = torch.from_numpy(stream[:M * S].reshape(M, S).astype(np.int64))
    res = dict(cfg=cfg); t0 = time.time()
    pr = Prior(stream, cfg.get('delta', 0.9)) if use else None
    ops_count = 2.0 * len(stream) if use else 0.0
    yte = ctx.yte[:20000]; Xte = torch.from_numpy(split_windows(ctx.te_i, ctx.mte, ctx.pte, len(yte), seq)); yt = torch.from_numpy(yte)
    if use:
        lpb = torch.cat([pr.logp(Xte[i:i + 2000, -1]) for i in range(0, len(Xte), 2000)])
        res['bigram_only_bits'] = bits(lpb[torch.arange(len(yt)), yt].exp().numpy().astype(np.float64)); log('2-gram単独 %.4f' % res['bigram_only_bits'])
    net = GT(d, nl, S); a = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW(list(net.parameters()) + ([a] if use else []), lr=lr, weight_decay=0.0)
    steps = int(cfg.get('epochs', 1) * (M // bs)); nev = 10; every = max(1, steps // nev)
    curve = []; flops = 0.0; st = 0; ttrain = 0.0

    def evaluate():
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Xte), 1024):
                z = net(Xte[i:i + 1024], last_only=True)
                if use: z = z + a * pr.logp(Xte[i:i + 1024, -1])
                out.append(torch.log_softmax(z, 1)[torch.arange(len(z)), yt[i:i + 1024]].exp().numpy())
        net.train(); return bits(np.concatenate(out).astype(np.float64))
    log('chunks %d steps %d' % (M, steps))
    while st < steps:
        for i in torch.randperm(M).split(bs):
            if len(i) < bs or st >= steps: break
            t = time.time()
            for g in opt.param_groups:
                g['lr'] = lr * min(1, (st + 1) / 300) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.95 + 0.05)
            b = chunks[i]; z = net(b[:, :-1])
            if use:
                z = z + a * pr.logp(b[:, :-1].reshape(-1), b[:, 1:].reshape(-1)).reshape(len(b), seq, V)
            loss = torch.nn.functional.cross_entropy(z.reshape(-1, V), b[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
            ttrain += time.time() - t; flops += 6.0 * macs_per_token(d, nl, seq) * bs * seq
            if st % every == 0 or st == steps:
                bb = evaluate(); curve.append((st, round(ttrain, 1), float('%.3e' % flops), round(bb, 4)))
                log('step %d 学習時間 %.0fs FLOP %.2e bit/語 %.4f a=%.3f' % (st, ttrain, flops, bb, a.item()))
    res.update(curve=curve, final_bits=curve[-1][3], flops=flops, ops_count=ops_count, train_sec=ttrain, a=a.item())
    log('RESULT ' + json.dumps(res))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
