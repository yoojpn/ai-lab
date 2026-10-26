"""
T4: 学習の無駄を削る「てこ」を、同じ土俵(学習FLOP vs bit/語)で測る。モデルは因果Transformer。
  select : all=全部の塊で学習 / random=ランダムにkeep割 / hard=数える方式が苦手な塊(損失の大きい塊)だけkeep割
           (数える方式の表は、学習用の半分Aから作り、残り半分Bの塊の難しさを測る。Bの塊だけで学習する)
  grow   : 1=浅く始めて、途中で層を複製して深くする(1→2→4層)
  lr, bs : 学習率とバッチ
  評価: テスト先頭20000窓、文脈seq語、同じ指標。FLOPは「学習した語数×1語あたりの積和×6」で数える(層は学習中の実際の深さ)。
使い方: LADDER_CORPUS=... python3 exp_t4.py '{"N":16000000,"d":128,"nl":4,"seq":64,"epochs":3,"select":"all"}'
"""
import os, sys, time, json, math, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, HIST, bits, log
import exp_scale as es
from exp_g import Chain
from exp_t3 import split_windows
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


class GT(torch.nn.Module):
    def __init__(s, d, nl_max, seq, heads=4):
        super().__init__()
        s.E = torch.nn.Embedding(V, d); s.pos = torch.nn.Parameter(torch.zeros(seq, d))
        s.layers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True) for _ in range(nl_max)])
        s.out = torch.nn.Linear(d, V); s.active = nl_max

    def forward(s, X, last_only=False):
        n = X.shape[1]; mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
        h = s.E(X) + s.pos[:n]
        for L in s.layers[:s.active]:
            h = L(h, src_mask=mask, is_causal=True)
        return s.out(h[:, -1] if last_only else h)


def macs_per_token(d, layers, seq):
    return layers * (12 * d * d + seq * d) + d * V


def evaluate(net, X, y):
    net.eval(); pt = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            lp = torch.log_softmax(net(torch.from_numpy(X[i:i + 1024]), last_only=True), 1)
            pt.append(lp[torch.arange(len(lp)), torch.from_numpy(y[i:i + 1024])].exp().numpy())
    net.train()
    return bits(np.concatenate(pt).astype(np.float64))


def run(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0); rng = np.random.default_rng(0)
    ctx = lab.load(cfg['N']); d, nl, seq = cfg['d'], cfg['nl'], cfg.get('seq', 64); S = seq + 1
    bs = cfg.get('bs', 128); lr = cfg.get('lr', 2e-3); sel = cfg.get('select', 'all'); keep = cfg.get('keep', 1.0)
    stream = np.concatenate(ctx.tr_i)[:cfg['N']]; h = len(stream) // 2
    A, B = stream[:h], stream[h:]
    M = len(B) // S; res = dict(cfg=cfg)
    idx = np.arange(M)
    if sel != 'all':
        if sel == 'random':
            idx = np.sort(rng.choice(M, int(M * keep), replace=False))
        else:
            # 数える方式(Aから作る)でBの各語の損失を測り、塊ごとの平均損失で難しい塊を選ぶ
            wa = np.stack([A[j:len(A) - HIST + j] for j in range(HIST)], 1); ya = A[HIST:]
            ch = Chain(wa, ya, [tuple(range(1, k + 1)) for k in range(4)], 0.9); del wa
            loss = np.zeros(len(B)); step = 2_000_000
            for a in range(0, len(B) - HIST, step):
                b = min(len(B) - HIST, a + step)
                wb = np.stack([B[a + j:b + j] for j in range(HIST)], 1); yb = B[a + HIST:b + HIST]
                loss[a + HIST:b + HIST] = -np.log2(ch.probs(wb, yb)[0] + 1e-12)
            cl = loss[:M * S].reshape(M, S).mean(1)
            res['easy_share'] = float((loss[HIST:] < 1.0).mean())            # 数える方式が易しく当てる語(1bit未満)の割合
            order = np.argsort(-cl); idx = np.sort(order[:int(M * keep)])
            res['loss_chunk_mean_all'] = float(cl.mean()); res['loss_chunk_mean_kept'] = float(cl[idx].mean())
            del ch
    chunks = torch.from_numpy(B[:M * S].reshape(M, S).astype(np.int64))[torch.from_numpy(idx)]
    n = len(chunks); steps = int(cfg.get('epochs', 1) * (n // bs))
    grow = cfg.get('grow', 0)
    net = GT(d, nl, S); net.active = 1 if grow else nl
    sched = [(0.2, 1), (0.5, 2), (1.01, nl)] if grow else [(1.01, nl)]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    yte = ctx.yte[:20000]; Xte = split_windows(ctx.te_i, ctx.mte, ctx.pte, len(yte), seq)
    nev = 10; every = max(1, steps // nev); curve = []; flops = 0.0; st = 0; ttrain = 0.0
    log('chunks %d (選択後 %d) steps %d' % (M, n, steps))
    while st < steps:
        for i in torch.randperm(n).split(bs):
            if len(i) < bs or st >= steps: break
            frac = st / steps
            want = next(l for f, l in sched if frac < f)
            if grow and want > net.active:
                # 層を複製して深くする(最後の層をコピー)
                while net.active < want:
                    net.layers[net.active].load_state_dict(copy.deepcopy(net.layers[net.active - 1].state_dict())); net.active += 1
                opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
            t = time.time()
            for g in opt.param_groups:
                g['lr'] = lr * min(1, (st + 1) / 300) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.95 + 0.05)
            b = chunks[i]
            loss = torch.nn.functional.cross_entropy(net(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
            ttrain += time.time() - t; flops += 6.0 * macs_per_token(d, net.active, seq) * bs * seq
            if st % every == 0 or st == steps:
                bb = evaluate(net, Xte, yte); curve.append((st, round(ttrain, 1), float('%.3e' % flops), round(bb, 4)))
                log('step %d 学習時間 %.0fs FLOP %.2e bit/語 %.4f' % (st, ttrain, flops, bb))
    res.update(curve=curve, final_bits=curve[-1][3], flops=flops, train_sec=ttrain)
    log('RESULT ' + json.dumps(res))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
