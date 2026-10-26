"""
VM: 語彙を粗くする(SuperBPE/BPE風)。頻出する2語連続を1トークンに併合(語彙は5000のまま: 単語4000+併合トークン最大999+<unk>)。
 M=0 は併合なし(同じ4000語の語彙)。評価は「元の単語1つあたりのbit」(全トークンの損失の合計 ÷ 元の単語数)で、併合の有無で公平に比べる。
 steps を指定すると、併合ありでも同じステップ数(=同じFLOP)で学習する。省略時は epochs でトークン列を回す。
使い方: LADDER_CORPUS=... python3 exp_vm.py '{"N":16000000,"M":999,"steps":5766}'
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, log
import exp_scale as es
from exp_t4 import GT, macs_per_token
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)
B = 4000


def apply_merge(d, keys, newid):
    d = np.where(d > B, 0, d)
    wl = np.ones(len(d), np.int64)
    if len(keys) == 0 or len(d) < 2: return d, wl
    code = d[:-1] * V + d[1:]
    i = np.minimum(np.searchsorted(keys, code), len(keys) - 1); hit = (keys[i] == code) & (d[:-1] > 0) & (d[1:] > 0)
    acc = []; last = -2
    for p in np.nonzero(hit)[0]:
        if p > last + 1: acc.append(p); last = p
    acc = np.array(acc, np.int64)
    if len(acc) == 0: return d, wl
    tok = d.copy(); tok[acc] = newid[i[acc]]; wl[acc] = 2
    keep = np.ones(len(d), bool); keep[acc + 1] = False
    return tok[keep], wl[keep]


def run(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0); t0 = time.time()
    ctx = lab.load(100000); N, M = cfg['N'], cfg.get('M', 0); d, nl, seq = cfg.get('d', 128), cfg.get('nl', 4), cfg.get('seq', 64); S = seq + 1
    docs = []; tot = 0
    for a in ctx.tr_i:
        if tot >= N: break
        docs.append(a[:N - tot]); tot += len(docs[-1])
    keys = np.zeros(0, np.int64); newid = np.zeros(0, np.int64)
    if M:
        codes = np.concatenate([np.where(x > B, 0, x)[:-1] * V + np.where(x > B, 0, x)[1:] for x in docs if len(x) > 1])
        a_, b_ = codes // V, codes % V; codes = codes[(a_ > 0) & (b_ > 0)]
        u, c = np.unique(codes, return_counts=True); top = np.argsort(-c)[:M]
        o = np.argsort(u[top]); keys = u[top][o]; newid = (B + 1 + np.argsort(np.argsort(-c[top])))[o]
        log('併合 %d 種(最頻は %d 回)' % (len(keys), c[top].max()))
    tr = [apply_merge(x, keys, newid) for x in docs]; te = [apply_merge(x, keys, newid) for x in ctx.te_i]
    trt = np.concatenate([t for t, _ in tr]); trw = np.concatenate([w for _, w in tr])
    tet = np.concatenate([t for t, _ in te]); tew = np.concatenate([w for _, w in te])
    log('訓練: 単語 %d → トークン %d (トークン/単語 %.3f) [%.0fs]' % (tot, len(trt), len(trt) / tot, time.time() - t0))
    nt = min(len(tet) // S, 3000); tec = torch.from_numpy(tet[:nt * S].reshape(nt, S)); tew_c = tew[:nt * S].reshape(nt, S)
    words_eval = float(tew_c[:, 1:].sum())
    M_ = len(trt) // S; chunks = torch.from_numpy(trt[:M_ * S].reshape(M_, S).astype(np.int64))
    net = GT(d, nl, S); lr = cfg.get('lr', 4e-3); bs = 128
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    steps = cfg.get('steps') or int(cfg.get('epochs', 3) * (M_ // bs)); every = max(1, steps // 10)
    macs = nl * (12 * d * d + seq * d) + d * V; flops = 0.0; st = 0; curve = []

    def evaluate():
        net.eval(); tot_bits = 0.0
        with torch.no_grad():
            for i in range(0, nt, 256):
                b = tec[i:i + 256].long(); lp = torch.log_softmax(net(b[:, :-1]), -1)
                tot_bits += -(lp.gather(2, b[:, 1:, None]).squeeze(2) / math.log(2)).sum().item()
        net.train(); return tot_bits / words_eval
    log('チャンク %d steps %d' % (M_, steps))
    while st < steps:
        for i in torch.randperm(M_).split(bs):
            if len(i) < bs or st >= steps: break
            for g in opt.param_groups: g['lr'] = lr * min(1, (st + 1) / 300) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.95 + 0.05)
            b = chunks[i]; loss = torch.nn.functional.cross_entropy(net(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); st += 1; flops += 6.0 * macs * bs * seq
            if st % every == 0 or st == steps:
                bb = evaluate(); curve.append((st, float('%.3e' % flops), round(bb, 4))); log('step %d FLOP %.2e bit/元の語 %.4f' % (st, flops, bb))
    res = dict(cfg=cfg, tokens_per_word=len(trt) / tot, curve=curve, final_bits_per_word=curve[-1][2], flops=flops, words_read=st * bs * seq * tot / len(trt), sec=time.time() - t0)
    log('RESULT ' + json.dumps(res))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
