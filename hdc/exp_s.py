"""
S: 索引(数える表)のサイズで質は伸びるか。Transformerを固定して、索引だけ大きくする(学習FLOPゼロ)。
 tf_runs : [[学習語数, エポック], ...]  各Transformerを1回ずつ学習(d128, 4層, 文脈64語, lr4e-3)
 stores  : 索引に使う語数のリスト。数える側 = 文脈0〜3語のバックオフ(4-gram)。
 出力: (TF, 索引サイズ)ごとの bit/語(混ぜた結果, 数えるだけ, TFだけ)。TFの学習FLOP。
       頻度帯ごとの bit/語と全体に占める割合(分業の測定): 数える/TF/混ぜる の3つ。
使い方: LADDER_CORPUS=... python3 exp_s.py '{"N":40000000,"tf_runs":[[16000000,3],[40000000,3]],"stores":[4000000,16000000,40000000]}'
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, bits, log
import exp_scale as es
from exp_t3 import split_windows
from exp_t4 import GT, macs_per_token
from exp_g import Chain, mix_bucket
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)
BANDS = [(1, 50), (51, 200), (201, 1000), (1001, 3000), (3001, 4999), (0, 0)]


def train_tf(stream, d, nl, seq, epochs, lr, bs=128):
    S = seq + 1; M = len(stream) // S; chunks = torch.from_numpy(stream[:M * S].reshape(M, S).astype(np.int64))
    net = GT(d, nl, S); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    steps = int(epochs * (M // bs)); st = 0; t0 = time.time()
    while st < steps:
        for i in torch.randperm(M).split(bs):
            if len(i) < bs or st >= steps: break
            for g in opt.param_groups:
                g['lr'] = lr * min(1, (st + 1) / 300) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.95 + 0.05)
            b = chunks[i]
            loss = torch.nn.functional.cross_entropy(net(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
            if st % 500 == 0: log('  TF step %d/%d %.0fs' % (st, steps, time.time() - t0))
    net.eval(); return net, 6.0 * macs_per_token(d, nl, seq) * steps * bs * seq


def tf_probs(net, X, y):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            lp = torch.log_softmax(net(torch.from_numpy(X[i:i + 1024]), last_only=True), 1)
            out.append(lp[torch.arange(len(lp)), torch.from_numpy(y[i:i + 1024])].exp().numpy())
    return np.concatenate(out).astype(np.float64)


def band_table(y, plist, names, ctx):
    rows = {}
    tot = {n: float((-np.log2(p + 1e-12)).sum()) for n, p in zip(names, plist)}
    for lo, hi in BANDS:
        m = (y == 0) if lo == 0 else ((y >= lo) & (y <= hi))
        if not m.any(): continue
        r = dict(frac_tokens=float(m.mean()))
        for n, p in zip(names, plist):
            b = -np.log2(p[m] + 1e-12); r[n + '_bits'] = float(b.mean()); r[n + '_share'] = float(b.sum() / tot[n])
        rows['%d-%d' % (lo, hi)] = r
    return rows


def run(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0); t0 = time.time()
    ctx = lab.load(cfg['N']); d, nl, seq = cfg.get('d', 128), cfg.get('nl', 4), cfg.get('seq', 64)
    n = min(20000, len(ctx.yva), len(ctx.yte)); yva, yte = ctx.yva[:n], ctx.yte[:n]; Xva, Xte = ctx.Xva[:n], ctx.Xte[:n]
    Lva = split_windows(ctx.va_i, ctx.mva, ctx.pva, n, seq); Lte = split_windows(ctx.te_i, ctx.mte, ctx.pte, n, seq)
    res = dict(cfg=cfg, tf={}, mix={}, count={})
    levels = [tuple(range(1, k + 1)) for k in range(4)]
    cp = {}
    for S in cfg['stores']:
        ch = Chain(ctx.Xtr[:S], ctx.ytr[:S], levels, 0.9)
        pv, mkv = ch.probs(Xva, yva); pt, mkt = ch.probs(Xte, yte); cp[S] = (pv, mkv, pt, mkt)
        res['count'][str(S)] = bits(pt); log('索引 %d語: 数えるだけ %.4f [%.0fs]' % (S, bits(pt), time.time() - t0)); del ch
    for tw, ep in cfg['tf_runs']:
        key = '%d_ep%s' % (tw, ep)
        stream = np.concatenate(ctx.tr_i)[:tw]
        net, fl = train_tf(stream, d, nl, seq, ep, cfg.get('lr', 4e-3))
        qv, qt = tf_probs(net, Lva, yva), tf_probs(net, Lte, yte)
        res['tf'][key] = dict(bits=bits(qt), flops=fl); log('TF %s: %.4f FLOP %.2e [%.0fs]' % (key, bits(qt), fl, time.time() - t0))
        for S in cfg['stores']:
            pv, mkv, pt, mkt = cp[S]
            mix = mix_bucket(np.array([pv, qv]), np.array([pt, qt]), mkv, mkt)
            res['mix']['%s|%d' % (key, S)] = bits(mix); log('TF %s + 索引 %d語: 混ぜる %.4f' % (key, S, bits(mix)))
            if S == cfg['stores'][1] and tw == cfg['tf_runs'][0][0]:
                res['bands'] = band_table(yte, [pt, qt, mix], ['count', 'tf', 'mix'], ctx)
                log('頻度帯 ' + json.dumps(res['bands'], ensure_ascii=False))
    res['sec'] = time.time() - t0
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
