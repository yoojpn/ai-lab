"""ハッシュ表で数える: 並べ替え(np.unique)方式と、実測時間と品質を比べる。
 sort : 次数1..4の (文脈,次語) を np.unique で数える(従来)。
 hash : 文脈と(文脈,次語)を64bit多項式ハッシュ→2^B個の表へ bincount(並べ替えなし)。
 品質は同じ式(加算平滑の補間 p_k=(c+a*p_{k-1})/(tot+a), a=0.5)で、hash版のみ測る(衝突の影響を含む)。
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lab
from lab import V, HIST, bits, log
import exp_scale as es
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)
M1 = np.uint64(0x9E3779B97F4A7C15)

def hctx(X, K):
    h = np.zeros(len(X), np.uint64)
    for p in range(1, K + 1):
        h = (h + X[:, HIST - p].astype(np.uint64) + np.uint64(1)) * M1
        h ^= h >> np.uint64(29)
    return h

def run(cfg):
    N = cfg['N']; B = cfg.get('B', 26); ctx = lab.load(N); N = len(ctx.Xtr); y = ctx.ytr
    res = dict(cfg=cfg, N=N)
    t = time.time()
    for K in range(1, 5):
        c = np.zeros(N, np.int64)
        for j in range(K): c += ctx.Xtr[:, HIST - 1 - j].astype(np.int64) * (V ** j)
        np.unique(c * V + y, return_counts=True); np.unique(c, return_counts=True)
    res['sec_sort'] = time.time() - t; log('sort 数え %.1fs' % res['sec_sort'])
    t = time.time(); S = 1 << B; mask = np.uint64(S - 1); tabs = []
    for K in range(1, 5):
        h = hctx(ctx.Xtr, K)
        tot = np.bincount((h & mask).astype(np.int64), minlength=S).astype(np.int32)
        hp = ((h + y.astype(np.uint64) + np.uint64(1)) * M1); hp ^= hp >> np.uint64(31)
        pr = np.bincount((hp & mask).astype(np.int64), minlength=S).astype(np.int32)
        tabs.append((tot, pr))
    res['sec_hash'] = time.time() - t; log('hash 数え %.1fs' % res['sec_hash'])
    pu = np.bincount(y, minlength=V) + 1.0; pu /= pu.sum()
    def prob(X, yy, a=0.5):
        p = pu[yy]
        for K in range(1, 5):
            tot, pr = tabs[K - 1]; h = hctx(X, K)
            hp = ((h + yy.astype(np.uint64) + np.uint64(1)) * M1); hp ^= hp >> np.uint64(31)
            ct = tot[(h & mask).astype(np.int64)]; cp = pr[(hp & mask).astype(np.int64)]
            p = (cp + a * p) / (ct + a)
        return p
    res['bits_hash'] = bits(prob(ctx.Xte[:20000], ctx.yte[:20000]))
    res['speedup'] = res['sec_sort'] / res['sec_hash']
    log('hash版 bit/語 %.4f | 速度比 %.1f倍' % (res['bits_hash'], res['speedup']))
    log('RESULT ' + json.dumps(res))
if __name__ == '__main__': run(json.loads(sys.argv[1]))
