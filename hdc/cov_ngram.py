import sys; sys.path.insert(0, '/home/claude/ai-lab/hdc')
import numpy as np, engine as E
D = sys.argv[1].rstrip('/') + '/'   # final_o*_k.bin, final_o*_c.bin, val.bin を置いた場所
val = np.fromfile(D + 'val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(8, np.uint16), val])
N = 6000; rng = np.random.default_rng(3); idx = rng.integers(8, len(val) - 1, N) + 8; y = pad[idx].astype(np.int64)
T = {n: (np.memmap(D + 'final_o%d_k.bin' % n, dtype=np.uint64, mode='r'), np.memmap(D + 'final_o%d_c.bin' % n, dtype=np.uint32, mode='r')) for n in (1, 2, 3, 4, 5)}
k1, c1 = T[1]; uni_tok = (np.asarray(k1) & np.uint64(0xFFFF)).astype(np.int64); uni_order = uni_tok[np.argsort(-np.asarray(c1).astype(np.int64), kind='stable')]
rng_ = {}
for n in (5, 4, 3, 2):
    k, c = T[n]; h = E.ctx_hash(pad, idx, n - 1, n); lo = np.searchsorted(k, h << np.uint64(16)); hi = np.searchsorted(k, (h + np.uint64(1)) << np.uint64(16)); rng_[n] = (lo, hi)
rank = np.full(N, 10**9); hiord = np.ones(N, int); lens = np.zeros(N, int)
for i in range(N):
    cand = []; seen = set()
    for n in (5, 4, 3, 2):
        lo, hi = rng_[n][0][i], rng_[n][1][i]
        if hi > lo:
            if hiord[i] == 1: hiord[i] = n
            k, c = T[n]; tk = (np.asarray(k[lo:hi]) & np.uint64(0xFFFF)).astype(np.int64); cc = np.asarray(c[lo:hi]).astype(np.int64)
            for t in tk[np.argsort(-cc, kind='stable')]:
                if t not in seen: seen.add(t); cand.append(t)
    lens[i] = len(cand)
    for t in uni_order[:2048]:
        if t not in seen: seen.add(t); cand.append(t)
    try: rank[i] = cand.index(y[i])
    except ValueError: pass
print('N=%d  候補(n-gram由来)の平均個数 %.1f  中央値 %d' % (N, lens.mean(), np.median(lens)))
for K in (1, 4, 16, 64, 256, 1024): print('被覆率@%-5d 全体 %.1f%%' % (K, 100 * (rank < K).mean()))
print('最高次数ごと (割合 / 被覆率@16 / @64 / @256):')
for n in (1, 2, 3, 4, 5):
    m = hiord == n
    if m.any(): print('  次数%d: %.1f%%  %.1f%% / %.1f%% / %.1f%%' % (n, 100 * m.mean(), 100 * (rank[m] < 16).mean(), 100 * (rank[m] < 64).mean(), 100 * (rank[m] < 256).mean()))
