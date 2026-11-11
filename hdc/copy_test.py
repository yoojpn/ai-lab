"""窓の外の情報の下限を測る: 文書内の最長一致の続き (in-context copy) を n-gram に足す。 数え上げだけ。
P(w) = p * [w == 続き] + (1-p) * P_ngram(w)。 p は (一致の長さ, n-gram が当たった最高次数) ごとに tune 位置で数えた 的中率。
使い方: python3 copy_test.py <W> <WORK> <S>   (窓 L = 200, 2000, 20000)"""
import sys, os, time, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(WORK, exist_ok=True); t0 = time.time(); log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
sub = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); sub.tofile(WORK + '/sub.bin'); n_tok = len(sub) - 8
for n in (1, 2, 3, 4, 5):
    if os.path.exists(WORK + '/final_o%d_k.bin' % n): continue
    runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], WORK + '/final_o%d' % n, 1 if n <= 2 else 2, 20_000_000, lambda s: None)
    for f in glob.glob(WORK + '/runs/*'): os.remove(f)
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val])
pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
idx_te, idx_tu = pos(777, 20000), pos(123, 6000)
M = E.Model({n: WORK + '/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}, kn_base=False)
R = {}
for key, idx in (('te', idx_te), ('tu', idx_tu)):
    lp, hi = M.logprob_at(pad, idx, return_hi=True); R[key] = (np.exp2(lp), hi)
base = -np.log2(R['te'][0]).mean(); print('RESULT S=%d n-gram %.4f' % (n_tok, base), flush=True)
def longest(idx, L, maxm=32):
    """各位置 t について、 直前 L 語の中で t の直前の並びと 最も長く一致する場所の 続きの語と 一致の長さ"""
    cont = np.zeros(len(idx), np.int64); mlen = np.zeros(len(idx), np.int64); p = pad.astype(np.int64)
    for i, t in enumerate(idx):
        lo = max(64, t - L); seg = p[lo:t - 1]; c = np.flatnonzero(seg == p[t - 1]) + lo
        if len(c) == 0: continue
        m = np.ones(len(c), np.int64); alive = np.ones(len(c), bool)
        for k in range(1, maxm):
            ok = alive & (c - k >= lo) & (p[np.maximum(c - k, 0)] == p[t - 1 - k]); m[ok] += 1; alive = ok
            if not alive.any(): break
        j = len(c) - 1 - np.argmax(m[::-1]); cont[i] = p[c[j] + 1]; mlen[i] = m[j]
    return cont, mlen
for L in (200, 2000, 20000):
    D = {}
    for key, idx in (('te', idx_te), ('tu', idx_tu)): D[key] = longest(idx, L)
    mb = lambda m: np.minimum(m, 6)
    tab = np.zeros((7, 6)); cnt = np.zeros((7, 6)); c, m = D['tu']; hit = (c == pad[idx_tu]) & (m > 0)
    np.add.at(tab, (mb(m), R['tu'][1]), hit); np.add.at(cnt, (mb(m), R['tu'][1]), 1); pt = (tab + 0.5) / (cnt + 2.0); pt[0] = 0
    c, m = D['te']; p = pt[mb(m), R['te'][1]]; hitte = (c == pad[idx_te]) & (m > 0); P = p * hitte + (1 - p) * R['te'][0]; b = -np.log2(P).mean()
    print('RESULT S=%d 窓 %5d: n-gram+文書内の一致 %.4f (得 %.4f) | 一致あり %.3f, 続きの的中率 %.3f, 長さ>=3 の割合 %.3f その的中率 %.3f' % (n_tok, L, b, base - b, (m > 0).mean(), hitte.mean(), (m >= 3).mean(), hitte[m >= 3].mean() if (m >= 3).any() else 0), flush=True)
