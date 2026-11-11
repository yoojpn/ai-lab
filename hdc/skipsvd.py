"""距離ごとの PMI を 特異値分解した 低ランクの連想 (数え上げ + 線形代数、 勾配なし) を n-gram に対数で足す。 正規化は 全語彙 (32768) で厳密。
  距離 k ごとに  PMI_k(a, w) = log P(w | k 語前が a) / P(w)  を 数えて  u_k[a] . f_k[w] に 低ランク近似。
  P'(w) = P_ngram(w) exp( sum_k l_k u_k[w_{-k}] . f_k[w] ) / Z      l_k は 調整位置の 損失だけで 座標ごとに決める。
  n-gram の P_ngram(w) は 表から 全語彙分を 厳密に 作る (Model.logprob_at と 同じ計算を 密な配列で)。
使い方: python3 skipsvd.py <W> <WORK (boost.py の 表が あれば 流用)> <S> [ランク=48]"""
import sys, os, time, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); RK = int(sys.argv[4]) if len(sys.argv) > 4 else 48; os.makedirs(WORK, exist_ok=True); t0 = time.time(); V = E.VOC
log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
if not os.path.exists(WORK + '/sub.bin'):
    np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]).tofile(WORK + '/sub.bin')
tok = np.fromfile(WORK + '/sub.bin', dtype=np.uint16); n_tok = len(tok) - 8
for n in (1, 2, 3, 4, 5):
    if os.path.exists(WORK + '/final_o%d_k.bin' % n): continue
    runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], WORK + '/final_o%d' % n, 1 if n <= 2 else 2, 20_000_000, lambda s: None)
    for f in glob.glob(WORK + '/runs/*'): os.remove(f)
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
idx_tu, idx_te = pos(123, 1500), pos(777, 4000)
LAMB = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model({n: WORK + '/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=LAMB, kn_base=False)
def dense_pa(idx):
    """各位置の n-gram の確率を 全語彙で (厳密)。 [位置, V] の float32"""
    out = np.zeros((len(idx), V), np.float32)
    for r, i in enumerate(idx):
        P = M.uni_p.copy(); ii = np.array([i])
        for n in M.orders:
            if n == 1: continue
            h = E.ctx_hash(pad, ii, n - 1, n)[0]; t = M.t[n]; j = min(np.searchsorted(t['uc'], h), len(t['uc']) - 1)
            if t['uc'][j] != h: continue
            N = float(t['N'][j]); T = float(t['T'][j]); X = float(t['X'][j]); lo = np.searchsorted(t['k'], h << np.uint64(16)); hi = np.searchsorted(t['k'], (h + np.uint64(1)) << np.uint64(16))
            ws = (np.asarray(t['k'][lo:hi]) & np.uint64(0xFFFF)).astype(np.int64); cs = np.asarray(t['c'][lo:hi]).astype(np.float64); lm = M.lam[n]
            cnt = np.zeros(V); cnt[ws] = cs; P = (cnt + lm * (T + X) * P) / max(N - X + lm * (T + X), 1e-9)
        out[r] = P
    return out
Pt = dense_pa(idx_tu); Pe = dense_pa(idx_te); chk = np.exp2(M.logprob_at(pad, idx_te[:200])); err = np.abs(Pe[np.arange(200), pad[idx_te[:200]]] - chk).max(); log('密な確率の検算: 最大差 %.2e, 和 %.4f' % (err, Pe[0].sum()))
base_t = -np.log2(Pt[np.arange(len(idx_tu)), pad[idx_tu]]).mean(); base_e = -np.log2(Pe[np.arange(len(idx_te)), pad[idx_te]]).mean(); print('RESULT S=%d n-gram のみ tune %.4f / test %.4f bit/語' % (n_tok, base_t, base_e), flush=True)
# ---- 距離 k ごとの PMI -> 低ランク ----
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]; UF = {}; x = tok[8:].astype(np.int64); N = len(x)
ca = np.bincount(x, minlength=V).astype(np.float64) + 1e-9; cb = ca ** 0.75; cb = cb / cb.sum()
for k in KS:
    fp = WORK + '/svd_k%d_r%d.npz' % (k, RK)
    if os.path.exists(fp): z = np.load(fp); UF[k] = (z['u'], z['f']); continue
    key = x[:-k] * V + x[k:]; u_, cnt = np.unique(key, return_counts=True); del key                       # 前の語 a = x[t], 次の語 w = x[t+k]
    a = u_ // V; w = u_ % V; pm = np.log(cnt * (N - k) / (ca[a] * cb[w] * (N - k))); keep = pm > 0
    Pm = sp.coo_matrix((pm[keep], (a[keep], w[keep])), shape=(V, V)).tocsr(); U, s, Vt = randomized_svd(Pm, RK, n_iter=3, random_state=0)
    u = (U * np.sqrt(s)[None]).astype(np.float32); f = (Vt.T * np.sqrt(s)[None]).astype(np.float32); np.savez(fp, u=u, f=f); UF[k] = (u, f); log('距離 %d の SVD 完了 (非零 %d)' % (k, keep.sum()))
def scores(idx):
    """[位置, 距離の数, V] ではなく 距離ごとに [位置, V] を 返す (メモリ: 位置 x V x 4 バイト)"""
    return {k: (UF[k][0][pad[idx - k]] @ UF[k][1].T).astype(np.float32) for k in KS}
St = scores(idx_tu); log('調整位置の連想を計算'); 
tr_t = pad[idx_tu]; ar_t = np.arange(len(idx_tu))
def bits(P, Sk, lam, tr):
    s = np.zeros_like(P)
    for k in KS:
        if lam[k]: s += lam[k] * Sk[k]
    e = P * np.exp(s - s.max(1, keepdims=True)); z = e.sum(1); return -np.log2(e[np.arange(len(P)), tr] / z)
lam = {k: 0.0 for k in KS}; grid = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]; cur = base_t
for sweep in range(2):
    for k in KS:
        best = None
        for g in grid:
            lam[k] = g; v = bits(Pt, St, lam, tr_t).mean()
            if best is None or v < best[0] - 1e-9: best = (v, g)
        lam[k] = best[1]; cur = best[0]
    log('掃引 %d: tune %.4f  重み %s' % (sweep + 1, cur, {k: lam[k] for k in KS}))
del St; tot = 0.0
for c0 in range(0, len(idx_te), 500):
    sl = slice(c0, c0 + 500); Sc = scores(idx_te[sl]); tot += bits(Pe[sl], Sc, lam, pad[idx_te[sl]]).sum(); del Sc
be = tot / len(idx_te)
print('RESULT S=%d 連想 (SVD ランク %d) を 対数で足す: test %.4f -> %.4f bit/語 (得 %.4f = %.2f%%)  重み %s' % (n_tok, RK, base_e, be, base_e - be, 100 * (base_e - be) / base_e, {k: lam[k] for k in KS}), flush=True)
