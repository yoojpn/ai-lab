"""文書に 既に出た語 (キャッシュ) を 対数で足す。 全語彙で厳密に正規化。 数え上げのみ (勾配なし)。
  s(w) = sum_f  l_{f, 頻度の段(w)} * log(1 + n_f(w))     n_f(w) = 文書の前の部分に w が出た回数 (f = 遠い窓 / 近い窓 / 指数で減る重み)。 頻度の段 = w の 全体での出現頻度 (6 段)
  P'(w) = P(w) exp(s(w)) / Z ,  l は 調整位置の損失だけで 座標ごとに決める。
使い方: python3 cache_poe.py <W> <WORK (表と E.npy がある所)> <S>"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); t0 = time.time(); V = E.VOC; log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
tok = np.fromfile(WORK + '/sub.bin', dtype=np.uint16); n_tok = len(tok) - 8
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); docs = np.load(W + '/val_docs.npy') + 64
pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
idx_tu, idx_te = pos(123, 1500), pos(777, 4000)
M = E.Model({n: WORK + '/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}, kn_base=False)
def dense_pa(idx):
    out = np.zeros((len(idx), V), np.float32); hi_ = np.ones(len(idx), int)
    for r, i in enumerate(idx):
        P = M.uni_p.copy(); ii = np.array([i])
        for n in M.orders:
            if n == 1: continue
            h = E.ctx_hash(pad, ii, n - 1, n)[0]; t = M.t[n]; j = min(np.searchsorted(t['uc'], h), len(t['uc']) - 1)
            if t['uc'][j] != h: continue
            hi_[r] = n; N = float(t['N'][j]); T = float(t['T'][j]); X = float(t['X'][j]); lo = np.searchsorted(t['k'], h << np.uint64(16)); hi = np.searchsorted(t['k'], (h + np.uint64(1)) << np.uint64(16))
            ws = (np.asarray(t['k'][lo:hi]) & np.uint64(0xFFFF)).astype(np.int64); cs = np.asarray(t['c'][lo:hi]).astype(np.float64); lm = M.lam[n]
            cnt = np.zeros(V); cnt[ws] = cs; P = (cnt + lm * (T + X) * P) / max(N - X + lm * (T + X), 1e-9)
        out[r] = P
    return out, hi_
Pt, hit = dense_pa(idx_tu); Pe, hie = dense_pa(idx_te); tr_t, tr_e = pad[idx_tu], pad[idx_te]
base_t = -np.log2(Pt[np.arange(len(idx_tu)), tr_t]).mean(); base_e = -np.log2(Pe[np.arange(len(idx_te)), tr_e]).mean(); print('RESULT n-gram のみ tune %.4f / test %.4f bit/語' % (base_t, base_e), flush=True)
uni = np.bincount(tok[8:], minlength=V).astype(np.float64); uni /= uni.sum(); FB = np.digitize(uni, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]); NB = 6      # 頻度の段
def cache_feats(idx):
    """各位置の 3 つの特徴 [位置, 3, V]: 遠い窓 (文書の頭から 最大 3000 語) の回数 / 近い窓 (直前 48 語) の回数 / 指数で減る重み (半減 200 語)"""
    n = len(idx); F = np.zeros((3, n, V), np.float32)
    for r, i in enumerate(idx):
        ds = docs[np.searchsorted(docs, i, side='right') - 1]; lo = max(ds, i - 3000); seg = pad[lo:i].astype(np.int64)
        if len(seg) == 0: continue
        F[0, r] = np.bincount(seg, minlength=V); near = seg[-48:]; F[1, r] = np.bincount(near, minlength=V)
        wgt = 0.5 ** ((len(seg) - 1 - np.arange(len(seg))) / 200.0); F[2, r] = np.bincount(seg, weights=wgt, minlength=V)
    return np.log1p(F)
Ft = cache_feats(idx_tu); log('調整位置の特徴 完了'); NF = 3
LAM = np.zeros((NF, NB))
def bits(P, F, tr, lam):
    n = len(P); out = np.zeros(n)
    for c0 in range(0, n, 500):
        sl = slice(c0, min(c0 + 500, n)); s = np.zeros((sl.stop - sl.start, V), np.float32)
        for f in range(NF): s += F[f][sl] * lam[f][FB][None, :]
        e = P[sl] * np.exp(s - s.max(1, keepdims=True)); z = e.sum(1); out[sl] = -np.log2(e[np.arange(len(e)), tr[sl]] / z)
    return out.mean()
grid = [0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2]
for sweep in range(2):
    for f in range(NF):
        for b in range(NB):
            best = None
            for g in grid:
                LAM[f, b] = g; v = bits(Pt, Ft, tr_t, LAM)
                if best is None or v < best[0] - 1e-9: best = (v, g)
            LAM[f, b] = best[1]
    log('掃引 %d: tune %.4f' % (sweep + 1, best[0]))
del Ft; Fe = cache_feats(idx_te); be = bits(Pe, Fe, tr_e, LAM)
print('RESULT 文書のキャッシュを 対数で足す: test %.4f -> %.4f bit/語 (得 %.4f = %.2f%%)  重み (行=遠い/近い/指数, 列=頻度の段 低->高) %s' % (base_e, be, base_e - be, 100 * (base_e - be) / base_e, np.round(LAM, 2).tolist()), flush=True)
