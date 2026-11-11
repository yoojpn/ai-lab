"""次の語を「クラス」で先に予測する (階層分解)。 クラス = 語の埋め込み (PMI-SVD) を k-means。 クラス同士の n-gram は 統計が密なので 長い文脈まで使える。
  P'(w) = P_ngram(w) * (Pc(c(w)|文脈) / Q(c(w)))^l / Z    Q(c) = n-gram の確率の クラスごとの和,  Pc = クラスの n-gram (次数 1..M, Witten-Bell)。
  正規化は 全語彙で厳密。 l は 調整位置で選ぶ。 数え上げ + k-means のみ (勾配なし)。
使い方: python3 classlm.py <W> <WORK (skipsvd.py/boost.py の表と E.npy があれば流用)> <S> <クラス数=256> <M=8>"""
import sys, os, time, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 256; MM = int(sys.argv[5]) if len(sys.argv) > 5 else 8
os.makedirs(WORK, exist_ok=True); t0 = time.time(); V = E.VOC; log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
if not os.path.exists(WORK + '/sub.bin'): np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]).tofile(WORK + '/sub.bin')
tok = np.fromfile(WORK + '/sub.bin', dtype=np.uint16); n_tok = len(tok) - 8
for n in (1, 2, 3, 4, 5):
    if os.path.exists(WORK + '/final_o%d_k.bin' % n): continue
    runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], WORK + '/final_o%d' % n, 1 if n <= 2 else 2, 20_000_000, lambda s: None)
    for f in glob.glob(WORK + '/runs/*'): os.remove(f)
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
idx_tu, idx_te = pos(123, 1500), pos(777, 4000)
M = E.Model({n: WORK + '/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}, kn_base=False)
def dense_pa(idx):
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
Pt = dense_pa(idx_tu); Pe = dense_pa(idx_te); tr_t, tr_e = pad[idx_tu], pad[idx_te]
base_t = -np.log2(Pt[np.arange(len(idx_tu)), tr_t]).mean(); base_e = -np.log2(Pe[np.arange(len(idx_te)), tr_e]).mean(); print('RESULT S=%d n-gram のみ tune %.4f / test %.4f bit/語' % (n_tok, base_t, base_e), flush=True)
# ---- クラス ----
Em = np.load(WORK + '/E.npy') if os.path.exists(WORK + '/E.npy') else None
if Em is None:
    import scipy.sparse as sp, scipy.sparse.linalg as sl
    x = tok[8:].astype(np.int64); key = x[:-1] * V + x[1:]; u, cnt = np.unique(key, return_counts=True); del key
    Mx = sp.coo_matrix((cnt.astype(np.float64), (u // V, u % V)), shape=(V, V)).tocsr(); tot = Mx.sum(); rm = np.asarray(Mx.sum(1)).ravel() + 1e-9; cm = np.asarray(Mx.sum(0)).ravel() ** 0.75 + 1e-9; cm = cm / cm.sum() * tot
    Mx = Mx.tocoo(); pm = np.log(Mx.data * tot / (rm[Mx.row] * cm[Mx.col])); keep = pm > 0; Pm = sp.coo_matrix((pm[keep], (Mx.row[keep], Mx.col[keep])), shape=(V, V)).tocsr()
    U, sg, _ = sl.svds(Pm, k=48); Em = U * np.sqrt(sg)[None]; Em /= np.linalg.norm(Em, axis=1, keepdims=True) + 1e-9; Em = Em.astype(np.float32); np.save(WORK + '/E.npy', Em)
cf = WORK + '/cls%d.npy' % K
if os.path.exists(cf): CLS = np.load(cf)
else:
    from sklearn.cluster import MiniBatchKMeans
    uni = np.bincount(tok[8:], minlength=V).astype(np.float64); w = np.sqrt(uni + 1); CLS = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=1, random_state=0).fit(Em, sample_weight=w).labels_.astype(np.uint16); np.save(cf, CLS)
log('クラス %d 作成' % K); ct = CLS[tok]                                          # 学習側の クラス列
# ---- クラスの n-gram 表 (次数 1..MM) ----
C1 = np.uint64(0x9E3779B97F4A7C15); SH = np.uint64(29)
def mix(h, t): h = (h ^ (t.astype(np.uint64) + np.uint64(1))) * C1; return h ^ (h >> SH)
TAB = {}
for m in range(1, MM + 1):
    hi = len(ct) - 1; st = 8 + MM; ks = np.empty(hi - st, np.uint64)
    for a in range(st, hi, 2_000_000):
        b = min(a + 2_000_000, hi); h = np.full(b - a, np.uint64(777 + m), np.uint64)
        for k in range(m, 0, -1): h = mix(h, ct[a - k:b - k])
        ks[a - st:b - st] = ((h & np.uint64((1 << 48) - 1)) << np.uint64(16)) | ct[a:b].astype(np.uint64)
    ks.sort(); first = np.r_[True, ks[1:] != ks[:-1]]; uk = ks[first]; cn = np.diff(np.r_[np.flatnonzero(first), len(ks)]).astype(np.float32); del ks
    cx = uk >> np.uint64(16); fc = np.r_[True, cx[1:] != cx[:-1]]; ctxs = cx[fc]; starts_ = np.flatnonzero(fc); ends_ = np.r_[starts_[1:], len(uk)]
    TAB[m] = (uk, cn, ctxs, starts_, ends_); log('クラス n-gram 次数 %d: 文脈 %d, 組 %d' % (m, len(ctxs), len(uk)))
cu = np.bincount(ct[8:], minlength=K).astype(np.float64); cu = (cu + 1) / (cu + 1).sum()
def class_dist(idx, lam=2.0):
    """各位置の クラスの確率分布 [位置, K] (WB 補間, 次数 1..MM)"""
    out = np.tile(cu, (len(idx), 1)); cls_pad = CLS[pad]
    for m in range(1, MM + 1):
        uk, cn, ctxs, st_, en_ = TAB[m]; h = np.full(len(idx), np.uint64(777 + m), np.uint64)
        for k in range(m, 0, -1): h = mix(h, cls_pad[idx - k])
        h &= np.uint64((1 << 48) - 1); j = np.minimum(np.searchsorted(ctxs, h), len(ctxs) - 1); ok = ctxs[j] == h
        for r in np.flatnonzero(ok):
            a, b = st_[j[r]], en_[j[r]]; cnt = np.zeros(K); cnt[(uk[a:b] & np.uint64(0xFFFF)).astype(np.int64)] = cn[a:b]; N = cnt.sum(); T = b - a
            out[r] = (cnt + lam * T * out[r]) / (N + lam * T)
    return out
import scipy.sparse as sp
OH = sp.csr_matrix((np.ones(V), (np.arange(V), CLS.astype(np.int64))), shape=(V, K))
def prep(P, idx):
    Q = np.asarray(P @ OH); Pc = class_dist(idx); return Q, Pc
Qt, Pct = prep(Pt, idx_tu); Qe, Pce = prep(Pe, idx_te); log('クラス分布を計算')
def bits(P, Q, Pc, tr, lam):
    r = (Pc / np.maximum(Q, 1e-30)) ** lam[:, None] if np.ndim(lam) else (Pc / np.maximum(Q, 1e-30)) ** lam
    Z = (Q * r).sum(1); ct_ = CLS[tr].astype(np.int64); ar = np.arange(len(P)); return -np.log2(P[ar, tr] * r[ar, ct_] / Z)
best = None
for l in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    v = bits(Pt, Qt, Pct, tr_t, l).mean()
    if best is None or v < best[0]: best = (v, l)
    print('  l=%.1f tune %.4f' % (l, v), flush=True)
l = best[1]; be = bits(Pe, Qe, Pce, tr_e, l).mean()
print('RESULT S=%d クラス %d (次数 1..%d) で次の語のクラスを予測して 置き換える: test %.4f -> %.4f bit/語 (得 %.4f = %.2f%%) l=%.1f' % (n_tok, K, MM, base_e, be, base_e - be, 100 * (base_e - be) / base_e, l), flush=True)
