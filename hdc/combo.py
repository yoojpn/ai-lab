"""(統合版: 文書の特徴 (キャッシュ/意味/帰納) + クラスの 2 専門家 (h64, h256) を 1 つの対数モデルに。 位置ごとの bit を保存する)
(第 3 版: 文書内の帰納 = 今の文脈 (直前 1〜2 語 / そのクラス) と同じ並びが 文書の前の方にあれば その続きの語を足す。 第 2 版: 重みの範囲を広げ、 文書の埋め込みの平均との近さ (意味の近い語) の 特徴を 足す)
文書に 既に出た語 (キャッシュ) を 対数で足す。 全語彙で厳密に正規化。 数え上げのみ (勾配なし)。
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
idx_tu, idx_te = pos(123, 1200), pos(777, 4000)
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
Em = np.load(WORK + '/E.npy'); IDFw = np.log(1.0 / np.maximum(uni, 1e-9)).astype(np.float32); IDFw /= IDFw.max()
def sim_feats(idx):
    """文書の埋め込みの平均 (希少な語ほど重い) と 各語の埋め込みの 内積。 遠い窓 (文書の頭から) / 近い窓 (直前 100 語)。 [2, 位置, V]"""
    out = np.zeros((2, len(idx), V), np.float32)
    for r, i in enumerate(idx):
        ds = docs[np.searchsorted(docs, i, side='right') - 1]; lo = max(ds, i - 3000); seg = pad[lo:i].astype(np.int64)
        if len(seg) == 0: continue
        for f, sg in enumerate((seg, seg[-100:])):
            v = (Em[sg] * IDFw[sg][:, None]).sum(0); nv = np.linalg.norm(v)
            if nv > 0: out[f, r] = (Em @ (v / nv)).astype(np.float32)
    return out
from sklearn.cluster import MiniBatchKMeans
_cf = WORK + '/cls64.npy'
if os.path.exists(_cf): CL64 = np.load(_cf)
else: CL64 = MiniBatchKMeans(n_clusters=64, batch_size=4096, n_init=1, random_state=0).fit(Em, sample_weight=np.sqrt(uni * 1e9 + 1)).labels_.astype(np.uint16); np.save(_cf, CL64)
def induct_feats(idx):
    """文書内の帰納 [4, 位置, V]: (a) 直前 1 語が同じ (b) 直前 2 語が同じ (c) 直前 2 語の クラスが同じ (d) 直前 3 語の クラスが同じ。 同じ並びの 後に出た語の回数 (log1p)。 窓は 文書の頭から 最大 3000 語"""
    out = np.zeros((4, len(idx), V), np.float32)
    for r, i in enumerate(idx):
        ds = docs[np.searchsorted(docs, i, side='right') - 1]; lo = max(ds, i - 3000); seg = pad[lo:i].astype(np.int64); n = len(seg)
        if n < 4: continue
        cs = CL64[seg].astype(np.int64); nxt = seg[3:]                                    # 位置 j (>= 3) の 語 seg[j] と その前の 3 語
        m1 = seg[2:-1] == seg[-1]; m2 = m1 & (seg[1:-2] == seg[-2]); c2 = (cs[2:-1] == cs[-1]) & (cs[1:-2] == cs[-2]); c3 = c2 & (cs[:-3] == cs[-3])
        t1 = seg[3:]
        for f, m in enumerate((m1, m2, c2, c3)):
            if m.any(): out[f, r] = np.bincount(t1[m], minlength=V)
    return np.log1p(out)
def all_feats(idx): return np.concatenate([cache_feats(idx), sim_feats(idx), induct_feats(idx)], 0)
Ft = all_feats(idx_tu); log('調整位置の特徴 完了'); NF = 9

import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
_cls = {}
def get_cls(K):
    if K in _cls: return _cls[K]
    cf = WORK + '/cls%d.npy' % K
    if os.path.exists(cf): c = np.load(cf)
    else:
        uni = np.bincount(tok[8:], minlength=V).astype(np.float64); c = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=1, random_state=0).fit(Em, sample_weight=np.sqrt(uni + 1)).labels_.astype(np.uint16); np.save(cf, c)
    _cls[K] = c; return c
C1 = np.uint64(0x9E3779B97F4A7C15); SH = np.uint64(29); M48 = np.uint64((1 << 48) - 1)
def mix(h, t): h = (h ^ (t.astype(np.uint64) + np.uint64(1))) * C1; return h ^ (h >> SH)
def elem(arr, kind, Kc, a, b, k):
    v = arr[a - k:b - k]; return get_cls(Kc)[v] + np.uint16(40000) if kind == 'c' else v
def build_tab(Kt, Kc, spec, seed):
    ct = get_cls(Kt)[tok]; hi = len(tok) - 1; st = 8 + 16; ks = np.empty(hi - st, np.uint64)
    for a in range(st, hi, 2_000_000):
        b = min(a + 2_000_000, hi); h = np.full(b - a, np.uint64(seed), np.uint64)
        for k, kind in sorted(spec, key=lambda z: -z[0]): h = mix(h, elem(tok, kind, Kc, a, b, k))
        ks[a - st:b - st] = ((h & M48) << np.uint64(16)) | ct[a:b].astype(np.uint64)
    ks.sort(); first = np.r_[True, ks[1:] != ks[:-1]]; uk = ks[first]; cn = np.diff(np.r_[np.flatnonzero(first), len(ks)]).astype(np.float32); del ks
    cx = uk >> np.uint64(16); fc = np.r_[True, cx[1:] != cx[:-1]]; ctxs = cx[fc]; s_ = np.flatnonzero(fc); e_ = np.r_[s_[1:], len(uk)]; return uk, cn, ctxs, s_, e_
def class_dist(idx, Kt, Kc, chain, seed0, lam=2.0):
    """鎖に沿って 補間した クラスの確率 [位置, Kt] と 鎖の どこまで見つかったか [位置]"""
    ct = get_cls(Kt); cu = np.bincount(ct[tok[8:]], minlength=Kt).astype(np.float64); cu = (cu + 1) / (cu + 1).sum(); out = np.tile(cu, (len(idx), 1)); found = np.zeros(len(idx), int)
    for li, spec in enumerate(chain, 1):
        uk, cn, ctxs, s_, e_ = build_tab(Kt, Kc, spec, seed0 + li); h = np.full(len(idx), np.uint64(seed0 + li), np.uint64)
        for k, kind in sorted(spec, key=lambda z: -z[0]): h = mix(h, (get_cls(Kc)[pad[idx - k]] + np.uint16(40000)) if kind == 'c' else pad[idx - k])
        h &= M48; j = np.minimum(np.searchsorted(ctxs, h), len(ctxs) - 1); ok = ctxs[j] == h
        for r in np.flatnonzero(ok):
            a, b = s_[j[r]], e_[j[r]]; cnt = np.zeros(Kt); cnt[(uk[a:b] & np.uint64(0xFFFF)).astype(np.int64)] = cn[a:b]; N = cnt.sum(); T = b - a; out[r] = (cnt + lam * T * out[r]) / (N + lam * T); found[r] = li
        del uk, cn, ctxs, s_, e_
    return out, found
def cchain(m): return [[(k, 'c') for k in range(1, i + 1)] for i in range(1, m + 1)]
def hchain(m): return [[(1, 'w')] + [(k, 'c') for k in range(2, i + 1)] for i in range(1, m + 1)]       # 直前の 1 語は 語そのもの + その前は クラス
def h2chain(m): return [[(1, 'w'), (2, 'w')] + [(k, 'c') for k in range(3, i + 1)] for i in range(2, m + 1)]

# ---- クラスの専門家 2 つ (h64: 目標 64 クラス 6 段, h256: 目標 256 クラス 4 段) の 対数比 ----
CLE = [(64, 64, hchain(6)), (256, 256, hchain(4))]; CLD = []
for ei, (Kt, Kc, chain) in enumerate(CLE):
    ct = get_cls(Kt); OH = sp.csr_matrix((np.ones(V), (np.arange(V), ct.astype(np.int64))), shape=(V, Kt)); d = {}
    for nm, P, idx in (('tu', Pt, idx_tu), ('te', Pe, idx_te)):
        Q = np.asarray(P @ OH); Pc, fd = class_dist(idx, Kt, Kc, chain, 1000 * (ei + 1)); d[nm] = np.log(np.maximum(Pc, 1e-30)) - np.log(np.maximum(Q, 1e-30))
    CLD.append((ct, d)); log('クラス専門家 %d 完了' % ei)
Ft = all_feats(idx_tu); log('調整位置の特徴 完了')
NB4 = np.array([0, 0, 1, 1, 2, 3])                                        # 頻度の 6 段 -> 4 段 (低頻度 2 段を まとめる、 高頻度 2 段を 分ける)
LAM = np.zeros((NF, 4)); LC = np.zeros(len(CLE))
def bits_all(P, F, nm, sl, tr, lam, lc):
    s = np.zeros((sl.stop - sl.start, V), np.float32)
    for f in range(NF): s += F[f] * lam[f][NB4[FB]][None, :]
    for ei, (ct, d) in enumerate(CLD):
        if lc[ei]: s += lc[ei] * d[nm][sl][:, ct]
    e = P[sl] * np.exp(s - s.max(1, keepdims=True)); z = e.sum(1); return -np.log2(e[np.arange(len(e)), tr[sl]] / z)
def tune_bits(lam, lc): return bits_all(Pt, Ft, 'tu', slice(0, len(idx_tu)), tr_t, lam, lc).mean()
grid = [0, 0.15, 0.3, 0.6, 1.0, 1.6, 2.4, 3.5, 5.0, 7.5]
SKIPFIT = os.environ.get('SKIPFIT') == '1'
if SKIPFIT: LAM[:] = np.load(WORK + '/combo_lam.npy'); LC[:] = [0.2, 0.3]
for sweep in range(0 if SKIPFIT else 2):
    for ei in range(len(CLE)):
        best = None
        for g in [0, 0.1, 0.2, 0.3, 0.45, 0.6]:
            LC[ei] = g; v = tune_bits(LAM, LC)
            if best is None or v < best[0] - 1e-9: best = (v, g)
        LC[ei] = best[1]
    for f in range(NF):
        for b in range(4):
            best = None
            for g in grid:
                LAM[f, b] = g; v = tune_bits(LAM, LC)
                if best is None or v < best[0] - 1e-9: best = (v, g)
            LAM[f, b] = best[1]
    log('掃引 %d: tune %.4f  クラスの重み %s' % (sweep + 1, best[0], LC.tolist())); np.save(WORK + '/combo_lam.npy', LAM)
def score_mat(F, nm, sl, lam, lc):
    s = np.zeros((sl.stop - sl.start, V), np.float32)
    for f in range(NF): s += F[f] * lam[f][NB4[FB]][None, :]
    for ei, (ct, d) in enumerate(CLD):
        if lc[ei]: s += lc[ei] * d[nm][sl][:, ct]
    return s
if SKIPFIT:
    np.save(WORK + '/S_tu.npy', score_mat(Ft, 'tu', slice(0, len(idx_tu)), LAM, LC)); log('調整位置のスコア行列を保存')
    S_te = np.zeros((len(idx_te), V), np.float32)
    for c0 in range(0, len(idx_te), 400):
        sl = slice(c0, min(c0 + 400, len(idx_te))); Fc = all_feats(idx_te[sl]); S_te[sl] = score_mat(Fc, 'te', sl, LAM, LC); del Fc
    np.save(WORK + '/S_te.npy', S_te); log('テストのスコア行列を保存'); sys.exit(0)
del Ft; out = np.zeros(len(idx_te)); out0 = -np.log2(Pe[np.arange(len(idx_te)), tr_e])
for c0 in range(0, len(idx_te), 400):
    sl = slice(c0, min(c0 + 400, len(idx_te))); Fc = all_feats(idx_te[sl]); out[sl] = bits_all(Pe, Fc, 'te', sl, tr_e, LAM, LC); del Fc
np.save('/tmp/combo_te_bits.npy', out); np.save('/tmp/ngram_te_bits.npy', out0); np.save('/tmp/te_hi.npy', hie)
print('RESULT 統合 (文書の特徴 + クラス 2 専門家): test %.4f -> %.4f bit/語 (得 %.4f = %.2f%%)  クラスの重み %s  文書の特徴の重み %s' % (out0.mean(), out.mean(), out0.mean() - out.mean(), 100 * (out0.mean() - out.mean()) / out0.mean(), LC.tolist(), np.round(LAM, 2).tolist()), flush=True)
