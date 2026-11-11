"""クラスで次の語を先に予測する (階層分解) の 多段・混成版。  数え上げ + k-means のみ (勾配なし)。 正規化は 全語彙で厳密。
 専門家 e = (目標のクラス数 Kt, 文脈のクラス数 Kc, 鎖)。 鎖 = 文脈の型の列 (小さい文脈 -> 大きい文脈)。 型の要素 = (距離, 'w' 語そのもの | 'c' クラス)。
 P'(w) = P(w) exp( sum_e l_e[状態] log(Pc_e(c_e(w)|文脈) / Q_e(c_e(w))) ) / Z     Q_e = n-gram の確率の クラスごとの和
 l_e は (n-gram の最高次数, 鎖の どこまで見つかったか) の 状態ごとに 調整位置の損失だけで 座標ごとに決める。
使い方: python3 classlm2.py <W> <WORK> <S> <実験名>"""
import sys, os, time, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S, EXP = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; t0 = time.time(); V = E.VOC; log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
tok = np.fromfile(WORK + '/sub.bin', dtype=np.uint16); n_tok = len(tok) - 8; Em = np.load(WORK + '/E.npy')
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
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
base_t = -np.log2(Pt[np.arange(len(idx_tu)), tr_t]).mean(); base_e = -np.log2(Pe[np.arange(len(idx_te)), tr_e]).mean(); print('RESULT %s n-gram のみ tune %.4f / test %.4f' % (EXP, base_t, base_e), flush=True)
from sklearn.cluster import MiniBatchKMeans
import scipy.sparse as sp
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
EXPS = {'c64': [(64, 64, cchain(6))], 'h64': [(64, 64, hchain(6))], 'c64h64': [(64, 64, cchain(6)), (64, 64, hchain(6))], 'c16c64c256': [(16, 16, cchain(8)), (64, 64, cchain(6)), (256, 256, cchain(4))],
        'c64h64h2': [(64, 64, cchain(6)), (64, 64, hchain(6)), (64, 64, h2chain(6))], 'cross': [(64, 16, cchain(8)), (64, 256, cchain(4))]}
EXPS.update({'c256': [(256, 256, cchain(4))], 'h256': [(256, 256, hchain(4))], 'h128': [(128, 128, hchain(5))], 'c1024': [(1024, 1024, cchain(3))], 'h1024': [(1024, 1024, hchain(3))],
             'h64h256': [(64, 64, hchain(6)), (256, 256, hchain(4))], 'h16h64h256': [(16, 16, hchain(6)), (64, 64, hchain(6)), (256, 256, hchain(4))], 'h64h256h1024': [(64, 64, hchain(6)), (256, 256, hchain(4)), (1024, 1024, hchain(3))]})
experts = EXPS[EXP]; DATA = []
for ei, (Kt, Kc, chain) in enumerate(experts):
    ct = get_cls(Kt); OH = sp.csr_matrix((np.ones(V), (np.arange(V), ct.astype(np.int64))), shape=(V, Kt))
    d = {}
    for nm, P, idx in (('tu', Pt, idx_tu), ('te', Pe, idx_te)):
        Q = np.asarray(P @ OH); Pc, fd = class_dist(idx, Kt, Kc, chain, 1000 * (ei + 1)); d[nm] = (np.log(np.maximum(Pc, 1e-30)) - np.log(np.maximum(Q, 1e-30)), fd)
    DATA.append((ct, d)); log('専門家 %d (Kt=%d Kc=%d 鎖 %d 段) 完了' % (ei, Kt, Kc, len(chain)))
def total_bits(P, tr, nm, state, LAM, rows=None):
    """LAM[e][状態] -> 位置ごとの bit。 state: 位置ごとの (a, 見つかった段) を 状態番号に した配列"""
    n = len(P); out = np.zeros(n)
    for c0 in range(0, n, 500):
        sl = slice(c0, min(c0 + 500, n)); s = np.zeros((sl.stop - sl.start, V), np.float32)
        for ei, (ct, d) in enumerate(DATA):
            lr = d[nm][0][sl]; s += LAM[ei][state[sl], None] * lr[:, ct]
        e = P[sl] * np.exp(s - s.max(1, keepdims=True)); z = e.sum(1); out[sl] = -np.log2(e[np.arange(len(e)), tr[sl]] / z)
    return out
def state_of(hi, nm):
    f = DATA[0][1][nm][1]; return (np.minimum(hi, 5) - 1) * 7 + np.minimum(f, 6)            # 5 x 7 = 35 状態 (鎖の段が 6 を超えるものは 6 段にまとめる)
st_t, st_e = state_of(hit, 'tu'), state_of(hie, 'te'); NST = 35; grid = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.9, 1.2]
# まず 専門家ごとの 全体で 1 つの重み
LAM = [np.zeros(NST) for _ in DATA]; ALL0 = np.zeros(len(idx_tu), int)
def fit_global():
    for sweep in range(2):
        for ei in range(len(DATA)):
            best = None
            for g in grid:
                LAM[ei][:] = g; v = total_bits(Pt, tr_t, 'tu', st_t, LAM).mean()
                if best is None or v < best[0] - 1e-9: best = (v, g)
            LAM[ei][:] = best[1]
    return best[0]
vt = fit_global(); be = total_bits(Pe, tr_e, 'te', st_e, LAM).mean()
print('RESULT %s  全体の重みだけ: tune %.4f test %.4f -> %.4f (得 %.4f = %.2f%%) 重み %s' % (EXP, base_t, base_e, be, base_e - be, 100 * (base_e - be) / base_e, [float(L[0]) for L in LAM]), flush=True)
# 状態ごとの重み (座標を 状態ごとに 動かす: 調整位置が少ない状態は 全体の重みに寄せる)
for sweep in range(1):
    for ei in range(len(DATA)):
        for stt in range(NST):
            m = st_t == stt
            if m.sum() < 25: continue
            best = None; keep = LAM[ei][stt]
            for g in grid:
                LAM[ei][stt] = g; v = total_bits(Pt[m], tr_t[m], 'tu', st_t[m], LAM, ).mean() if False else None
            LAM[ei][stt] = keep
log('状態ごとの重みは 次の版で')
