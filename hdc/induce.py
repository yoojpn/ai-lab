"""(特徴の自動探索: 設計の候補を たくさん作り、 いまのモデルに足したとき 調整位置の損失が いちばん下がるものを 貪欲に採る。 重みは 数えた損失だけで決める)
クラスで次の語を先に予測する (階層分解) の 多段・混成版。  数え上げ + k-means のみ (勾配なし)。 正規化は 全語彙で厳密。
 専門家 e = (目標のクラス数 Kt, 文脈のクラス数 Kc, 鎖)。 鎖 = 文脈の型の列 (小さい文脈 -> 大きい文脈)。 型の要素 = (距離, 'w' 語そのもの | 'c' クラス)。
 P'(w) = P(w) exp( sum_e l_e[状態] log(Pc_e(c_e(w)|文脈) / Q_e(c_e(w))) ) / Z     Q_e = n-gram の確率の クラスごとの和
 l_e は (n-gram の最高次数, 鎖の どこまで見つかったか) の 状態ごとに 調整位置の損失だけで 座標ごとに決める。
使い方: python3 classlm2.py <W> <WORK> <S> <実験名>"""
import sys, os, time, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); EXP = 'induce'; t0 = time.time(); V = E.VOC; log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
tok = np.fromfile(WORK + '/sub.bin', dtype=np.uint16); n_tok = len(tok) - 8; Em = np.load(WORK + '/E.npy')
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
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

def skip1(m): return [[(k, 'c') for k in range(2, i + 1)] for i in range(2, m + 1)]
def w2c(m): return [[(1, 'c'), (2, 'w')] + [(k, 'c') for k in range(3, i + 1)] for i in range(2, m + 1)]
def h1skip(m): return [[(1, 'w')] + [(k, 'c') for k in range(3, i + 1)] for i in range(2, m + 1)]
def w3c(m): return [[(1, 'c'), (2, 'c'), (3, 'w')] + [(k, 'c') for k in range(4, i + 1)] for i in range(3, m + 1)]
def skip2(m): return [[(k, 'c') for k in range(3, i + 1)] for i in range(3, m + 1)]
def w2o(m): return [[(2, 'w')] + [(k, 'c') for k in range(3, i + 1)] for i in range(2, m + 1)]
def w1w3(m): return [[(1, 'w'), (2, 'c'), (3, 'w')] + [(k, 'c') for k in range(4, i + 1)] for i in range(3, m + 1)]
TEMPL = {'c': cchain, 'h1': hchain, 'h2': h2chain, 'skip1': skip1, 'w2c': w2c, 'h1skip': h1skip, 'w3c': w3c, 'skip2': skip2, 'w2o': w2o, 'w1w3': w1w3}
S_tu = np.load(WORK + '/S_tu.npy')[:len(idx_tu)]; S_te = np.load(WORK + '/S_te.npy')[:len(idx_te)]
def tbits(P, S, tr):
    out = np.zeros(len(P))
    for c0 in range(0, len(P), 500):
        sl = slice(c0, c0 + 500); e = P[sl] * np.exp(S[sl] - S[sl].max(1, keepdims=True)); out[sl] = -np.log2(e[np.arange(len(e)), tr[sl]] / e.sum(1))
    return out.mean()
b0t, b0e = tbits(Pt, S_tu, tr_t), tbits(Pe, S_te, tr_e); print('RESULT 出発点 (統合モデル): tune %.4f test %.4f' % (b0t, b0e), flush=True)
CANDS = [(tn, Kt, Kc) for tn in ('c', 'h1', 'h2', 'skip1', 'w2c', 'h1skip') for Kt in (64, 256, 1024) for Kc in (64, 256)] + [(tn, Kt, Kc) for tn in TEMPL for Kt in (1024, 4096) for Kc in (256, 1024)]
CANDS = list(dict.fromkeys(CANDS)); LRS = {}
for ci, (tn, Kt, Kc) in enumerate(CANDS):
    fp = WORK + '/ind_%s_%d_%d.npz' % (tn, Kt, Kc)
    if os.path.exists(fp): z = np.load(fp); LRS[(tn, Kt, Kc)] = (z['tu'], z['te']); continue
    ct = get_cls(Kt); OH = sp.csr_matrix((np.ones(V), (np.arange(V), ct.astype(np.int64))), shape=(V, Kt)); chain = TEMPL[tn](5 if Kt < 1024 else 4); d = {}
    for nm, P, idx in (('tu', Pt, idx_tu), ('te', Pe, idx_te)):
        Q = np.asarray(P @ OH); Pc, fd = class_dist(idx, Kt, Kc, chain, 100000 * (ci + 1)); d[nm] = (np.log(np.maximum(Pc, 1e-30)) - np.log(np.maximum(Q, 1e-30))).astype(np.float32)
    np.savez(fp, tu=d['tu'], te=d['te']); LRS[(tn, Kt, Kc)] = (d['tu'], d['te']); log('候補 %d/%d %s Kt=%d Kc=%d 作成' % (ci + 1, len(CANDS), tn, Kt, Kc))
grid = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7]; cur_t, cur_e = b0t, b0e; chosen = []
for rnd in range(12):
    best = None
    for key, (lt, le) in LRS.items():
        ct = get_cls(key[1])
        for g in grid:
            v = tbits(Pt, S_tu + g * lt[:, ct], tr_t)
            if best is None or v < best[0]: best = (v, key, g)
    if cur_t - best[0] < 0.0015: print('RESULT 打ち切り: 最良の候補でも 調整位置で %.4f しか下がらない' % (cur_t - best[0]), flush=True); break
    v, key, g = best; ct = get_cls(key[1]); S_tu = S_tu + g * LRS[key][0][:, ct]; S_te = S_te + g * LRS[key][1][:, ct]; cur_t = v; cur_e = tbits(Pe, S_te, tr_e); chosen.append((key, g))
    print('RESULT 採用 %d: %s Kt=%d Kc=%d 重み %.2f -> tune %.4f test %.4f (出発点からの得 %.4f = %.2f%%)' % (rnd + 1, key[0], key[1], key[2], g, cur_t, cur_e, b0e - cur_e, 100 * (b0e - cur_e) / b0e), flush=True)
