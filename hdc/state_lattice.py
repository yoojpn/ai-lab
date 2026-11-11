"""状態の住所 (state_lattice): 履歴を「次に来る語の分布が似ているか」でまとめた離散の状態 z を 数え上げだけで作り、 住所にして数える。
  z^0_t = 語 w_t のクラス。 z^r_t = g_r(z^{r-1}_{t-1}, w_t): 組 (前の状態, 今の語) を、 その後に来る語の平均ベクトル (PMI-SVD, 閉形式) の k-means でまとめる。 r 回で r+1 語ぶんの履歴を見る。
  勾配なし。 各回は 1 回の集計 + クラスタリング。 比較対象は 語クラスの格子 (ngram_lattice.py と同じ物差し)。
元の説明: 数え上げだけの 2 つの進化を測る (同じ物差し: val の bit/語, 表を作る語数 S を変える)
  (1) 次数を 5 から 6,8,12,16 まで伸ばす
  (2) 抽象度の格子: 飛び越し (w-1 を無視) とクラス (語を 256 種に分けた文脈) の専門家を足す
専門家 = 鎖 (Witten-Bell で 1 -> 細かい文脈 と補間)。 混ぜ方は 文脈だけで決まる重み (A の最高次数ごとに 5 通り, EM)。 正解の語は 重みに使わない。
使い方: python3 ngram_lattice.py <W: train.bin, val.bin> <作業場所> <S 語数> [クラスの元にする語数(既定 48000000)] ; 環境変数 ZK (状態数, 既定 4096), ZR (回数, 既定 6), ZMIN (組の最低出現数, 既定 5)
"""
import sys, os, time, glob, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
V = E.VOC; W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); NCLS = int(sys.argv[4]) if len(sys.argv) > 4 else 48_000_000
t0 = time.time(); log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True); os.makedirs(WORK, exist_ok=True)
SPECS = {}; CLS = np.zeros(V, np.uint16); ZS = {}
ZK = int(os.environ.get('ZK', 4096)); ZR = int(os.environ.get('ZR', 6)); ZMIN = int(os.environ.get('ZMIN', 5))
_mk, _ch = E.make_keys, E.ctx_hash

def spec_hash(tok, idx, spec, seed):
    h = np.full(len(idx), np.uint64(seed) * E.C1, np.uint64)
    for dist, kind in sorted(spec, key=lambda z: -z[0]):
        t = tok[idx - dist].astype(np.int64)
        if kind == 'c': t = CLS[t].astype(np.int64) + 40000
        elif kind != 'w': t = ZS[kind][len(tok)][idx - dist].astype(np.int64) + 80000 + 70000 * kind
        h = (h ^ (t.astype(np.uint64) + np.uint64(1))) * E.C1; h ^= h >> np.uint64(29)
    return h & E.M48

def make_keys(tok, lo, hi, n):
    if n in SPECS: idx = np.arange(lo, hi); return (spec_hash(tok, idx, SPECS[n], n) << np.uint64(16)) | tok[idx].astype(np.uint64)
    return _mk(tok, lo, hi, n)

def ctx_hash(tok, idx, nctx, seed):
    return spec_hash(tok, idx, SPECS[seed], seed) if seed in SPECS else _ch(tok, idx, nctx, seed)

E.make_keys = make_keys; E.ctx_hash = ctx_hash
w_, c_ = 'w', 'c'
SPECS.update({101: [(2, w_)], 102: [(2, w_), (3, w_)], 103: [(2, w_), (3, w_), (4, w_)], 111: [(1, w_), (3, w_)], 112: [(1, w_), (3, w_), (4, w_)],
              121: [(1, c_)], 122: [(1, c_), (2, c_)], 123: [(1, w_), (2, c_)], 124: [(1, w_), (2, c_), (3, c_)], 125: [(1, w_), (2, w_), (3, c_)],
              131: [(1, c_), (2, c_), (3, c_)], 132: [(1, c_), (2, c_), (3, c_), (4, c_)], 133: [(1, w_), (2, c_), (3, c_), (4, c_)]})
CHAINS = {'A': [2, 3, 4, 5], 'B': [101, 102, 103], 'B2': [111, 112], 'C': [121, 122, 123, 124, 125], 'C2': [131, 132, 133]}
ORD_ALL = sorted({n for ch in CHAINS.values() for n in ch})
MINC = lambda n: 1 if (n in SPECS and len(SPECS[n]) <= 2) or n <= 2 else 2

# --- 学習側の語列 (40 ブロック等間隔) ---
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
sub = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); sub.tofile(WORK + '/sub.bin'); n_tok = len(sub) - 8; log('語数 %d' % n_tok)

# --- クラス: 隣り合う語の PMI の SVD -> k-means (256 種) ---
cp = WORK + '/cls.npy'
if not os.path.exists(cp):
    import scipy.sparse as sp, scipy.sparse.linalg as sl
    from sklearn.cluster import MiniBatchKMeans
    nb2 = 40; bl2 = NCLS // nb2; st2 = np.linspace(8, len(Ttr) - bl2 - 8, nb2).astype(np.int64); x = np.concatenate([np.asarray(Ttr[s:s + bl2]) for s in st2]).astype(np.int64)
    key = x[:-1] * V + x[1:]; u, cnt = np.unique(key, return_counts=True); del key
    M = sp.coo_matrix((cnt.astype(np.float64), (u // V, u % V)), shape=(V, V)).tocsr(); tot = M.sum(); rm = np.asarray(M.sum(1)).ravel() + 1e-9; cm = np.asarray(M.sum(0)).ravel() ** 0.75 + 1e-9; cm = cm / cm.sum() * tot
    M = M.tocoo(); pm = np.log(M.data * tot / (rm[M.row] * cm[M.col])); keep = pm > 0
    P = sp.coo_matrix((pm[keep], (M.row[keep], M.col[keep])), shape=(V, V)).tocsr(); log('PMI 非零 %d' % P.nnz)
    U, s, Vt = sl.svds(P, k=48); F = U * np.sqrt(s)[None]; F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    G2 = Vt.T * np.sqrt(s)[None]; G2 /= np.linalg.norm(G2, axis=1, keepdims=True) + 1e-9; np.save(WORK + '/emb.npy', np.concatenate([F, G2], 1).astype(np.float32))
    km = MiniBatchKMeans(n_clusters=256, batch_size=4096, n_init=1, random_state=0).fit(F); np.save(cp, km.labels_.astype(np.uint16)); log('クラス作成')
CLS[:] = np.load(cp)


# --- 状態 z を作る (数え上げ + k-means) ---
def build_states():
    import scipy.sparse as sp
    from sklearn.cluster import MiniBatchKMeans
    G = np.load(WORK + '/emb.npy'); d = G.shape[1]; tr = np.asarray(sub).astype(np.int64); va = np.concatenate([np.zeros(64, np.int64), np.fromfile(W + '/val.bin', dtype=np.uint16).astype(np.int64)])
    nxt = np.r_[tr[1:], 0]
    def nearest(X, C, bs=20000):
        out = np.empty(len(X), np.int32); c2 = (C * C).sum(1)
        for i in range(0, len(X), bs): out[i:i + bs] = np.argmin(c2[None] - 2 * X[i:i + bs] @ C.T, 1)
        return out
    def follow_vec(ids, n):          # ids: 各位置の群の番号 (-1 は除く) -> 群ごとの 次の語の平均ベクトル, 出現数
        m = ids >= 0; M = sp.coo_matrix((np.ones(int(m.sum()), np.float32), (ids[m], nxt[m])), shape=(n, V)).tocsr(); cnt = np.asarray(M.sum(1)).ravel()
        X = (M @ G) / np.maximum(cnt, 1)[:, None]; return np.asarray(X, np.float32), cnt
    # 0 回目: 語だけで決まる状態 (語の 後ろに来る語の分布で まとめる)
    k0 = min(ZK, 2048); km = MiniBatchKMeans(n_clusters=k0, batch_size=8192, n_init=1, random_state=0).fit(G[:, :d // 2]); W0 = km.labels_.astype(np.int64)
    ztr, zva = W0[tr], W0[va]; ZS[0] = {len(sub): ztr.astype(np.uint16), len(va): zva.astype(np.uint16)}; kprev = k0
    for r in range(1, ZR + 1):
        key = np.r_[0, ztr[:-1]] * V + tr; u, inv, c = np.unique(key, return_inverse=True, return_counts=True); fr = c >= ZMIN; fid = np.full(len(u), -1, np.int64); fid[fr] = np.arange(int(fr.sum()))
        X, cnt = follow_vec(fid[inv], int(fr.sum()))
        km = MiniBatchKMeans(n_clusters=min(ZK, len(X)), batch_size=16384, n_init=1, random_state=r, max_iter=30).fit(X, sample_weight=cnt); C = km.cluster_centers_.astype(np.float32); lab = nearest(X, C)
        Xp, _ = follow_vec(ztr, kprev); back = nearest(Xp, C)          # 稀な組は 前の回の状態から 近い状態へ
        ukeys = u[fr]
        def apply(z_prev_shift, z_prev_same, w):
            k = z_prev_shift * V + w; j = np.searchsorted(ukeys, k); j[j >= len(ukeys)] = 0; hit = ukeys[j] == k
            return np.where(hit, lab[j], back[z_prev_same]).astype(np.int64), hit.mean()
        ztr2, h1 = apply(np.r_[0, ztr[:-1]], ztr, tr); zva2, h2 = apply(np.r_[0, zva[:-1]], zva, va); ztr, zva = ztr2, zva2; kprev = len(C)
        ZS[r] = {len(sub): ztr.astype(np.uint16), len(va): zva.astype(np.uint16)}
        log('状態 %d 回目: 組 %d 種 (うち %d 種が 出現 %d 回以上), 状態 %d, 直接きまる位置の割合 学習 %.3f / val %.3f' % (r, len(u), int(fr.sum()), ZMIN, kprev, h1, h2))
build_states()
z0, zm, zr = 0, max(1, ZR // 3), ZR

SPECS.update({141: [(1, z0)], 142: [(1, zm)], 143: [(1, zr)], 151: [(1, w_), (2, zr)], 152: [(1, w_), (2, w_), (3, zr)], 153: [(1, w_), (2, w_), (3, w_), (4, zr)], 161: [(1, w_), (2, zm)], 162: [(1, w_), (2, w_), (3, zm)]})
CHAINS.update({'Z': [141, 142, 143], 'ZW': [151, 152, 153], 'ZW2': [161, 162]})
ORD_ALL = sorted({n for ch in CHAINS.values() for n in ch})
# --- 表を作る (次数ごとに 作って 併合して ランを消す) ---
for n in ORD_ALL:
    if os.path.exists(WORK + '/final_o%d_k.bin' % n): continue
    runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], WORK + '/final_o%d' % n, MINC(n), 20_000_000, lambda s: None)
    for f in glob.glob(WORK + '/runs/*'): os.remove(f)
    log('次数/型 %d 完了' % n)
if not os.path.exists(WORK + '/final_o1_k.bin'):
    runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [1], 8_000_000, lambda s: None); E.merge(runs[1], WORK + '/final_o1', 1, 20_000_000, lambda s: None)
    for f in glob.glob(WORK + '/runs/*'): os.remove(f)

# --- 評価 ---
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val])
def pos(seed, n): return np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
idx_te, idx_tu = pos(777, 20000), pos(123, 6000)
LAM = {n: (3.0 if n >= 6 or n in SPECS else {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}[n]) for n in ORD_ALL}
P_te, P_tu, HI = {}, {}, {}
for name, ch in CHAINS.items():
    M = E.Model({1: WORK + '/final_o1', **{n: WORK + '/final_o%d' % n for n in ch}}, smooth='wb', lam={**LAM, 1: 1.0}, kn_base=False)
    for key, idx, D in (('te', idx_te, P_te), ('tu', idx_tu, P_tu)):
        lp, hi = M.logprob_at(pad, idx, return_hi=True); D[name] = np.exp2(lp)
        if name == 'A': HI[key] = hi
    log('%s 評価 te %.4f bit' % (name, -np.log2(P_te[name]).mean()))
def mixture(names):
    """A の最高次数ごとに EM で重み (tune 位置)、 test で損失"""
    tot = np.zeros(len(idx_te))
    for b in range(1, 6):
        mt, me = HI['tu'] == b, HI['te'] == b
        if not mt.any() or not me.any(): tot[me] = P_te['A'][me]; continue
        Xt = np.stack([P_tu[n][mt] for n in names], 1); w = np.full(len(names), 1.0 / len(names))
        for _ in range(200): r = Xt * w; r /= r.sum(1, keepdims=True); w = r.mean(0)
        tot[me] = (np.stack([P_te[n][me] for n in names], 1) * w).sum(1)
    return -np.log2(tot).mean()
for label, names in (('A (次数1〜5)', ['A']), ('A+飛び越し', ['A', 'B', 'B2']), ('A+クラス', ['A', 'C', 'C2']), ('A+飛び越し+クラス (格子)', ['A', 'B', 'B2', 'C', 'C2']),
                     ('A+状態 (Z, ZW, ZW2)', ['A', 'Z', 'ZW', 'ZW2']), ('A+状態の語つきだけ (ZW)', ['A', 'ZW']), ('格子+状態', ['A', 'B', 'B2', 'C', 'C2', 'Z', 'ZW', 'ZW2'])):
    print('RESULT S=%d  %-26s %.4f bit/語' % (n_tok, label, mixture(names)), flush=True)
