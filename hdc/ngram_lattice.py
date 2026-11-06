"""数え上げだけの 2 つの進化を測る (同じ物差し: val の bit/語, 表を作る語数 S を変える)
  (1) 次数を 5 から 6,8,12,16 まで伸ばす
  (2) 抽象度の格子: 飛び越し (w-1 を無視) とクラス (語を 256 種に分けた文脈) の専門家を足す
専門家 = 鎖 (Witten-Bell で 1 -> 細かい文脈 と補間)。 混ぜ方は 文脈だけで決まる重み (A の最高次数ごとに 5 通り, EM)。 正解の語は 重みに使わない。
使い方: python3 ngram_lattice.py <W: train.bin, val.bin> <作業場所> <S 語数> [クラスの元にする語数(既定 48000000)]
"""
import sys, os, time, glob, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
V = E.VOC; W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); NCLS = int(sys.argv[4]) if len(sys.argv) > 4 else 48_000_000
t0 = time.time(); log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True); os.makedirs(WORK, exist_ok=True)
SPECS = {}; CLS = np.zeros(V, np.uint16)
_mk, _ch = E.make_keys, E.ctx_hash

def spec_hash(tok, idx, spec, seed):
    h = np.full(len(idx), np.uint64(seed) * E.C1, np.uint64)
    for dist, kind in sorted(spec, key=lambda z: -z[0]):
        t = tok[idx - dist].astype(np.int64)
        if kind == 'c': t = CLS[t].astype(np.int64) + 40000
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
CHAINS = {'A': [2, 3, 4, 5], 'Ahi': [2, 3, 4, 5, 6, 8, 12, 16], 'B': [101, 102, 103], 'B2': [111, 112], 'C': [121, 122, 123, 124, 125], 'C2': [131, 132, 133]}
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
    U, s, _ = sl.svds(P, k=48); F = U * np.sqrt(s)[None]; F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    km = MiniBatchKMeans(n_clusters=256, batch_size=4096, n_init=1, random_state=0).fit(F); np.save(cp, km.labels_.astype(np.uint16)); log('クラス作成')
CLS[:] = np.load(cp)

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
for label, names in (('A (次数1〜5)', ['A']), ('次数を16まで (Ahi)', ['Ahi']), ('A+飛び越し', ['A', 'B', 'B2']), ('A+クラス', ['A', 'C', 'C2']), ('A+飛び越し+クラス (格子)', ['A', 'B', 'B2', 'C', 'C2']),
                     ('Ahi+格子', ['Ahi', 'B', 'B2', 'C', 'C2'])):
    print('RESULT S=%d  %-26s %.4f bit/語' % (n_tok, label, mixture(names)), flush=True)
