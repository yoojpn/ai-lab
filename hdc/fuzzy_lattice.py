"""ほころび (1語を穴にした文脈) の専門家を足す 実験。 ngram_lattice.py の 派生。
元の説明: 数え上げだけの 2 つの進化を測る (同じ物差し: val の bit/語, 表を作る語数 S を変える)
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


IDF = np.zeros(V, np.float32); WIN = 64
def cue_of(tok, idx, chunk=400000):
    """各位置の 直前 (3〜66 語前) の窓で いちばん珍しい語 と 2 番目 (違う語) を返す。 珍しさ = idf (出現 5 回未満の語は 0)"""
    c1 = np.zeros(len(idx), np.int64); c2 = np.zeros(len(idx), np.int64); off = np.arange(3, 3 + WIN)
    for a in range(0, len(idx), chunk):
        ii = idx[a:a + chunk]; g = tok[np.maximum(ii[:, None] - off[None, :], 0)].astype(np.int64); sc = IDF[g]
        j = sc.argmax(1); r = np.arange(len(ii)); t1 = g[r, j]; ok1 = sc[r, j] > 0; sc = np.where(g == t1[:, None], 0, sc); j2 = sc.argmax(1); t2 = g[r, j2]; ok2 = sc[r, j2] > 0
        c1[a:a + chunk] = np.where(ok1, t1, 0) + 50000; c2[a:a + chunk] = np.where(ok2, t2, 0) + 50000
    return c1, c2

def spec_hash(tok, idx, spec, seed):
    h = np.full(len(idx), np.uint64(seed) * E.C1, np.uint64)
    for dist, kind in sorted(spec, key=lambda z: -z[0]):
        if kind in ('k1', 'k2'): t = cue_of(tok, idx)[0 if kind == 'k1' else 1]
        else: t = tok[idx - dist].astype(np.int64)
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
SPECS.update({})
for L_, base_ in ((6, 300), (8, 320)):
    for j_ in range(1, L_ + 1): SPECS[base_ + j_] = [(k, w_) for k in range(1, L_ + 1) if k != j_]
SPECS.update({101: [(2, w_)], 102: [(2, w_), (3, w_)], 111: [(1, w_), (3, w_)]})
CHAINS = {'A': [2, 3, 4, 5], 'B': [101, 102], 'B2': [111]}
for L_, base_ in ((6, 300), (8, 320)):
    for j_ in range(1, L_ + 1): CHAINS['F%d_%d' % (L_, j_)] = [base_ + j_]
ORD_ALL = sorted({n for ch in CHAINS.values() for n in ch})
MINC = lambda n: 1 if (n in SPECS and len(SPECS[n]) <= 2) or n <= 2 else 2

# --- 学習側の語列 (40 ブロック等間隔) ---
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
sub = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); sub.tofile(WORK + '/sub.bin'); n_tok = len(sub) - 8; log('語数 %d' % n_tok)

cnt_ = np.bincount(sub[8:], minlength=V).astype(np.float64); IDF[:] = np.where(cnt_ >= 5, np.log(cnt_.sum() / (cnt_ + 1)), 0); IDF[:4] = 0; log('IDF 作成')

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
P_te, P_tu, HI, AV = {}, {}, {}, {}
for name, ch in CHAINS.items():
    M = E.Model({1: WORK + '/final_o1', **{n: WORK + '/final_o%d' % n for n in ch}}, smooth='wb', lam={**LAM, 1: 1.0}, kn_base=False)
    for key, idx, D in (('te', idx_te, P_te), ('tu', idx_tu, P_tu)):
        lp, hi = M.logprob_at(pad, idx, return_hi=True); D[name] = np.exp2(lp); AV[(name, key)] = (hi > 1) if name != 'A' else np.ones(len(idx), bool)
        if name == 'A': HI[key] = hi
    log('%s 評価 te %.4f bit' % (name, -np.log2(P_te[name]).mean()))
for nm in ['F6_1', 'F6_3', 'F8_2', 'F8_5']:
    m = AV[(nm, 'te')]; print('DEBUG %s 文脈あり %.2f%%  その位置で 専門家 %.3f bit / A %.3f bit' % (nm, 100 * m.mean(), -np.log2(P_te[nm][m]).mean() if m.any() else -1, -np.log2(P_te['A'][m]).mean() if m.any() else -1), flush=True)
def mixture(names):
    """A の最高次数ごとに EM で重み。 文脈が見つかった専門家 (A は常に) だけで混ぜる。 見つからない専門家は 重みから外して 残りで 正規化"""
    tot = np.zeros(len(idx_te))
    for b in range(1, 6):
        mt, me = HI['tu'] == b, HI['te'] == b
        if not mt.any() or not me.any(): tot[me] = P_te['A'][me]; continue
        Xt = np.stack([P_tu[n][mt] for n in names], 1); At = np.stack([AV[(n, 'tu')][mt] for n in names], 1).astype(float); w = np.full(len(names), 1.0 / len(names))
        for _ in range(200): r = Xt * w * At; r /= r.sum(1, keepdims=True); w = r.mean(0)
        Xe = np.stack([P_te[n][me] for n in names], 1); Ae = np.stack([AV[(n, 'te')][me] for n in names], 1).astype(float); ww = w * Ae; ww /= ww.sum(1, keepdims=True)
        tot[me] = (Xe * ww).sum(1)
    return -np.log2(tot).mean()
F6 = ['F6_%d' % j for j in range(1, 7)]; F8 = ['F8_%d' % j for j in range(1, 9)]
for label, names in (('A (次数1〜5)', ['A']), ('A+飛び越し', ['A', 'B', 'B2']), ('A+穴あき L=6 (1語違い)', ['A'] + F6), ('A+穴あき L=8', ['A'] + F8), ('A+穴あき L=6,8', ['A'] + F6 + F8),
                     ('A+飛び越し+穴あき L=6,8', ['A', 'B', 'B2'] + F6 + F8)):
    print('RESULT S=%d  %-30s %.4f bit/語' % (n_tok, label, mixture(names)), flush=True)
