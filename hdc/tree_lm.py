"""
tree_lm.py: 決定木 (ランダムフォレスト) 型の言語モデル。 勾配なし、 数え上げだけ。
  - 文脈の特徴 (core.features の 27 個) で 文脈を分割する木を作る。 分け方は 「次の語の語クラス」 の分布のばらつきが一番減る特徴 (多分岐; 罰則つき)。
  - 葉に 「次の語」 の回数を持たせる。 検証/テストの文脈は 木をたどって葉に着く。
  - 葉の回数を 事前分布=n-gram LM で補った事後確率  p = (c + a * p_lm) / (N + a)  にする。 複数の木は平均。
  使い方: python3 tree_lm.py '{"nstruct":1000000,"ncount":6000000,"ntree":4}'
"""
import sys, json, time
import numpy as np
import engine as E, core as C

V = 32768


def best_split(Fr, lr, feats, minleaf):
    """二分岐: 「特徴 f の値が v か」。 全部の (f, v) を ヒストグラムで一度に評価。 戻り値 (得点, f, v)"""
    n = len(lr); L = np.bincount(lr, minlength=256).astype(np.float64); Lp = int((L > 0).sum())
    hl = float((L[L > 0] * np.log2(n / L[L > 0])).sum()); best = (0.0, None, None); pen = 0.5 * Lp * np.log2(n)
    for f in feats:
        cnt = np.bincount(Fr[:, f].astype(np.int32) * 256 + lr, minlength=65536).reshape(256, 256).astype(np.float64)
        nv = cnt.sum(1); ok = (nv >= minleaf) & (n - nv >= minleaf)
        if not ok.any(): continue
        Lc = cnt[ok]; Rc = L[None, :] - Lc; ln = nv[ok][:, None]; rn = (n - nv[ok])[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            cl = np.where(Lc > 0, Lc * np.log2(ln / Lc), 0.0).sum(1); cr = np.where(Rc > 0, Rc * np.log2(rn / Rc), 0.0).sum(1)
        sc = hl - cl - cr - pen; k = int(np.argmax(sc))
        if sc[k] > best[0]: best = (float(sc[k]), f, int(np.flatnonzero(ok)[k]))
    return best


def build_tree(F, lab, rows, feats, minleaf=40, maxdepth=18, maxnodes=200000, minn=100):
    feat_of = [-1]; lut = {}; stack = [(0, rows, 0)]; nnode = 1
    while stack:
        nid, r, d = stack.pop()
        if d >= maxdepth or len(r) < minn or nnode > maxnodes: continue
        sc, f, v = best_split(F[r], lab[r], feats, minleaf)
        if f is None: continue
        lc, rc = nnode, nnode + 1; nnode += 2; feat_of += [-1, -1]
        table = np.full(256, rc, np.int32); table[v] = lc; feat_of[nid] = f; lut[nid] = table
        m = F[r, f] == v; stack.append((lc, r[m], d + 1)); stack.append((rc, r[~m], d + 1))
    fo = np.array(feat_of, np.int32); LUT = np.zeros((len(fo), 256), np.int32)
    for k, t in lut.items(): LUT[k] = t
    return fo, LUT


def route(fo, LUT, F):
    cur = np.zeros(len(F), np.int32)
    while True:
        f = fo[cur]; m = f >= 0
        if not m.any(): return cur
        idx = np.flatnonzero(m)
        cur[idx] = LUT[cur[idx], F[idx, f[idx]]]


class Leaves:
    def __init__(self, leaf, y, nleaf):
        key = leaf.astype(np.int64) * V + y; self.u, self.c = np.unique(key, return_counts=True); self.N = np.bincount(leaf, minlength=nleaf).astype(np.float64)

    def get(self, leaf, y):
        q = leaf.astype(np.int64) * V + y; i = np.minimum(np.searchsorted(self.u, q), len(self.u) - 1)
        return np.where(self.u[i] == q, self.c[i], 0).astype(np.float64), self.N[leaf]


def sample_pos(T, lo, hi, n, rng):
    p = rng.integers(lo, hi, int(n * 1.02)); p = p[np.asarray(T[p]) != 0]
    return np.unique(p)[:n] if len(p) >= n else np.unique(p)


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    W = cfg.get('dir', '/tmp/tok'); rng = np.random.default_rng(cfg.get('seed', 0)); aux = np.load(W + '/aux.npz'); fm = aux['fm']; cls = aux['cls']
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); Tva = np.fromfile(W + '/val.bin', dtype=np.uint16); Tte = np.fromfile(W + '/test.bin', dtype=np.uint16)
    st = {s: C.Struct(T, np.load('%s/%s_docs.npy' % (W, s)), [1, 2]) for s, T in (('train', Ttr), ('val', Tva), ('test', Tte))}
    lam = {int(k): v for k, v in json.load(open(W + '/lam.json')).items()}
    model = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    pc = sample_pos(Ttr, 8, len(Ttr), cfg['ncount'], rng); pva = sample_pos(Tva, 8, len(Tva), cfg.get('nval', 400_000), rng); pte = sample_pos(Tte, 8, len(Tte), cfg.get('ntest', 1_000_000), rng)
    Fc, names, _ = C.features(Ttr, pc, st['train'], fm, cls); yc = np.asarray(Ttr[pc]).astype(np.int64); labc = cls[yc].astype(np.int64)
    Fv, _, _ = C.features(Tva, pva, st['val'], fm, cls); yv = Tva[pva].astype(np.int64)
    Ft, _, _ = C.features(Tte, pte, st['test'], fm, cls); yt = Tte[pte].astype(np.int64)
    lpv, hiv = model.logprob_at(Tva, pva, return_hi=True); lpt, hit = model.logprob_at(Tte, pte, return_hi=True); pv = 2.0 ** lpv; pt = 2.0 ** lpt
    log('特徴 %d 個, 数える標本 %d, 検証 %d, テスト %d | LM5 bit/token 検証 %.4f テスト %.4f' % (len(names), len(pc), len(pva), len(pte), -lpv.mean(), -lpt.mean()))
    nf = len(names); ns = cfg['nstruct']; kf = cfg.get('kfeat', max(6, int(np.sqrt(nf) * 2)))
    outs = []
    for t in range(cfg['ntree']):
        feats = np.sort(rng.choice(nf, kf, replace=False)); rows = rng.choice(len(pc), ns, replace=True) if cfg.get('boot', True) else np.arange(ns)
        fo, LUT = build_tree(Fc, labc, rows, feats, minleaf=cfg.get('minleaf', 40), maxdepth=cfg.get('maxdepth', 18), minn=cfg.get('minn', 100))
        lc = route(fo, LUT, Fc); lv = route(fo, LUT, Fv); lt = route(fo, LUT, Ft); nleaf = len(fo)
        Lf = Leaves(lc, yc, nleaf); cv, Nv = Lf.get(lv, yv); ct, Nt = Lf.get(lt, yt)
        isleaf = fo < 0; outs.append((cv, Nv, ct, Nt))
        log('木 %d: 特徴 %s 節点 %d (葉 %d) 数える標本のうち葉の平均大きさ %.0f' % (t, [names[i] for i in feats][:8], nleaf, int(isleaf.sum()), len(pc) / max(1, isleaf.sum())))
    A = [8, 32, 128, 512, 2048]; G = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    def ens(k, a, which):
        ps = []
        for (cv, Nv, ct, Nt) in outs[:k]:
            c, N, p = (cv, Nv, pv) if which == 'v' else (ct, Nt, pt)
            ps.append((c + a * p) / (N + a))
        return np.mean(ps, axis=0)
    bits = lambda P, m=None: float(-np.log2(P if m is None else P[m]).mean())
    uv = hiv <= 2; ut = hit <= 2
    res = dict(lm5_test=bits(pt), lm5_test_unseen=bits(pt, ut), frac_unseen=float(ut.mean()), curve={})
    def mixp(k, a, g, which):
        P = ens(k, a, which); plm = pv if which == 'v' else pt
        return (1 - g) * plm + g * P
    for k in sorted(set([1, 2, 4, 8, cfg['ntree']])):
        if k > cfg['ntree']: continue
        grid = [(bits(mixp(k, a, g, 'v')), a, g) for a in A for g in G]; bv, a, g = min(grid); P = mixp(k, a, g, 't')
        res['curve'][k] = dict(a=a, g=g, test=bits(P), unseen=bits(P, ut), val=bv)
        log('木 %d 本: a=%s g=%s テスト %.4f bit (LM5 %.4f) | 初出文脈 %.4f (LM5 %.4f, 割合 %.3f)' % (k, a, g, bits(P), res['lm5_test'], bits(P, ut), res['lm5_test_unseen'], res['frac_unseen']))
    log('RESULT_TREE ' + json.dumps(res, ensure_ascii=False))
    json.dump(dict(cfg=cfg, res=res), open('tree_lm_result.json', 'w'), indent=1)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {'nstruct': 1_000_000, 'ncount': 6_000_000, 'ntree': 4})
