"""会話の表と Wikipedia の表の 適応的な混合 (ベイズ混合の格子版)。
  96 語の連続した断片を取り、 各位置で 「直近の語で どちらの表が当たっていたか」 の減衰つき累積 s = sum rho^k (log2 Pw - log2 Pd) を証拠とする (rho を 2 通り)。
  割合 w は 証拠の区分 (分位点で 6x6) ごとに 検証データで格子探索 (勾配なし)。 P = (1-w) P_会話 + w P_wiki。
  文脈が長くなるほど 領域を見分けられる。 断片の最初の 8 語は評価に含めない。
使い方: python3 eval_mix2.py '{"segs":3000,"L":96}'"""
import sys, json
import numpy as np, engine as E
from eval_mix import load

GRID = np.r_[0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 0.97, 1.0]
RHO = (0.85, 0.97)


def segments(T, docs, nseg, L, rng):
    ends = np.r_[docs[1:], len(T)]; out = []
    while len(out) < nseg:
        d = int(rng.integers(0, len(docs))); a, b = docs[d] + 1, ends[d]
        if b - a < 24: continue
        s = int(rng.integers(a, max(a + 1, b - L))); out.append((s, min(s + L, b)))
    return out


def evidence(lpd, lpw):
    d = lpw - lpd; ev = []
    for rho in RHO:
        s = np.zeros_like(d); acc = 0.0
        for i in range(len(d)):
            s[i] = acc; acc = rho * acc + d[i]                               # s[i] = 位置 i より前の証拠
        ev.append(s)
    return ev


def collect(Md, Mw, T, docs, nseg, L, rng):
    P_d = []; P_w = []; E1 = []; E2 = []
    for s, e in segments(T, docs, nseg, L, rng):
        idx = np.arange(s, e); lpd = Md.logprob_at(T, idx); lpw = Mw.logprob_at(T, idx); e1, e2 = evidence(lpd, lpw)
        keep = np.arange(len(idx)) >= 8; P_d.append(2.0 ** lpd[keep]); P_w.append(2.0 ** lpw[keep]); E1.append(e1[keep]); E2.append(e2[keep])
    return [np.concatenate(x) for x in (P_d, P_w, E1, E2)]


def main(cfg):
    rng = np.random.default_rng(0); Md = load('/tmp/tok'); Mw = load('/tmp/wiki'); S = {}
    for nm, W in (('dlg', '/tmp/tok'), ('wiki', '/tmp/wiki')):
        docs = np.load(W + '/test_docs.npy')
        for sp in ('val', 'test'):
            T = np.fromfile('%s/%s.bin' % (W, sp), dtype=np.uint16); docs = np.load('%s/%s_docs.npy' % (W, sp))
            S[(nm, sp)] = collect(Md, Mw, T, docs, cfg.get('segs', 3000), cfg.get('L', 96), rng); print(nm, sp, len(S[(nm, sp)][0]), flush=True)
    allE1 = np.concatenate([S[(n, 'val')][2] for n in ('dlg', 'wiki')]); allE2 = np.concatenate([S[(n, 'val')][3] for n in ('dlg', 'wiki')])
    q = np.linspace(0, 1, 7)[1:-1]; e1 = np.quantile(allE1, q); e2 = np.quantile(allE2, q)
    cell = lambda a, b: np.digitize(a, e1) * 6 + np.digitize(b, e2)
    w = np.zeros(36)
    for c in range(36):
        best = None
        for g in GRID:
            ll = 0.0
            for n in ('dlg', 'wiki'):
                pd, pw, a, b = S[(n, 'val')]; m = cell(a, b) == c
                if m.any(): ll += np.log2((1 - g) * pd[m] + g * pw[m]).sum()
            if best is None or ll > best[0]: best = (ll, g)
        w[c] = best[1]
    res = {}
    for n in ('dlg', 'wiki'):
        pd, pw, a, b = S[(n, 'test')]; g = w[cell(a, b)]; pm = (1 - g) * pd + g * pw; bb = lambda P: float(-np.log2(P).mean())
        res[n] = dict(n=int(len(pd)), dlg_table=bb(pd), wiki_table=bb(pw), mix=bb(pm), mean_w=float(g.mean()))
        print(n, json.dumps(res[n]), flush=True)
    json.dump(dict(res=res, w=w.tolist(), e1=e1.tolist(), e2=e2.tolist(), rho=RHO), open('/tmp/mix2.json', 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
