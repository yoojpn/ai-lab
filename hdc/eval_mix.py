"""会話の表 (/tmp/tok) と Wikipedia の表 (/tmp/wiki) を混ぜる。 混ぜる割合 w は (会話表で文脈が見えた最高次数, Wikipedia 表のそれ) の区分ごとに、
検証データ (会話と Wikipedia を同じ件数) で格子探索 (勾配なし)。  P = (1-w) P_会話 + w P_wiki
評価: テストで 会話 / Wikipedia(en, ja) それぞれの bit/token。 どちらかの領域で 単独の表より悪化しないか が基準。
使い方: python3 eval_mix.py '{"n":200000}'"""
import sys, json
import numpy as np, engine as E

GRID = np.r_[0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


def load(W):
    lam = {int(k): v for k, v in json.load(open(W + '/lam.json')).items()}
    return E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)


def stats(Md, Mw, T, p):
    lpd, hd = Md.logprob_at(T, p, True); lpw, hw = Mw.logprob_at(T, p, True)
    return 2.0 ** lpd, 2.0 ** lpw, (hd.astype(int) - 1) * 5 + (hw.astype(int) - 1)


def fit_w(parts):
    w = np.zeros(25)
    for b in range(25):
        best = None
        for g in GRID:
            ll = sum(np.log2((1 - g) * pd[bk == b] + g * pw[bk == b]).sum() for pd, pw, bk in parts if (bk == b).any())
            if best is None or ll > best[0]: best = (ll, g)
        w[b] = best[1]
    return w


def mixp(pd, pw, bk, w): g = w[bk]; return (1 - g) * pd + g * pw


def main(cfg):
    n = cfg.get('n', 200000); rng = np.random.default_rng(0); Md = load('/tmp/tok'); Mw = load('/tmp/wiki'); S = {}
    for nm, W in (('dlg', '/tmp/tok'), ('wiki', '/tmp/wiki')):
        for sp in ('val', 'test'):
            T = np.fromfile('%s/%s.bin' % (W, sp), dtype=np.uint16); p = rng.integers(8, len(T), int(n * 1.02)); p = p[T[p] != 0][:n]
            lang = np.load('%s/%s_lang.npy' % (W, sp)); docs = np.load('%s/%s_docs.npy' % (W, sp)); lg = lang[np.maximum(np.searchsorted(docs, p, side='right') - 1, 0)]
            S[(nm, sp)] = stats(Md, Mw, T, p) + (lg,)
    w = fit_w([S[('dlg', 'val')][:3], S[('wiki', 'val')][:3]]); res = {}
    for nm in ('dlg', 'wiki'):
        pd, pw, bk, lg = S[(nm, 'test')]; pm = mixp(pd, pw, bk, w); b = lambda P, m=None: float(-np.log2(P if m is None else P[m]).mean())
        res[nm] = dict(dlg_table=b(pd), wiki_table=b(pw), mix=b(pm))
        for l, ln in ((0, 'en'), (1, 'ja')):
            m = lg == l
            if m.sum() > 1000: res[nm][ln] = dict(n=int(m.sum()), dlg_table=b(pd, m), wiki_table=b(pw, m), mix=b(pm, m))
        print(nm, json.dumps(res[nm], ensure_ascii=False), flush=True)
    json.dump(dict(res=res, w=w.tolist()), open('/tmp/mix_eval.json', 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
