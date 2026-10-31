"""表ごとの λ (WB の逃げ幅の倍率) を 検証データで座標降下の格子探索 (勾配なし) し、 <dir>/lam.json に保存。
使い方: python3 tune_lam.py '{"dir":"/tmp/wiki"}'"""
import sys, json
import numpy as np, engine as E


def main(cfg):
    W = cfg['dir']; Tva = np.fromfile(W + '/val.bin', dtype=np.uint16); rng = np.random.default_rng(0)
    pv = rng.integers(8, len(Tva), 150000); pv = pv[Tva[pv] != 0]
    P = {n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}
    def val(lam): return -E.Model(P, smooth='wb', lam=lam, kn_base=True).logprob_at(Tva, pv).mean()
    lam = {n: 1.0 for n in (2, 3, 4, 5)}; cur = val(lam); print('start val %.4f' % cur)
    for rnd in range(2):
        for n in (2, 3, 4, 5):
            best = (cur, lam[n])
            for l in (0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.5, 6.0, 9.0, 14.0):
                t = dict(lam); t[n] = l; v = val(t)
                if v < best[0]: best = (v, l)
            lam[n] = best[1]; cur = best[0]
        print('round', rnd, lam, 'val %.4f' % cur)
    json.dump({str(k): v for k, v in lam.items()}, open(W + '/lam.json', 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
