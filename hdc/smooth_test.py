"""n-gram 本体の平滑化の質を測る: Witten-Bell (今の基準) と 絶対割引 (Kneser-Ney 型) を 同じ位置で 比べる。 D は 次数ごとに 調整位置で 座標ごとに決める。
使い方: python3 smooth_test.py <W> <WORK>"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK = sys.argv[1], sys.argv[2]; val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
idx_tu, idx_te = pos(123, 6000), pos(777, 20000)
ords = (1, 2, 3, 4, 5); pf = {n: WORK + '/final_o%d' % n for n in ords}
def bits(M, idx): return -M.logprob_at(pad, idx).mean()
Mw = E.Model(pf, smooth='wb', lam={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}, kn_base=False); print('RESULT Witten-Bell (基準)  tune %.4f test %.4f' % (bits(Mw, idx_tu), bits(Mw, idx_te)), flush=True)
best = (None, None)
for kn in (False, True):
    D = {n: 0.8 for n in ords}; M = E.Model(pf, smooth='ad', D=D, kn_uni=True, kn_base=kn)
    grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for sweep in range(2):
        for n in ords[1:]:
            b = None
            for g in grid:
                M.D[n] = g; v = bits(M, idx_tu)
                if b is None or v < b[0]: b = (v, g)
            M.D[n] = b[1]
    print('RESULT 絶対割引 (kn_base=%s)  D=%s  tune %.4f test %.4f' % (kn, {n: M.D[n] for n in ords[1:]}, bits(M, idx_tu), bits(M, idx_te)), flush=True)
