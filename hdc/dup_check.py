"""重複の確認: val の文章と同じ並びが train にあるか。 単発を残した数え上げの下がり方が「丸写し」でないかを調べる。 CPU。
1. train の全位置の 24 語の窓 (その位置で終わる) のハッシュを並べ替えて持つ。 val の各位置の窓がそこにあれば「丸写し位置」。
2. val の文書 (0 の語で区切る) に 丸写し位置が 1 つでもあれば「重複文書」、 無ければ「きれいな文書」。
3. 規模 (次数 1..6, しきい値 1) ごとに 全位置 / きれいな文書の位置 / 重複文書の位置 の bit を測る (lam は前回の値で固定)。
使い方: python3 dup_check.py TOK_DIR WORK_DIR 28000000,110000000,456000000 [N_VAL]"""
import os, sys, time, shutil
import numpy as np
import engine as E

TOK = sys.argv[1]; D = sys.argv[2]; SIZES = [int(x) for x in sys.argv[3].split(',')]; NV = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); os.makedirs(D, exist_ok=True)
W = 24; C1 = np.uint64(0x9E3779B97F4A7C15)
Ttr = np.memmap(TOK + '/train.bin', dtype=np.uint16, mode='r'); NT = len(Ttr); Tv = np.fromfile(TOK + '/val.bin', dtype=np.uint16)


def mix(h): h = h ^ (h >> np.uint64(29)); return h


def win_train(a, b):
    """位置 [a,b) で終わる W 語の窓のハッシュ (a >= W-1)"""
    n = b - a; x = np.asarray(Ttr[a - W + 1:b]).astype(np.uint64); h = np.zeros(n, np.uint64)
    for j in range(W): h = h * C1 + x[j:j + n] + np.uint64(1)
    return mix(h)


def win_val(p):
    h = np.zeros(len(p), np.uint64)
    for j in range(W): h = h * C1 + Tv[p - W + 1 + j].astype(np.uint64) + np.uint64(1)
    return mix(h)


with np.errstate(over='ignore'):
    H = np.empty(NT - W + 1, np.uint64); B = 40_000_000
    for a in range(W - 1, NT, B):
        b = min(NT, a + B); H[a - W + 1:b - W + 1] = win_train(a, b)
    H.sort(); lg('train の窓 %d 個 並べた' % len(H))
    pv_all = np.arange(W - 1, len(Tv)); hv = win_val(pv_all); i = np.minimum(np.searchsorted(H, hv), len(H) - 1); verb = np.zeros(len(Tv), bool); verb[pv_all] = H[i] == hv
del H
did = np.cumsum(Tv == 0); ndoc = int(did.max()) + 1; dd = np.bincount(did, weights=verb.astype(np.float64), minlength=ndoc); dl = np.bincount(did, minlength=ndoc)
docdup = dd[did] > 0
lg('DUP_RESULT val 全 %d 語: 丸写し位置(24語の窓が train にある) %.3f | 重複文書(1 つでもある) の語の割合 %.3f | 文書数 %d' % (len(Tv), verb.mean(), docdup.mean(), ndoc))
rng = np.random.default_rng(777); p = rng.integers(16, len(Tv), int(NV * 1.02)); p = p[Tv[p] != 0][:NV]; clean = ~docdup[p]; vb = verb[p]
lg('DUP_RESULT 評価 %d 位置: きれいな文書 %.3f | 重複文書 %.3f | 丸写し位置 %.3f' % (len(p), clean.mean(), (~clean).mean(), vb.mean()))
ORD = (1, 2, 3, 4, 5, 6); LAM = {2: 2.0, 3: 6.0, 4: 9.0, 5: 6.0, 6: 8.0}
for n_ in SIZES:
    pad = D + '/sub.bin'
    with open(pad, 'wb') as f:
        np.zeros(8, np.uint16).tofile(f)
        if n_ >= NT:
            for a in range(0, NT, 50_000_000): np.asarray(Ttr[a:a + 50_000_000]).tofile(f)
            nt = NT
        else:
            nb = 40; bl = n_ // nb; st = np.linspace(8, NT - bl - 8, nb).astype(np.int64)
            for s in st: np.asarray(Ttr[s:s + bl]).tofile(f)
            nt = nb * bl
    lg('規模 %d 語: 表を作る' % nt); pre = {}
    for o in ORD:
        pf = '%s/s_o%d' % (D, o); runs = E.spill(pad, nt, D + '/runs', [o], 8_000_000, lg); E.merge(runs[o], pf, 1, 20_000_000, lg); shutil.rmtree(D + '/runs', ignore_errors=True); pre[o] = pf
    M = E.Model(pre, smooth='wb', lam=dict(LAM), kn_base=True); lp = np.zeros(len(p))
    for a in range(0, len(p), 5000): lp[a:a + 5000] = M.logprob_at(Tv, p[a:a + 5000])
    b = -lp
    lg('DUP_RESULT 規模 %d 語: 全位置 %.4f | きれいな文書 %.4f (%d 位置) | 重複文書 %.4f (%d 位置) | 丸写し位置 %.4f | 丸写しでない位置 %.4f' % (nt, b.mean(), b[clean].mean(), clean.sum(), b[~clean].mean() if (~clean).any() else float('nan'), (~clean).sum(), b[vb].mean() if vb.any() else float('nan'), b[~vb].mean()))
    del M
    for fn in os.listdir(D):
        if fn.startswith('s_o'): os.remove(os.path.join(D, fn))
lg('終わり')
