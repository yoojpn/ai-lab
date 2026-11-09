"""案C: 数え上げの規模曲線 (次数 1..6, しきい値 1 = 単発も残す)。 規模ごとに 表を作って val の bit/token を測る。 CPU。
使い方: python3 scale_curve.py TOK_DIR WORK_DIR 28000000,110000000,456000000 [N_VAL]   (規模 >= 全体 なら 全部)"""
import os, sys, time, shutil
import numpy as np
import engine as E

TOK = sys.argv[1]; D = sys.argv[2]; SIZES = [int(x) for x in sys.argv[3].split(',')]; NV = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); os.makedirs(D, exist_ok=True)
Ttr = np.memmap(TOK + '/train.bin', dtype=np.uint16, mode='r'); NT = len(Ttr)
Tv = np.fromfile(TOK + '/val.bin', dtype=np.uint16); rng = np.random.default_rng(777); p = rng.integers(16, len(Tv), int(NV * 1.02)); p = p[Tv[p] != 0][:NV]; h = len(p) // 2
ORD = (1, 2, 3, 4, 5, 6)
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
    lam = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0, 6: 4.0}; M = E.Model(pre, smooth='wb', lam=lam, kn_base=True); bits = lambda idx: -E.Model.logprob_at(M, Tv, idx).mean()
    for o in (3, 4, 5, 6):
        best = None
        for f_ in (0.5, 1.0, 2.0):
            M.lam[o] = lam[o] * f_; b = bits(p[:h])
            if best is None or b < best[0]: best = (b, lam[o] * f_)
        lam[o] = best[1]; M.lam[o] = best[1]
    lp, hi = M.logprob_at(Tv, p[h:], return_hi=True); cov = {o: round(float((hi >= o).mean()), 3) for o in ORD if o >= 3}
    lg('SCALE_RESULT 規模 %d 語 (次数 1-6, しきい値 1): %.4f bit/token (評価 %d 位置) lam=%s 被覆=%s' % (nt, -lp.mean(), len(p) - h, {o: lam[o] for o in (3, 4, 5, 6)}, cov))
    del M
    for fn in os.listdir(D):
        if fn.startswith('s_o'): os.remove(os.path.join(D, fn))
lg('終わり')
