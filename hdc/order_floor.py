"""次数の床の試験 (CPU, 数え上げのみ)。 次数 5 -> 6, 8, 12 に上げると val の bit/token がどれだけ下がるか。
本番の表 (次数 1..5, しきい値 2) は tok/ から。 新しい次数は train.bin 全体 (約 4.6 億語) から作る。 しきい値 1 (単発も残す) と 2 の両方。
使い方: python3 order_floor.py TOK_DIR WORK_DIR [NVAL]"""
import os, sys, time, json, shutil
import numpy as np
import engine as E

TOK = sys.argv[1]; D = sys.argv[2]; NV = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); os.makedirs(D, exist_ok=True)
Ttr = np.memmap(TOK + '/train.bin', dtype=np.uint16, mode='r'); NT = len(Ttr); lg('train %d 語' % NT)
pad = D + '/train_pad.bin'
if not os.path.exists(pad):
    with open(pad, 'wb') as f:
        np.zeros(8, np.uint16).tofile(f); B = 50_000_000
        for a in range(0, NT, B): np.asarray(Ttr[a:a + B]).tofile(f)
for n in (6, 8, 12):
    for mc in (1, 2):
        pf = '%s/o%d_m%d' % (D, n, mc)
        if os.path.exists(pf + '_k.bin'): continue
        if mc == 1:
            runs = E.spill(pad, NT, D + '/runs%d' % n, [n], 8_000_000, lg)
        E.merge(runs[n], pf, mc, 20_000_000, lg)
    shutil.rmtree(D + '/runs%d' % n, ignore_errors=True)
    lg('次数 %d 済み' % n)

Tv = np.fromfile(TOK + '/val.bin', dtype=np.uint16); rng = np.random.default_rng(777); p = rng.integers(16, len(Tv), int(NV * 1.02)); p = p[Tv[p] != 0][:NV]; h = len(p) // 2
lam0 = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}
base = {n: '%s/final_o%d' % (TOK, n) for n in (1, 2, 3, 4, 5)}


def run(name, new, grid=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """new: [(次数, しきい値)]。 新しい次数の lam を 前半で選び、 後半で測る"""
    pre = dict(base); lam = dict(lam0)
    for n, mc in new: pre[n] = '%s/o%d_m%d' % (D, n, mc); lam[n] = 4.0
    M = E.Model(pre, smooth='wb', lam=lam, kn_base=True)
    for n, mc in new:
        best = None
        for v in grid:
            M.lam[n] = v; b = -E.Model.logprob_at(M, Tv, p[:h]).mean()
            if best is None or b < best[0]: best = (b, v)
        M.lam[n] = best[1]
    lp, hi = M.logprob_at(Tv, p[h:], return_hi=True); b = -lp.mean(); cov = {int(n): float((hi >= n).mean()) for n in M.orders if n >= 5}
    lg('ORDER_RESULT %s: %.4f bit/token (評価 %d 位置) lam=%s 被覆(その次数以上の文脈が見つかった割合)=%s' % (name, b, len(p) - h, {n: M.lam[n] for n, _ in new}, {k: round(v, 3) for k, v in cov.items()}))


run('1-5 (本番)', [])
for mc in (2, 1):
    run('1-5,6 しきい値%d' % mc, [(6, mc)])
    run('1-5,6,8 しきい値%d' % mc, [(6, mc), (8, mc)])
    run('1-5,6,8,12 しきい値%d' % mc, [(6, mc), (8, mc), (12, mc)])
lg('終わり')
