"""
HY3: 周回(ブースティング)で、勾配降下・逆伝播なしに骨格を作る。 表は使わない(骨格だけの賢さを見る)。
  座標     : PPMI(距離1,2) の SVD (V x 64)。 文脈 = 位置の束8個の座標の平均 (8 x 64 = 512 特徴、標準化)
  F_0      : 1語の log 確率
  各周 r   : 現在の予測 p から「正解 − p」を作り、新しいランダムReLU特徴(幅 m)から最小二乗(リッジ, 閉形式)で当てる。
             F_r = F_{r-1} + nu * f_r / pu^gamma     (nu, gamma, リッジλ は調整窓の交差エントロピーで選ぶ)
  勾配降下は使わない。 使うのは出力側の「正解 − 予測」だけ。 周回ごとに増えるパラメータ = 512*m + (m+1)*V
使い方: LADDER_CORPUS=... python3 exp_hy3.py '{"N":16000000,"m":512,"R":16,"nfit":400000}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
from exp_cb import V, SEQ, LN2, BUCKETS, log

NF = 64; NB = len(BUCKETS); DIN = NB * NF
LAMS = [1e-2, 1e-1, 1.0]
GAMMAS = [0.0, 0.5, 1.0]
NUS = [0.25 * 2 ** k for k in range(0, 15)]


def pooled(X, E):
    return torch.stack([E[X[:, [SEQ - j for j in Bk]]].mean(1) for Bk in BUCKETS], 1)


def bits(z, y):
    return Fn.cross_entropy(z, y).item() / LN2


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000); nfit = cfg.get('nfit', 400000)
    m = cfg['m']; R = cfg['R']; c = cfg.get('chunk', 10000)
    x, Xva, yva, Xte, yte = B.load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d, 幅 m=%d, 周回 %d' % (len(x), len(yva), len(yte), m, R))
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); logpu = torch.log(pu)
    C = B.build_counts(x[:min(len(x), 16000000)])
    pm = [B.pmi_from_counts(C[k], pu, cfg.get('beta', 1280.0)) for k in range(2)]
    ppmi = torch.clamp(pm[0] + pm[1], min=0); del C
    U, Sv, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :NF] * Sv[:NF].sqrt()[None, :]; log('座標 (SVD) 完了')
    rng = np.random.default_rng(0); pos = np.sort(rng.choice(len(x) - SEQ, nfit, replace=False)) + SEQ
    xs = torch.from_numpy(x.astype(np.int64)); ar = torch.arange(-SEQ, 0)
    X0 = xs[torch.from_numpy(pos[:c])[:, None] + ar[None, :]]; P0 = pooled(X0, E); mu = P0.mean(0); sd = P0.std(0) + 1e-6
    Pf = torch.empty(nfit, DIN); yf = xs[torch.from_numpy(pos)]
    for a in range(0, nfit, c):
        pp = torch.from_numpy(pos[a:a + c]); Pf[a:a + c] = ((pooled(xs[pp[:, None] + ar[None, :]], E) - mu) / sd).reshape(len(pp), DIN)
    Pv = ((pooled(Xva, E) - mu) / sd).reshape(len(Xva), DIN); Pt = ((pooled(Xte, E) - mu) / sd).reshape(len(Xte), DIN)
    del x, xs
    zf = logpu.half()[None, :].expand(nfit, V).contiguous()
    zv = logpu[None, :].expand(len(Pv), V).contiguous(); zt = logpu[None, :].expand(len(Pt), V).contiguous()
    log('周回0 (1語だけ): 調整窓 %.4f, テスト %.4f bit/語' % (bits(zv, yva), bits(zt, yte)))
    hist = [dict(r=0, tune=bits(zv, yva), test=bits(zt, yte), params=0)]
    params = 0; g = torch.Generator().manual_seed(1)
    for r in range(1, R + 1):
        t0 = time.time()
        W1 = torch.randn(DIN, m, generator=g) / math.sqrt(DIN); b1 = torch.randn(m, generator=g) * 0.3
        hid = lambda P: torch.cat([torch.relu(P @ W1 + b1), torch.ones(len(P), 1)], 1)
        Gm = torch.zeros(m + 1, m + 1, dtype=torch.float64); HtG = torch.zeros(m + 1, V, dtype=torch.float64)
        for a in range(0, nfit, c):
            Hc = hid(Pf[a:a + c]); p = torch.softmax(zf[a:a + c].float(), 1); yc = yf[a:a + c]
            Gm += (Hc.T @ Hc).double()
            T = -(Hc.T @ p); T.index_add_(1, yc, Hc.T.contiguous()); HtG += T.double()
        Hv, Ht = hid(Pv), hid(Pt); best = None
        for lam in LAMS:
            W = torch.linalg.solve(Gm + lam * nfit * torch.eye(m + 1, dtype=torch.float64), HtG).float()
            fv = Hv @ W
            for gm in GAMMAS:
                fs = fv / (pu ** gm)[None, :]
                for nu in NUS:
                    b = bits(zv + nu * fs, yva)
                    if best is None or b < best[0]:
                        best = (b, lam, gm, nu)
        _, lam, gm, nu = best
        W = torch.linalg.solve(Gm + lam * nfit * torch.eye(m + 1, dtype=torch.float64), HtG).float()
        sc = (pu ** gm)[None, :]
        zv = zv + nu * (Hv @ W) / sc; zt = zt + nu * (Ht @ W) / sc
        for a in range(0, nfit, c):
            zf[a:a + c] = (zf[a:a + c].float() + nu * (hid(Pf[a:a + c]) @ W) / sc).half()
        params += DIN * m + (m + 1) * V
        h = dict(r=r, tune=bits(zv, yva), test=bits(zt, yte), params=params, lam=lam, gamma=gm, nu=nu, sec=time.time() - t0)
        hist.append(h)
        log('周回%d: 調整窓 %.4f, テスト %.4f bit/語, λ=%g γ=%g ν=%g, 累計パラメータ %.2e, %.0f 秒' % (r, h['tune'], h['test'], lam, gm, nu, params, h['sec']))
    log('RESULT_HY3 ' + json.dumps(dict(cfg=cfg, hist=hist)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
