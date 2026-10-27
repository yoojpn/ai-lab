"""
L: 低ランク3階テンソル分解(CP-ALS)で、3-gramの対数確率比を補完する「1回ずつ閉形式」の言語モデル。
  観測した3語組 (w2,w1,y) について、目標  T = log p_tri(y|w2,w1) - log p_bi(y|w1)  (3-gramが2-gramに足した情報)。
  モデル:  s(w2,w1,y) = sum_r A[w2,r] * B[w1,r] * C[y,r]       (3つの因子ベクトルの要素ごとの積の和 = 掛け算で組み合わせる)
  ALS: A,B,C のどれか1つを残り2つで固定したとき、その因子の各行は「重み付き最小二乗(リッジ)」=閉形式。 3つを順に解く(1掃引)。
       勾配降下も逆伝播も使わない。 1掃引 = 観測3語組を3回なぞる。
  予測: logit(y) = log p_tri_full(y) + beta * s(w2,w1,y)   (p_tri_fullは平滑化3-gram。beta は検証窓で決める)
  効果: 見たことのない (w2,w1,y) の組み合わせでも、似た語の因子が近いので、補完される。
計算量/語(推論): r*2 + V*r 回(語彙全体の採点)。 学習: 約 3*sweeps*nnz*r^2 回 + 解く V*r^3 回。
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, log
from exp_r import Prior


def bits_of(p): return float(-np.log2(np.clip(p, 1e-12, 1)).mean())


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); t0 = time.time(); torch.manual_seed(0)
    ctx = lab.load(cfg['N']); st = np.concatenate(ctx.tr_i)[:cfg['N']].astype(np.int64)
    pr = Prior(st); NUM = pr.NUM.double(); pbi = (NUM / NUM.sum(1, keepdim=True).clamp_min(1e-12)).float()      # p_bi(y|w1) (V,V)
    pbi = torch.where(pbi.sum(1, keepdim=True) > 0, pbi, pr.pu[None, :].expand_as(pbi)); delta = 0.75
    tk = (st[:-2] * V + st[1:-1]) * V + st[2:]; tu, tc = np.unique(tk, return_counts=True)
    w2 = tu // (V * V); w1 = (tu // V) % V; yy = tu % V; ctxk = tu // V
    cu, cinv = np.unique(ctxk, return_inverse=True); cn = np.bincount(cinv, weights=tc); ct = np.bincount(cinv).astype(np.float64)
    ptri = (np.maximum(tc - delta, 0) + delta * ct[cinv] * pbi.numpy()[w1, yy]) / cn[cinv]
    Tt = np.log1p(np.maximum(tc - delta, 0) / (delta * ct[cinv] * pbi.numpy()[w1, yy]))      # 文脈ごとの基準(見ていない語の確率)に対する上乗せ
    keep = tc >= cfg.get('minc', 2)
    i2, i1, iy = [torch.from_numpy(a[keep]) for a in (w2, w1, yy)]; T = torch.from_numpy(Tt[keep]).float(); W = torch.from_numpy(tc[keep].astype(np.float32)).sqrt()
    K_ = cfg.get('neg', 2)
    if K_:
        pu75 = torch.from_numpy(((np.bincount(st, minlength=V) + 1.0) ** 0.75).astype(np.float32)); ni = torch.multinomial(pu75, len(i2) * K_, replacement=True)
        i2 = torch.cat([i2, i2.repeat(K_)]); i1 = torch.cat([i1, i1.repeat(K_)]); iy = torch.cat([iy, ni])
        T = torch.cat([T, torch.zeros(len(ni))]); W = torch.cat([W, (W.repeat(K_)) * 0.5])
    nnz = len(i2); log('triples %d (fit on %d, count>=%d) [%.0fs]' % (len(tu), nnz, cfg.get('minc', 2), time.time() - t0))
    r = cfg.get('r', 32); sweeps = cfg.get('sweeps', 8); lam = cfg.get('lam', 0.05); ch = cfg.get('chunk', 40000)
    F = [torch.randn(V, r) * 0.5 for _ in range(3)]      # A (by w2), B (by w1), C (by y)
    idx = [i2, i1, iy]
    def fit_err():
        num = den = 0.0
        for a in range(0, nnz, 400000):
            z = F[0][i2[a:a + 400000]] * F[1][i1[a:a + 400000]] * F[2][iy[a:a + 400000]]
            e = z.sum(1) - T[a:a + 400000]; ww = W[a:a + 400000] ** 2
            num += float((ww * e * e).sum()); den += float(ww.sum())
        return (num / den) ** 0.5
    log('init rmse %.4f' % fit_err())
    for sw in range(sweeps):
        for f in range(3):
            o = [k for k in range(3) if k != f]
            G = torch.zeros(V, r * r); H = torch.zeros(V, r)
            for a in range(0, nnz, ch):
                z = F[o[0]][idx[o[0]][a:a + ch]] * F[o[1]][idx[o[1]][a:a + ch]]; wt = W[a:a + ch][:, None]
                G.index_add_(0, idx[f][a:a + ch], ((z * wt)[:, :, None] * (z * wt)[:, None, :]).reshape(len(z), r * r))
                H.index_add_(0, idx[f][a:a + ch], (wt * wt * T[a:a + ch][:, None]) * z)
            G = G.view(V, r, r); tr = torch.diagonal(G, dim1=1, dim2=2).sum(1, keepdim=True) / r
            Gs = G + (lam * tr + 1e-3)[:, :, None] * torch.eye(r)[None]
            sol = torch.linalg.solve(Gs, H[:, :, None])[:, :, 0]
            seen = tr[:, 0] > 0; F[f] = torch.where(seen[:, None], sol, F[f])
        log('sweep %d rmse %.4f [%.0fs]' % (sw + 1, fit_err(), time.time() - t0))
    A, B, C = F
    n = min(cfg.get('nq', 20000), len(ctx.yte)); Xq, yq = ctx.Xte[:n], ctx.yte[:n]; nv = min(5000, len(ctx.yva)); Xv, yv = ctx.Xva[:nv], ctx.yva[:nv]
    def rows(X, y):
        q = X[:, -2] * V + X[:, -1]; P = pbi[torch.from_numpy(X[:, -1])].clone()
        for i in range(len(q)):
            lo = np.searchsorted(tu, q[i] * V); hi = np.searchsorted(tu, (q[i] + 1) * V)
            if hi > lo:
                nn = cn[np.searchsorted(cu, q[i])]; tt = ct[np.searchsorted(cu, q[i])]
                P[i] = P[i] * float(delta * tt / nn); P[i, torch.from_numpy(yy[lo:hi])] += torch.from_numpy(np.maximum(tc[lo:hi] - delta, 0) / nn).float()
        S = (A[torch.from_numpy(X[:, -2])] * B[torch.from_numpy(X[:, -1])]) @ C.T
        return P, S
    res = dict(cfg=cfg, r=r, nnz=nnz, rmse=fit_err()); out = {}
    for name, X, y in (('val', Xv, yv), ('test', Xq, yq)):
        P, S = rows(X, y); yt = torch.from_numpy(y); L0 = torch.log(P.clamp_min(1e-9))
        out[name] = {b: bits_of(torch.softmax(L0 + b * S, 1)[torch.arange(len(y)), yt].numpy().astype(np.float64)) for b in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)}
    bstar = min(out['val'], key=out['val'].get); res['beta_star'] = bstar; res['bits_tri'] = out['test'][0.0]; res['bits_cp'] = out['test'][bstar]; res['table_test'] = out['test']
    res['macs_infer_extra'] = 2 * r + V * r; res['time'] = time.time() - t0
    log('tri %.4f  tri+CP(beta=%.2f) %.4f   (test) %s' % (res['bits_tri'], bstar, res['bits_cp'], {k: round(v, 4) for k, v in out['test'].items()}))
    log('RESULT_CP ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
