"""
K: スペクトル学習(Hsu-Kakade-Zhang 2009, 観測可能作用素モデル)。 勾配なし・1回の閉形式で、状態をもつ系列モデルを作る。
  P21[i,j] = Pr[x2=i, x1=j]  (隣り合う2語の確率の行列)。  SVD:  P21 = U S V^T、上位m個を使う。
  W = V S^-1。 語 x ごとの m x m 行列  B_x = sum_{(a,x,c)} Pr[a,x,c] * U[c]^T (x) W[a]   (3語の確率から、1回数えて作る)
  b1 = U^T P1、 binf = S^-1 V^T P1。  状態の更新: b <- B_x b / (binf . B_x b)   (1語あたり m^2 回)
  次の語の確率:  p(y) ∝ binf^T B_y b    (負になりうるので 0 に切って正規化)
計算量/語: 履歴を持つなら更新 m^2 + 採点 V*m。 文脈の長さに依らない。 学習: 数える+SVD+行列積(勾配降下なし)。
比較: 直前1語の平滑化2-gram(Prior)、両者の混合(混合比は検証窓で決め、テスト窓で報告)。
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, log, HIST
from exp_r import Prior


def bits_of(p):
    return float(-np.log2(np.clip(p, 1e-12, 1)).mean())


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); t0 = time.time()
    ctx = lab.load(cfg['N']); st = np.concatenate(ctx.tr_i)[:cfg['N']].astype(np.int64); N = len(st)
    C21 = np.bincount(st[1:] * V + st[:-1], minlength=V * V).reshape(V, V).astype(np.float64); P21 = C21 / C21.sum()
    P1 = np.bincount(st[:-1], minlength=V).astype(np.float64); P1 /= P1.sum()
    U, S, Vt = np.linalg.svd(P21, full_matrices=False); log('svd %.0fs' % (time.time() - t0))
    key = (st[1:-1] * V + st[2:]) * V + st[:-2]
    u, cn = np.unique(key, return_counts=True); a = u % V; c = (u // V) % V; x = u // (V * V); w = cn / cn.sum()
    xs, first = np.unique(x, return_index=True); last = np.r_[first[1:], len(x)]
    log('trigrams %d [%.0fs]' % (len(u), time.time() - t0))
    n = min(cfg.get('nq', 20000), len(ctx.yte)); Xq, yq = ctx.Xte[:n], ctx.yte[:n]
    nv = min(5000, len(ctx.yva)); Xv, yv = ctx.Xva[:nv], ctx.yva[:nv]
    pr = Prior(st)
    def bigram_true(X, y): return pr.logp(torch.from_numpy(X[:, -1])).exp().numpy()[np.arange(len(y)), y]
    pbi_q, pbi_v = bigram_true(Xq, yq), bigram_true(Xv, yv)
    tk = (st[:-2] * V + st[1:-1]) * V + st[2:]; tu, tc = np.unique(tk, return_counts=True)
    cx = tu // V; cu, cinv = np.unique(cx, return_inverse=True); cn_ = np.bincount(cinv, weights=tc); ct_ = np.bincount(cinv).astype(np.float64)
    def trigram_true(X, y, pbi, delta=0.75):
        q = X[:, -2] * V + X[:, -1]; pos = np.searchsorted(cu, q).clip(max=len(cu) - 1); okc = cu[pos] == q
        n_ = np.where(okc, cn_[pos], 0.0); T_ = np.where(okc, ct_[pos], 0.0)
        kk = q * V + y; p2 = np.searchsorted(tu, kk).clip(max=len(tu) - 1); c_ = np.where(tu[p2] == kk, tc[p2], 0.0)
        return np.where(n_ > 0, np.maximum(c_ - delta, 0) / np.maximum(n_, 1) + delta * T_ / np.maximum(n_, 1) * pbi, pbi)
    ptri_q, ptri_v = trigram_true(Xq, yq, pbi_q), trigram_true(Xv, yv, pbi_v)
    res = dict(cfg=cfg, bits_bigram=bits_of(pbi_q), bits_trigram=bits_of(ptri_q), rows=[]); log('bigram %.4f trigram %.4f' % (res['bits_bigram'], res['bits_trigram']))
    for m in cfg.get('ms', [32, 64, 128, 256]):
        Um = U[:, :m]; Sm = S[:m]; Wm = Vt[:m].T / Sm[None, :]
        Ut, Wt = torch.from_numpy(Um).float(), torch.from_numpy(Wm).float()
        B = torch.zeros(V, m, m)
        for xi, f, l in zip(xs, first, last):
            B[xi] = (Ut[c[f:l]] * torch.from_numpy(w[f:l]).float()[:, None]).T @ Wt[a[f:l]]
        b1 = torch.from_numpy(Um.T @ P1).float(); binf = torch.from_numpy((Vt[:m] @ P1) / Sm).float()
        Cy = torch.einsum('p,ypq->yq', binf, B)
        def probs(X, y, h=HIST):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 500):
                    xb = torch.from_numpy(X[i:i + 500]); b = b1[None].repeat(len(xb), 1)
                    for j in range(X.shape[1] - h, X.shape[1]):
                        b = torch.bmm(B[xb[:, j]], b[:, :, None])[:, :, 0]
                        z = (b @ binf)[:, None]; b = b / torch.where(z.abs() > 1e-12, z, torch.ones_like(z))
                    s = (b @ Cy.T).clamp_min(1e-9); s = s / s.sum(1, keepdim=True)
                    out.append(s[torch.arange(len(xb)), torch.from_numpy(y[i:i + 500])].numpy())
            return np.concatenate(out).astype(np.float64)
        for h in cfg.get('hs', [1, 2, 4, 12]):
            pq, pv = probs(Xq, yq, h), probs(Xv, yv, h)
            best = min(((bits_of(l * pv + (1 - l) * pbi_v), l) for l in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0)))
            b3 = min(((bits_of(l * pv + (1 - l) * ptri_v), l) for l in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)))
            lam = best[1]; row = dict(m=m, h=h, bits_spec=bits_of(pq), lam=lam, bits_mix=bits_of(lam * pq + (1 - lam) * pbi_q), lam_tri=b3[1], bits_mix_tri=bits_of(b3[1] * pq + (1 - b3[1]) * ptri_q),
                                      macs_update=h * m * m, macs_stream=m * m + V * m)
            res['rows'].append(row); log('m=%d h=%d spec %.4f mix(bi) %.4f mix(tri) %.4f (lam_tri %.2f) [%.0fs]' % (m, h, row['bits_spec'], row['bits_mix'], row['bits_mix_tri'], row['lam_tri'], time.time() - t0))
    log('RESULT_SPEC ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
