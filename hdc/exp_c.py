"""
C: 勾配ゼロ・1パスの「汎化」部品。
  trig : トリガー対の対数線形モデル。直前から k 語前の語 w が「次の語 y」をどれだけ予告するか(PMI)を、
         k=1..12 で表にして(1回読むだけ)、重み a_k を 13個の数だけ検証データで合わせて掛け合わせる。
         p(y|文脈) ∝ exp( sum_k a_k * log[ P_k(y|w_k)/P(y) ] + b log P(y) )
  rfr  : ランダム特徴 + リッジ回帰(閉形式)。直前語の埋め込みを入力に、ReLUのランダム特徴 D 個を作り、
         次の語(one-hot)への線形回帰を1回の行列計算で解く。学習FLOP ~ 2*N*D^2。
いずれも「訓練データを1回読む」だけで、勾配降下を使わない。
"""
import time, numpy as np, torch
from lab import V, HIST, bits, log


def trig(ctx, Xva, yva, Xte, yte, pu, K=12, alpha=5.0, nfit=6000):
    t0 = time.time(); N = len(ctx.ytr); pu32 = pu.astype(np.float32)
    LP = []
    for k in range(1, K + 1):
        x = ctx.Xtr[:, HIST - k].astype(np.int64)
        C = np.bincount(x * V + ctx.ytr, minlength=V * V).astype(np.float32).reshape(V, V)
        R = C.sum(1, keepdims=True)
        LP.append(np.log((C + alpha * pu32[None, :]) / (R + alpha) / pu32[None, :]).astype(np.float32)); del C
    log('trig 表(k=1..%d) [%.0fs]' % (K, time.time() - t0))
    lpu = torch.from_numpy(np.log(pu32))

    def feats(X, rows):
        return torch.from_numpy(np.stack([LP[k - 1][X[rows, HIST - k]] for k in range(1, K + 1)], 1))   # (n,K,V)
    a = torch.full((K,), 1.0 / K, requires_grad=True); b = torch.ones(1, requires_grad=True)
    rows = np.arange(min(nfit, len(yva))); F = feats(Xva, rows); yt = torch.from_numpy(yva[rows])
    opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=40, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(); s = torch.einsum('nkv,k->nv', F, a) + b * lpu
        loss = torch.nn.functional.cross_entropy(s, yt); loss.backward(); return loss
    opt.step(closure); del F
    log('trig 重み a=%s b=%.3f [%.0fs]' % (np.round(a.detach().numpy(), 3).tolist(), b.item(), time.time() - t0))

    def prob(X, y):
        out = np.zeros(len(y))
        with torch.no_grad():
            for i in range(0, len(y), 1000):
                r = np.arange(i, min(i + 1000, len(y)))
                s = torch.einsum('nkv,k->nv', feats(X, r), a) + b * lpu
                out[r] = torch.softmax(s, 1)[torch.arange(len(r)), torch.from_numpy(y[r])].numpy()
        return out
    pv, pt = prob(Xva, yva), prob(Xte, yte)
    return pv, pt, float(K * N * 3 + K * V * V), dict(a=[float(x) for x in a.detach()], b=float(b))


def rfr(ctx, E0, Xva, yva, Xte, yte, D=1536, near=8, lam=1e-2, seed=0):
    t0 = time.time(); torch.set_num_threads(__import__('os').cpu_count() or 2)
    de = E0.shape[1]; din = near * de + de
    E = torch.from_numpy(E0.astype(np.float32)); g = torch.Generator().manual_seed(seed)
    Wr = torch.randn(din, D, generator=g) / np.sqrt(din); br = torch.randn(D, generator=g) * 0.3

    def phi(X):
        X = torch.from_numpy(X)
        x = torch.cat([E[X[:, -near:]].reshape(len(X), -1), E[X[:, :4]].mean(1)], 1)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        return torch.cat([torch.relu(x @ Wr + br), torch.ones(len(X), 1)], 1)
    G = torch.zeros(D + 1, D + 1, dtype=torch.float64); B = torch.zeros(V, D + 1)
    N = len(ctx.ytr); ch = 40000
    for i in range(0, N, ch):
        P = phi(ctx.Xtr[i:i + ch]); G += (P.T @ P).double(); B.index_add_(0, torch.from_numpy(ctx.ytr[i:i + ch]), P)
    log('rfr Gram完了 D=%d [%.0fs]' % (D, time.time() - t0))
    Wt = torch.linalg.solve(G + lam * torch.diag(G).mean() * torch.eye(D + 1, dtype=torch.float64), B.double().T).float()   # (D+1,V)

    def prob(X, y, eps):
        out = np.zeros(len(y))
        with torch.no_grad():
            for i in range(0, len(y), 4000):
                S = (phi(X[i:i + 4000]) @ Wt).clamp_min(eps); S = S / S.sum(1, keepdim=True)
                out[i:i + 4000] = S[torch.arange(len(S)), torch.from_numpy(y[i:i + 4000])].numpy()
        return out
    best = min(((bits(prob(Xva[:5000], yva[:5000], e)), e) for e in (1e-5, 1e-4, 1e-3)))
    eps = best[1]
    return prob(Xva, yva, eps), prob(Xte, yte, eps), float(2.0 * N * (D + 1) ** 2 + N * din * D * 2), dict(eps=eps, val_bits=best[0], lam=lam)
