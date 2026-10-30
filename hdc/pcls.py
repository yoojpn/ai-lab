"""
予測でまとめる語のクラス (勾配なし, 数え上げだけ)。
  語 w を、 「w の次に来る語の分布」 と 「w の前に来た語の分布」 の両方が似たものどうしで同じクラスにまとめる (硬い EM / 交換法に近い)。
  目的 = クラスの2-gram対数尤度 (クラスが決まると、 前の語のクラス c から 次の語 x を P_c(x) で予測する) を増やす向きにだけ動く。
  語の同一性は予測時にも分かっているので、 文脈クラスタのときの「学習では正解を使い、 予測では使えない」ずれは起きない。
  C1[a, b] = 「a の直後に b が来た回数」 (V x V)。
"""
import torch


def predictive_classes(C1, K, init, iters=15, beta=20.0, log=print):
    V = C1.shape[0]; C1 = C1.float(); tot = C1.sum()
    pr0 = (C1.sum(0) + 1) / (tot + V); pl0 = (C1.sum(1) + 1) / (tot + V)
    cls = init.clone().long(); ar = torch.arange(V)
    for it in range(iters):
        M = torch.zeros(V, K); M[ar, cls] = 1.0
        R = M.T @ C1; L = M.T @ C1.T
        Pr = (R + beta * pr0[None, :]) / (R.sum(1, keepdim=True) + beta); Pl = (L + beta * pl0[None, :]) / (L.sum(1, keepdim=True) + beta)
        score = C1 @ torch.log(Pr).T + C1.T @ torch.log(Pl).T
        best, new = score.max(1)
        ll = (score[ar, cls]).sum().item() / tot.item() / 2
        ch = int((new != cls).sum()); cls = new
        log('  予測クラス 反復%d: 語の入れ替わり %d, 現在の割り当ての対数尤度/語 %.4f, 空クラス %d' % (it, ch, ll, int(K - len(torch.unique(cls)))))
        if ch == 0:
            break
    return cls
