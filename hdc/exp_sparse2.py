"""
(実験E2: 難しい試験版。ヒントは直前32語。直前8語は位置つきで、その前の24語は平均して使う)
実験E: ③スパース活性。隠れ層の「上位k個だけ」を動かす小さな言語モデルで、動かす割合と精度の関係を測る。
  文脈=直前3語 → 埋め込み(64次元) → 隠れ層H(ReLU) → 上位k個以外を0 → 出力層(Hから語彙V)
  k=H なら通常(密)。k<H がスパース。出力層の計算量は k×V なので、kを小さくするほど軽い。
  ※ ここでは「精度が保てるか」だけを測る(上位kの選び方は全部計算してから選ぶので、第1層は密のまま)。
    第1層も軽くできるか(選び方の学習)は、次の段階。
使い方: LADDER_CORPUS=... python3 exp_sparse.py <訓練語数N> <H> <k> [エポック数=1]
"""
import os, sys, time, math
sys.path.insert(0, '/home/claude/hdc')
from collections import Counter
import numpy as np
import torch
import exp_scale as es
import exp_far as XF

V = es.V; HIST = 4  # lwindows_meta の窓は先頭に履歴を持つ。末尾3語だけ使う
DE = 64; CTX = 8; LONG = 32; B = 512


def log(s):
    print(s, flush=True)


class Net(torch.nn.Module):
    def __init__(s, H, k):
        super().__init__()
        s.E = torch.nn.Embedding(V, DE)
        s.l1 = torch.nn.Linear((CTX + 1) * DE, H)
        s.l2 = torch.nn.Linear(H, V)
        s.H, s.k = H, k

    def forward(s, X):
        e = s.E(X.long())
        near = e[:, -CTX:].reshape(len(X), -1)
        far = e[:, :-CTX].mean(1)
        h = torch.relu(s.l1(torch.cat([near, far], 1)))
        if s.k < s.H:
            thr = torch.topk(h, s.k, dim=1).values[:, -1:]
            h = h * (h >= thr)
        return s.l2(h)


def main():
    N = int(sys.argv[1]); H = int(sys.argv[2]); k = int(sys.argv[3])
    ep = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    torch.set_num_threads(os.cpu_count() or 2)
    torch.manual_seed(0)
    t0 = time.time()
    corp = es.build_corpus()
    cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr_i, te_i = conv(corp[2]), conv(corp[0])
    def win(mods, limit):
        X, Y, tot = [], [], 0
        for a in mods:
            n = len(a)
            if n <= LONG:
                continue
            idx = np.arange(LONG)[None, :] + np.arange(n - LONG)[:, None]
            X.append(a[idx].astype(np.int16)); Y.append(a[LONG:].astype(np.int64)); tot += n - LONG
            if tot >= limit:
                break
        return torch.from_numpy(np.concatenate(X)[:limit]), torch.from_numpy(np.concatenate(Y)[:limit])
    Xtr, ytr = win(tr_i, N)
    Xte, yte = win(te_i, es.N_TEST)
    net = Net(H, k)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    steps = int(ep * (len(Xtr) // B))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1.0 if t < 0.6 * steps else 0.5)
    step = 0
    while step < steps:
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm) - B + 1, B):
            ix = perm[i:i + B]
            loss = torch.nn.functional.cross_entropy(net(Xtr[ix]), ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if step >= steps:
                break
    with torch.no_grad():
        tot = 0.0
        for i in range(0, len(Xte), 2048):
            tot += torch.nn.functional.cross_entropy(net(Xte[i:i + 2048]), yte[i:i + 2048], reduction='sum').item()
    bits = tot / len(Xte) / math.log(2)
    I = (CTX + 1) * DE
    macs_dense = I * H + H * V
    macs_sparse2 = I * H + k * V      # 第1層は密のまま
    macs_routed = k * I + k * V     # 第1層も動く所だけ(理想)
    params = sum(p.numel() for p in net.parameters())
    log('H=%d k=%d (%.1f%%動く) N=%d: %.3f bit/語 | 1語の積和 密%.2fM 出力層だけ疎%.2fM 理想%.2fM | パラメーター%.1fM | %.0fs'
        % (H, k, 100 * k / H, N, bits, macs_dense / 1e6, macs_sparse2 / 1e6, macs_routed / 1e6, params / 1e6, time.time() - t0))


if __name__ == '__main__':
    main()
