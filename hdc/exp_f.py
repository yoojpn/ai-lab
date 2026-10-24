"""
F群: 勾配ゼロで作るモデル。「数えて書き込むだけ」の学習計算量で、どこまで品質が出るか。
  f1 <N> : 単語n-gram(数えるだけ) + 文書内キャッシュ/コピー + 索引(最近傍) を、EMで重みだけ決めて混ぜる。
           NN(勾配)は使わない。出力: 各構成のbit/語と、学習の計算量の見積り(演算回数)。
  比較用: 勾配で学ぶ小さいNN(H=64, N語)のbit/語は別実験(d2)で 6.04(N=16M)。
使い方: LADDER_CORPUS=... python3 exp_f.py f1 <N> [nq=20000]
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab, items
from lab import V, HIST, DE, bits, log
import exp_scale as es, exp_cards as XC, exp_ladder as L

es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


def f1(ctx, nq):
    t0 = time.time(); torch.set_num_threads(os.cpu_count() or 2)
    N = len(ctx.Xtr)
    res = dict(mode='f1', N=N, nq=nq)
    w3 = lambda X: X[:, HIST - 3:]
    # --- 1. 数える(n-gram)。演算回数の見積り = 並べ替え N*log2(N) を各次数で
    ng = es.NG(w3(ctx.Xtr), ctx.ytr)
    res['ops_count'] = float(4 * N * np.log2(N))
    pu = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1.0; pu /= pu.sum()
    sl = slice(0, nq)
    def ngp(X, y):
        r = ng.probs(w3(X), y); return r[0] if isinstance(r, tuple) else r
    ng_va, ng_te = ngp(ctx.Xva[sl], ctx.yva[sl]), ngp(ctx.Xte[sl], ctx.yte[sl])
    res['ngram'] = bits(ng_te)
    log('n-gram のみ: %.4f [%.0fs]' % (res['ngram'], time.time() - t0))
    # --- 2. 文書内のコピー/キャッシュ(学習なし)
    cp_va, _ = XC.copy_probs(ctx.Xva[sl], ctx.yva[sl], pu); cp_te, _ = XC.copy_probs(ctx.Xte[sl], ctx.yte[sl], pu)
    ca_va = XC.cache_probs(ctx.Xva[sl], ctx.yva[sl], pu); ca_te = XC.cache_probs(ctx.Xte[sl], ctx.yte[sl], pu)
    Pva, Pte = [ng_va, cp_va, ca_va], [ng_te, cp_te, ca_te]
    w = L.em(np.array(Pva)); res['ngram+cache'] = bits((w[:, None] * np.array(Pte)).sum(0))
    log('n-gram+cache: %.4f [%.0fs]' % (res['ngram+cache'], time.time() - t0))
    # --- 3. 索引(最近傍)。キー=直前3語のPPMI埋め込み(SVD)。SVDの演算回数 ~ 2*V*V*DE*7
    E0 = items.ppmi_embeddings(ctx, DE)
    res['ops_svd'] = float(2 * V * V * DE * 7)
    def keys(X):
        k = E0[X[:, -3:]].reshape(len(X), -1).astype(np.float32)
        return k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)
    Kd = torch.cat([torch.from_numpy(np.round(keys(ctx.Xtr[i:i + 500000]) * 127).astype(np.int8)) for i in range(0, N, 500000)])   # int8で3GB(32M語)
    yd = torch.from_numpy(ctx.ytr)
    res['ops_index'] = float(N * 3 * DE)          # 鍵を作る = 埋め込み引きと正規化(数回のN倍)
    def knn(Xq, yq, T, k=64):
        Q = torch.from_numpy(keys(Xq)); out = np.zeros(len(Q))
        for qi in range(0, len(Q), 500):
            qb = Q[qi:qi + 500]; bs = bi = None
            for c in range(0, N, 500000):
                s = qb @ (Kd[c:c + 500000].float() / 127).T
                v, ix = s.topk(k, 1); ix = ix + c
                if bs is None: bs, bi = v, ix
                else:
                    bs, j = torch.cat([bs, v], 1).topk(k, 1); bi = torch.cat([bi, ix], 1).gather(1, j)
            wt = torch.softmax(bs / T, 1)
            out[qi:qi + 500] = (wt * (yd[bi] == torch.from_numpy(yq[qi:qi + 500])[:, None])).sum(1).numpy()
        return np.maximum(out, 0) * 0.98 + 0.02 * pu[yq]       # 当たりの確率 + 少しだけ全体頻度
    best = None
    for T in (0.02, 0.05):
        kv = knn(ctx.Xva[sl], ctx.yva[sl], T)
        w = L.em(np.array(Pva + [kv]))
        b = bits((w[:, None] * np.array(Pva + [kv])).sum(0))
        if best is None or b < best[0]: best = (b, T, w)
    _, T, w = best
    kt = knn(ctx.Xte[sl], ctx.yte[sl], T)
    res['ngram+cache+index'] = bits((w[:, None] * np.array(Pte + [kt])).sum(0)); res['T'] = T; res['w'] = [float(x) for x in w]
    res['ops_total_train'] = res['ops_count'] + res['ops_svd'] + res['ops_index']
    res['sec'] = time.time() - t0
    log('n-gram+cache+索引: %.4f [%.0fs] | 学習の演算回数 約%.2e' % (res['ngram+cache+index'], time.time() - t0, res['ops_total_train']))
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    mode = sys.argv[1]; N = int(sys.argv[2]); nq = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    ctx = lab.load(N)
    {'f1': f1}[mode](ctx, nq)
