"""
TRACE-ALS: 学習に出てこなかった (一の位, 一の位) の組を、 勾配なしで当てられるか。
  表 (丸暗記) は 見ていない組で当たらない (除いた組 4.6% / 10.3%, exp_trace.py)。
  そこで 各桁の数字に 学習で決める座標 (r 次元) を持たせ、 出力の予測 = Σ_k F1[a,k] F2[b,k] ... W[k,s] とする (CP分解)。
  座標 F と W は 交互最小二乗 (ALS: 片方を固定して、 もう片方を最小二乗で厳密に解く) だけで決める。 勾配降下・逆伝播なし。
  観測 = 学習に出た組の、 正解の one-hot。 評価 = 学習に出なかった組の 正解率 (argmax)。
  sum_ones : 入力 (a1, b1) -> s1 = (a1+b1) mod 10           (100組のうち20組を除く)
  sum_tens : 入力 (a0, b0, a1, b1) -> s0 = (a0+b0+繰り上がり) mod 10   (a1,b1 の組を20個除く: 10^4 組のうち 20% が未観測)
使い方: python3 exp_trace_als.py '{"ranks":[2,4,6,8,10,16,24],"restarts":5}'
"""
import sys, json, time
import numpy as np


def cp_als(idx, Y, dims, r, lam, iters, rng):
    """idx: (cells x m) 各入力の値, Y: (cells x S) one-hot。 戻り値: 因子 (m 個の dims[i] x r), W (r x S)"""
    m = idx.shape[1]; S = Y.shape[1]
    F = [rng.normal(0, 0.5, (dims[i], r)) for i in range(m)]; W = rng.normal(0, 0.5, (r, S))
    for _ in range(iters):
        G = np.ones((len(idx), r))
        for i in range(m):
            G = G * F[i][idx[:, i]]
        A = G.T @ G + lam * np.eye(r); W = np.linalg.solve(A, G.T @ Y)
        for i in range(m):
            g = np.ones((len(idx), r))
            for j in range(m):
                if j != i:
                    g = g * F[j][idx[:, j]]
            for v in range(dims[i]):
                c = np.flatnonzero(idx[:, i] == v)
                if len(c) == 0:
                    continue
                X = (g[c][:, :, None] * W[None, :, :]).transpose(0, 2, 1).reshape(-1, r)       # (cells*S) x r
                y = Y[c].reshape(-1)
                F[i][v] = np.linalg.solve(X.T @ X + lam * np.eye(r), X.T @ y)
    return F, W


def predict(F, W, idx):
    G = np.ones((len(idx), W.shape[0]))
    for i in range(idx.shape[1]):
        G = G * F[i][idx[:, i]]
    return (G @ W).argmax(1), G @ W


def main(cfg):
    rng = np.random.default_rng(0)
    pairs = np.array([(a, b) for a in range(10) for b in range(10)])
    held = pairs[rng.choice(100, cfg.get('nheld', 20), replace=False)]
    held_set = set(map(tuple, held.tolist()))
    t0 = time.time(); out = {}
    # --- 一の位
    obs = np.array([p for p in pairs.tolist() if tuple(p) not in held_set]); hel = held
    yo = (obs.sum(1)) % 10; yh = hel.sum(1) % 10
    Yo = np.eye(10)[yo]
    out['ones'] = {}
    for r in cfg['ranks']:
        best = None
        for s in range(cfg['restarts']):
            F, W = cp_als(obs, Yo, [10, 10], r, cfg.get('lam', 1e-3), cfg.get('iters', 60), np.random.default_rng(100 + s))
            tr = (predict(F, W, obs)[0] == yo).mean(); te = (predict(F, W, hel)[0] == yh).mean()
            if best is None or tr > best[0] + 1e-9:
                best = (tr, te)
        out['ones'][r] = dict(train=best[0], held=best[1]); print('[%4ds] 一の位 rank=%d: 学習 %.3f 除いた組 %.3f' % (time.time() - t0, r, best[0], best[1]), flush=True)
    # --- 十の位 (4入力)
    cells = np.array([(a0, b0, a1, b1) for a0 in range(10) for b0 in range(10) for a1 in range(10) for b1 in range(10)])
    carry = (cells[:, 2] + cells[:, 3]) // 10; y = (cells[:, 0] + cells[:, 1] + carry) % 10
    isheld = np.array([(a1, b1) in held_set for a1, b1 in cells[:, 2:]])
    out['tens'] = {}
    for r in cfg['ranks_tens']:
        best = None
        for s in range(cfg['restarts']):
            F, W = cp_als(cells[~isheld], np.eye(10)[y[~isheld]], [10] * 4, r, cfg.get('lam', 1e-3), cfg.get('iters', 60), np.random.default_rng(200 + s))
            tr = (predict(F, W, cells[~isheld])[0] == y[~isheld]).mean(); te = (predict(F, W, cells[isheld])[0] == y[isheld]).mean()
            if best is None or tr > best[0] + 1e-9:
                best = (tr, te)
        out['tens'][r] = dict(train=best[0], held=best[1]); print('[%4ds] 十の位 rank=%d: 学習 %.3f 除いた組 %.3f' % (time.time() - t0, r, best[0], best[1]), flush=True)
    print('RESULT_TRACE_ALS ' + json.dumps(dict(cfg=cfg, res=out)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
