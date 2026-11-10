"""生成モデルで 100 個の候補を順位づけ: 得点 = 返事の対数尤度 (前の発言あり) - 返事の対数尤度 (前の発言を 無関係な発言に差し替え)。 返事選び (R@1 0.216) と比べる。"""
import numpy as np, sys
import rpc_gen as G
rng = np.random.default_rng(1); pairs = []
for d in G.test_d:
    for i in range(1, len(d)): pairs.append((d[i - 1], d[i]))
allr = [p[1] for p in pairs]; sel = rng.choice(len(pairs), int(sys.argv[1]) if len(sys.argv) > 1 else 100, replace=False)


def ll(prev, resp, beta):
    T = [0] * 9 + G.enc(prev) + [2]; av = G.assoc(G.enc(prev)); tot = 0.0
    for y in G.enc(resp) + [2]:
        P = G.dist(np.array(T, np.uint16), av, beta); tot += np.log(max(P[y], 1e-12)); T.append(y)
    return tot


null = '。'
for beta in (0.0, 1.0):
    ranks = []
    for i in sel:
        c, r = pairs[i]; neg = [allr[j] for j in rng.choice(len(allr), 99, replace=False) if allr[j] != r][:99]; cand = [r] + neg
        s = np.array([ll(c, x, beta) - ll(null, x, 0.0) for x in cand]); ranks.append(int((s > s[0]).sum() + 1))
    rk = np.array(ranks); G.lg('GENR1_RESULT beta=%.1f: R@1 %.3f  R@5 %.3f  (評価 %d 組, 候補 100 個)' % (beta, (rk == 1).mean(), (rk <= 5).mean(), len(rk)))
