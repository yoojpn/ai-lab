"""隠れた状態 (繰り上がり) を 数え上げと探索だけで 見つけられるか。 診断用の小さな例だけを使う。 CPU。
課題: 下の桁から a_i, b_i が交互に並び、 = のあとに 和の桁 s_i を下から順に書く (繰り上がりは書かない)。 最後に 最上位の桁 (繰り上がり) を書く。
仕組み: 状態 h (H 個の値) を持つ 有限状態機械。 h_0 = 0。  s_i = F[a_i, b_i, h_i] (数え上げの最頻),  h_(i+1) = G[a_i, b_i, h_i] (探索で決める)。
探索: G を乱数で作り、 表の 1 つの要素ずつ 値を変えて 学習例の 正解数が増えるなら採用 (座標ごとの山登り)。 F は そのつど 数え上げ直す。 勾配なし。
使い方: python3 automaton_search.py [H] [N_TRAIN] [RESTARTS]"""
import sys, random
import numpy as np

H = int(sys.argv[1]) if len(sys.argv) > 1 else 2; NT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; RS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
r = random.Random(0)


def make(n):
    a = [r.randrange(10) for _ in range(n)]; b = [r.randrange(10) for _ in range(n)]; c = 0; s = []
    for i in range(n): v = a[i] + b[i] + c; s.append(v % 10); c = v // 10
    return np.array(a), np.array(b), np.array(s + [c])


train = {n: [make(n) for _ in range(NT // 3)] for n in (2, 3, 4)}
D = {n: (np.stack([e[0] for e in L]), np.stack([e[1] for e in L]), np.stack([e[2] for e in L])) for n, L in train.items()}


def run_states(G, A, B):
    N, n = A.shape; h = np.zeros((N, n + 1), np.int64)
    for i in range(n): h[:, i + 1] = G[A[:, i], B[:, i], h[:, i]]
    return h


def score(G, fit=False):
    cnt = np.zeros((10, 10, H, 10)); cnt2 = np.zeros((H, 10)); hs = {}
    for n, (A, B, S) in D.items():
        h = run_states(G, A, B); hs[n] = h
        for i in range(n): np.add.at(cnt, (A[:, i], B[:, i], h[:, i], S[:, i]), 1)
        np.add.at(cnt2, (h[:, n], S[:, n]), 1)
    ok = cnt.max(3).sum() + cnt2.max(1).sum(); tot = cnt.sum() + cnt2.sum()
    return (ok / tot, cnt, cnt2) if fit else ok / tot


def climb(G):
    best = score(G); improved = True
    while improved:
        improved = False
        for a in range(10):
            for b in range(10):
                for h in range(H):
                    cur = G[a, b, h]
                    for v in range(H):
                        if v == cur: continue
                        G[a, b, h] = v; s = score(G)
                        if s > best + 1e-12: best = s; cur = v; improved = True
                        else: G[a, b, h] = cur
                    G[a, b, h] = cur
    return best, G


best = (0, None)
for k in range(RS):
    G = np.random.default_rng(k).integers(0, H, (10, 10, H)); s, G = climb(G)
    print('  やり直し %d 回目: 学習例の正解率 %.4f' % (k + 1, s), flush=True)
    if s > best[0]: best = (s, G.copy())
    if s > 0.9999: break
s, G = best; _, cnt, cnt2 = score(G, fit=True); F = cnt.argmax(3); F2 = cnt2.argmax(1)
carry_ok = all(G[a, b, 0] == int(a + b >= 10) and G[a, b, 1] == int(a + b + 1 >= 10) for a in range(10) for b in range(10)) if H == 2 else None
print('状態の数 H=%d: 学習例の正解率 %.4f | 状態の推移表が 「繰り上がり」と一致 (状態の名前の入れ替えを除く場合は別): %s' % (H, s, carry_ok))
ok_all = {}
for n in (3, 4, 6, 8, 12, 20):
    ok = tot = 0
    for _ in range(300):
        a, b, S = make(n); h = 0; out = []
        for i in range(n): out.append(int(F[a[i], b[i], h])); h = int(G[a[i], b[i], h])
        out.append(int(F2[h])); ok += out == list(S); tot += 1
    print('桁数 %2d (未知の足し算 %d 問): 状態機械 %.3f' % (n, tot, ok / tot))
