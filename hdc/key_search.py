"""鍵の自動探索 (数え上げだけ) で 見たことのない足し算が解けるかの試験。 CPU。 診断用の小さな例だけを使う (本番の学習データには混ぜない)。
並べ方: Q a0 b0 a1 b1 ... (下の桁から) = T0 T1 ... E   T_i = (和の桁 s_i, 繰り上がり c_{i+1}) を 1 語にしたもの。
鍵の候補: 直前の k 語 (全体 / 最後の 1 文字), 「Q からの距離 = stride * (= からの距離) + c」 で指す語 (stride 1,2)。
探索: 候補から 3 個までの組を全部試し、 検証用の例で 正解率が最大の組を選ぶ。 勾配なし。 予測 = その鍵での最頻の語。
比較: 直前 k 語を丸ごと鍵にする n-gram (最長一致)。
使い方: python3 key_search.py [N_TRAIN] [SEED]"""
import sys, itertools, collections, random
import numpy as np

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = random.Random(SEED)


def make(a, b):
    """a, b: 桁の列 (下の桁から)。 戻り値: (トークン列, = の位置)"""
    n = len(a); toks = ['Q']
    for i in range(n): toks += [str(a[i]), str(b[i])]
    toks.append('='); eq = len(toks) - 1; c = 0
    for i in range(n):
        v = a[i] + b[i] + c; toks.append('%d/%d' % (v % 10, v // 10)); c = v // 10
    toks.append('E'); return toks, eq


def sample(n):
    a = [rng.randrange(10) for _ in range(n)]; b = [rng.randrange(10) for _ in range(n)]; return a, b


def feats(toks, eq, t):
    """位置 t の語を予測するときの 特徴 (名前 -> 値)。 値が無いときは 'NA'"""
    f = {}
    for k in range(1, 13):
        v = toks[t - k] if t - k >= 0 else 'NA'; f['rel%d' % k] = v; f['rel%d_last' % k] = v[-1] if v != 'NA' else 'NA'
    for st in (1, 2):
        for c in range(-2, 4):
            j = 1 + st * (t - eq - 1) + c + 0; v = toks[j] if 0 <= j < t else 'NA'; f['diag%d_%d' % (st, c)] = v
    return f


def rows_of(examples):
    R = []
    for toks, eq in examples:
        for t in range(eq + 1, len(toks)): R.append((feats(toks, eq, t), toks[t]))
    return R


def best_key(rows, names, kmax=3, seed=0):
    r = random.Random(seed); idx = list(range(len(rows))); r.shuffle(idx); cut = int(len(idx) * 0.7); tr = [rows[i] for i in idx[:cut]]; va = [rows[i] for i in idx[cut:]]; best = None
    for k in range(1, kmax + 1):
        for sub in itertools.combinations(names, k):
            cnt = collections.defaultdict(collections.Counter)
            for f, y in tr: cnt[tuple(f[s] for s in sub)][y] += 1
            ok = 0
            for f, y in va:
                c = cnt.get(tuple(f[s] for s in sub))
                if c and c.most_common(1)[0][0] == y: ok += 1
            sc = ok / len(va)
            if best is None or sc > best[0] + 1e-9: best = (sc, sub)
    return best


def fit_table(rows, sub):
    cnt = collections.defaultdict(collections.Counter)
    for f, y in rows: cnt[tuple(f[s] for s in sub)][y] += 1
    return {k: c.most_common(1)[0][0] for k, c in cnt.items()}


def gen_key(table, sub, prompt, eq):
    toks = list(prompt)
    for _ in range(60):
        f = feats(toks, eq, len(toks)); y = table.get(tuple(f[s] for s in sub))
        if y is None: return None
        toks.append(y)
        if y == 'E': break
    return toks


class Ngram:
    def __init__(s, examples, N=12):
        s.N = N; s.t = [collections.defaultdict(collections.Counter) for _ in range(N + 1)]
        for toks, eq in examples:
            for t in range(eq + 1, len(toks)):
                for k in range(0, N + 1):
                    if t - k >= 0: s.t[k][tuple(toks[t - k:t])][toks[t]] += 1

    def gen(s, prompt):
        toks = list(prompt)
        for _ in range(60):
            y = None
            for k in range(s.N, -1, -1):
                if len(toks) >= k:
                    c = s.t[k].get(tuple(toks[len(toks) - k:])) if k else None
                    if c: y = c.most_common(1)[0][0]; break
            if y is None: return None
            toks.append(y)
            if y == 'E': break
        return toks


def main():
    train = []
    for _ in range(NT): train.append(make(*sample(rng.choice([2, 3, 4]))))
    seen = set(tuple(t[:e]) for t, e in train); names = list(feats(train[0][0], train[0][1], train[0][1] + 1).keys())
    rows = rows_of(train); sc, sub = best_key(rows, names); table = fit_table(rows, sub); ng = Ngram(train)
    print('学習例 %d (桁数 2-4), 候補 %d, 選ばれた鍵 %s (検証正解率 %.3f)' % (NT, len(names), sub, sc))
    for n in (3, 4, 5, 6, 8, 12):
        ok_k = ok_n = tot = 0
        for _ in range(300):
            a, b = sample(n); toks, eq = make(a, b)
            if tuple(toks[:eq + 1]) in seen: continue
            tot += 1; g = gen_key(table, sub, toks[:eq + 1], eq); ok_k += (g == toks); g2 = ng.gen(toks[:eq + 1]); ok_n += (g2 == toks)
        print('桁数 %2d (未知の足し算 %d 問): 鍵の探索 %.3f | 直前の語を丸ごと(最長一致) %.3f' % (n, tot, ok_k / tot, ok_n / tot))


main()
