"""鍵の自動探索 (数え上げだけ) を 課題 1〜3 に広げる。 診断用の小さな例だけを使う (本番の学習データには混ぜない)。
全課題で 同じ候補群を使う (課題ごとに 人が鍵を作らない):
  rel(k): 直前の k 語 (全体 / 最後の 1 文字), k=1..8
  diag(A, st, c): tok[A + st*(t-EQ) + c]   A = 先頭 (Q) か = の位置, st = -1, 1, 2, c = -3..3   (= からの進み具合に比例して 指す)
  ind(k): tok[t-k] と同じ語が 前に出ていたら その直後の語 (k=1..3)   (内容で指す)
探索: 3 個までの組を全部試し、 検証用の例で 正解率が最大の組 (同点なら 小さい組) を選ぶ。 予測 = その鍵での最頻の語。
課題: add_chain (途中式つき足し算) / add_plain_msb (上の桁から書く 普通の足し算) / add_lsb_nocarry (下の桁から、 繰り上がりを書かない) / bind (変数の参照) / reverse (反転)
使い方: python3 key_search2.py [課題名 ...]"""
import sys, itertools, random
import numpy as np

V = 256; vocab = {}
tid = lambda s: vocab.setdefault(s, len(vocab) + 1) if s is not None else 0      # 0 = 無し (NA)


def feats(toks, eq, t):
    f = []
    for k in range(1, 9):
        v = toks[t - k] if t - k >= 0 else None; f.append(tid(v)); f.append(tid(v[-1] if v else None))
    for A in (0, eq):
        for st in (-1, 1, 2):
            for c in range(-3, 4):
                j = A + st * (t - eq) + c; f.append(tid(toks[j] if 0 <= j < t else None))
    for k in (1, 2, 3):
        x = toks[t - k] if t - k >= 0 else None; r = None
        if x is not None:
            for p in range(t - k - 1, -1, -1):
                if toks[p] == x: r = toks[p + 1] if p + 1 < t else None; break
        f.append(tid(r))
    return f


NAMES = None


def fnames():
    n = []
    for k in range(1, 9): n += ['rel%d' % k, 'rel%d_last' % k]
    for A in ('Q', 'EQ'):
        for st in (-1, 1, 2):
            for c in range(-3, 4): n.append('diag(%s,%d,%+d)' % (A, st, c))
    for k in (1, 2, 3): n.append('ind%d' % k)
    return n


def rows_of(exs):
    X = []; Y = []
    for toks, eq in exs:
        for t in range(eq + 1, len(toks)): X.append(feats(toks, eq, t)); Y.append(tid(toks[t]))
    return np.array(X, np.int64), np.array(Y, np.int64)


def majority(key, y):
    pair = key * V + y; u, c = np.unique(pair, return_counts=True); k = u // V; yy = u % V; o = np.lexsort((-c, k)); k = k[o]; yy = yy[o]; first = np.r_[True, k[1:] != k[:-1]]
    return k[first], yy[first]


def search(X, Y, kmax=3, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(Y)); cut = int(len(Y) * 0.7); tr, va = idx[:cut], idx[cut:]; nf = X.shape[1]; best = (-1, ())
    for k in range(1, kmax + 1):
        for sub in itertools.combinations(range(nf), k):
            key = np.zeros(len(Y), np.int64)
            for s in sub: key = key * V + X[:, s]
            ks, ys = majority(key[tr], Y[tr]); q = key[va]; i = np.minimum(np.searchsorted(ks, q), len(ks) - 1); ok = ((ks[i] == q) & (ys[i] == Y[va])).mean()
            if ok > best[0] + 1e-9: best = (ok, sub)
    return best


class Table:
    def __init__(s, X, Y, sub):
        s.sub = sub; key = np.zeros(len(Y), np.int64)
        for c in sub: key = key * V + X[:, c]
        s.ks, s.ys = majority(key, Y)

    def pred(s, f):
        key = 0
        for c in s.sub: key = key * V + f[c]
        i = min(int(np.searchsorted(s.ks, key)), len(s.ks) - 1)
        return int(s.ys[i]) if s.ks[i] == key else None


inv = lambda: {v: k for k, v in vocab.items()}


def generate(tab, prompt, eq, maxlen=80):
    toks = list(prompt); I = inv()
    for _ in range(maxlen):
        y = tab.pred(feats(toks, eq, len(toks)))
        if y is None: return None
        toks.append(I[y])
        if I[y] == 'E': break
    return toks


class Ngram:
    def __init__(s, exs, N=12):
        import collections
        s.N = N; s.t = [collections.defaultdict(collections.Counter) for _ in range(N + 1)]
        for toks, eq in exs:
            for t in range(eq + 1, len(toks)):
                for k in range(N + 1):
                    if t - k >= 0: s.t[k][tuple(toks[t - k:t])][toks[t]] += 1

    def gen(s, prompt):
        toks = list(prompt)
        for _ in range(80):
            y = None
            for k in range(s.N, -1, -1):
                if len(toks) >= k:
                    c = s.t[k].get(tuple(toks[len(toks) - k:]))
                    if c: y = c.most_common(1)[0][0]; break
            if y is None: return None
            toks.append(y)
            if y == 'E': break
        return toks


# ---- 課題 ----
def dig(r, n): return [r.randrange(10) for _ in range(n)]


def t_add_chain(r, n):
    a, b = dig(r, n), dig(r, n); toks = ['Q']
    for i in range(n): toks += [str(a[i]), str(b[i])]
    toks.append('='); eq = len(toks) - 1; c = 0
    for i in range(n): v = a[i] + b[i] + c; toks.append('%d/%d' % (v % 10, v // 10)); c = v // 10
    toks.append('E'); return toks, eq


def t_add_plain_msb(r, n):
    a, b = dig(r, n), dig(r, n); s = int(''.join(map(str, a))) + int(''.join(map(str, b))); toks = ['Q'] + [str(x) for x in a] + ['+'] + [str(x) for x in b] + ['=']; eq = len(toks) - 1
    toks += list(str(s).zfill(n + 1)) + ['E']; return toks, eq


def t_add_lsb_nocarry(r, n):
    a, b = dig(r, n), dig(r, n); toks = ['Q']
    for i in range(n): toks += [str(a[i]), str(b[i])]
    toks.append('='); eq = len(toks) - 1; c = 0
    for i in range(n): v = a[i] + b[i] + c; toks.append(str(v % 10)); c = v // 10
    toks.append(str(c)); toks.append('E'); return toks, eq


LET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def t_bind(r, n, names=LET):
    ns = r.sample(names, n); vs = dig(r, n); toks = ['Q']
    for a, b in zip(ns, vs): toks += [a, str(b)]
    q = r.randrange(n); toks += ['?', ns[q], '=']; eq = len(toks) - 1; toks += [str(vs[q]), 'E']; return toks, eq


def t_reverse(r, n):
    d = [str(x) for x in dig(r, n)]; toks = ['Q'] + d + ['=']; eq = len(toks) - 1; toks += d[::-1] + ['E']; return toks, eq


def run(name):
    r = random.Random(1); NT = 2000
    if name == 'add_chain': mk = t_add_chain; trn = [2, 3, 4]; tests = [3, 4, 6, 8, 12]
    elif name == 'add_plain_msb': mk = t_add_plain_msb; trn = [2, 3, 4]; tests = [3, 4, 6, 8]
    elif name == 'add_lsb_nocarry': mk = t_add_lsb_nocarry; trn = [2, 3, 4]; tests = [3, 4, 6, 8]
    elif name == 'bind': mk = lambda r, n, nm=LET[:16]: t_bind(r, n, nm); trn = [3, 4, 5]; tests = [3, 5, 7, 9]
    elif name == 'reverse': mk = t_reverse; trn = [3, 4, 5, 6]; tests = [4, 6, 8, 10, 14]
    train = [mk(r, r.choice(trn)) for _ in range(NT)]; seen = set(tuple(t[:e + 1]) for t, e in train)
    X, Y = rows_of(train); sc, sub = search(X, Y); tab = Table(X, Y, sub); ng = Ngram(train); nm = fnames()
    print('== %s: 学習例 %d, 候補 %d, 選ばれた鍵 %s (検証 %.3f)' % (name, NT, len(nm), [nm[s] for s in sub], sc), flush=True)
    for n in tests:
        okk = okn = tot = 0
        for _ in range(300):
            if name == 'bind' and n >= 3: toks, eq = t_bind(r, n, LET[17:]) if n in (7, 9) else mk(r, n)     # 7, 9 個は 学習に出ない名前 (R..Z) だけで作る (未知の名前)
            else: toks, eq = mk(r, n)
            if tuple(toks[:eq + 1]) in seen: continue
            tot += 1; okk += generate(tab, toks[:eq + 1], eq) == toks; okn += ng.gen(toks[:eq + 1]) == toks
        print('   長さ %2d (未知の入力 %d 問): 鍵の探索 %.3f | 直前の語を丸ごと %.3f' % (n, tot, okk / max(tot, 1), okn / max(tot, 1)), flush=True)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['add_chain', 'add_plain_msb', 'add_lsb_nocarry', 'bind', 'reverse']): run(nm)
