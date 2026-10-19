"""
実験2: 単語単位で、超次元kNN(意味の層+引く層)は n-gram に勝てるか/補えるか。

コーパス: Python標準ライブラリのdocstring(英語の技術文書)。モジュール単位で
  訓練(80%)/検証(10%)/テスト(10%)に分けるので、テストは「見たことのないモジュールの文章」。
基準: 補間つき絶対値割引の4-gram(直前3単語)。語彙は頻度上位V語、他は<unk>。
kNN: 直前3単語の3個のベクトルの多数決→ハミング距離で近い32件の「次の単語」の重みつき多数決。
変種 R: 単語ベクトルはランダム。 変種 L: 周囲±2語の分布から作る(似た使われ方=近いベクトル)。
"""
import ast, hashlib, os, re, sys, time, sysconfig
import numpy as np

D = 512; W = D // 64; CTX = 3; TOPK = 32; TAU = 6.0; V = 5000; DELTA = 0.75
LAMS = [0, .05, .1, .2, .3, .5, .7, .9]
DEDGES = [0, 60, 130, 190]
N_TRAIN, N_VAL, N_TEST = 300_000, 6_000, 10_000
TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")


def corpus():
    root = sysconfig.get_paths()['stdlib']
    tr, va, te = [], [], []
    files = []
    for dp, dn, fn in os.walk(root):
        if 'site-packages' in dp:
            continue
        dn.sort()
        for f in sorted(fn):
            if f.endswith('.py'):
                files.append(os.path.join(dp, f))
    for p in files:
        try:
            tree = ast.parse(open(p, encoding='utf-8').read())
        except Exception:
            continue
        ds = [ast.get_docstring(n) for n in ast.walk(tree)
              if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        toks = TOK.findall('\n'.join(d for d in ds if d).lower())
        if len(toks) < 400:
            continue
        h = int(hashlib.md5(p.encode()).hexdigest(), 16) % 10
        (te if h == 0 else va if h == 1 else tr).append(toks)
    return tr, va, te


def take(mods, n):
    out, tot = [], 0
    for m in mods:
        if tot >= n:
            break
        out.append(m); tot += len(m)
    return out


class NGram:
    def __init__(self, order=CTX):
        self.o = order
        self.t = [dict() for _ in range(order + 1)]

    def train(self, a):
        a = list(map(int, a))
        for i in range(len(a)):
            for k in range(min(i, self.o) + 1):
                h = tuple(a[i - k:i])
                e = self.t[k].get(h)
                if e is None:
                    self.t[k][h] = [{a[i]: 1}, 1]
                else:
                    e[0][a[i]] = e[0].get(a[i], 0) + 1
                    e[1] += 1

    def prob(self, ctx, w):
        p = 1.0 / V
        maxk = 0
        for k in range(self.o + 1):
            h = ctx[len(ctx) - k:] if k else ()
            e = self.t[k].get(h)
            if e is None:
                break
            cnt, tot = e
            p = max(cnt.get(w, 0) - DELTA, 0) / tot + DELTA * len(cnt) / tot * p
            maxk = k
        return p, maxk


def make_codes(variant, gamma, train_ids, rng):
    R = rng.integers(0, 2, (V, D), dtype=np.uint8)
    if variant == 'R':
        return R
    Q = rng.integers(0, 2, (4, D), dtype=np.uint8)
    score = np.zeros((V, D), np.float32)
    tot = np.full(V, 1e-9, np.float32)
    for oi, off in enumerate((-2, -1, 1, 2)):
        C = np.zeros(V * V, np.float64)
        for a in train_ids:
            a = a.astype(np.int64)
            x, y = (a[-off:], a[:off]) if off < 0 else (a[:-off], a[off:])
            C += np.bincount(x * V + y, minlength=V * V)
        C = C.reshape(V, V).astype(np.float32)
        tot += C.sum(axis=1)
        S = (2.0 * (R ^ Q[oi]) - 1.0).astype(np.float32)
        score += C @ S
        del C
    score /= tot[:, None]
    score += gamma * (2.0 * R - 1.0)
    return (score > 0).astype(np.uint8)


class Encoder:
    def __init__(self, Vec, rng):
        P = rng.integers(0, 2, (CTX, D), dtype=np.uint8)
        self.E = [Vec ^ P[j] for j in range(CTX)]

    def encode(self, a):
        n = len(a)
        outs = []
        for s in range(CTX, n, 20000):
            e = min(n, s + 20000)
            acc = np.zeros((e - s, D), np.uint8)
            for j in range(CTX):
                acc += self.E[j][a[s - 1 - j:e - 1 - j]]
            outs.append(np.packbits((acc * 2 > CTX).astype(np.uint8), axis=1).view(np.uint64))
        return np.concatenate(outs), a[CTX:]


def knn(M, nxt, q):
    d = np.bitwise_count(M ^ q).sum(axis=1, dtype=np.int32)
    idx = np.argpartition(d, TOPK)[:TOPK]
    dd = d[idx]
    dmin = int(dd.min())
    w = np.exp(-(dd - dmin) / TAU)
    acc = {}
    for j, ww in zip(nxt[idx], w):
        acc[int(j)] = acc.get(int(j), 0.0) + float(ww)
    s = w.sum()
    return {k: v / s for k, v in acc.items()}, dmin


def dbucket(d):
    for i, e in enumerate(DEDGES):
        if d <= e:
            return i
    return len(DEDGES)


def kbucket(k):
    return 0 if k <= 1 else (1 if k == 2 else 2)


def collect(mods, ng, enc, M, nxt, cache):
    pp, pk, db, kb = [], [], [], []
    for a in mods:
        Q, y = enc.encode(a)
        al = a.tolist()
        for pos in range(len(y)):
            i = pos + CTX
            ctx = tuple(al[i - CTX:i])
            p1, mk = ng.prob(ctx, al[i])
            r = cache.get(ctx)
            if r is None:
                r = knn(M, nxt, Q[pos]); cache[ctx] = r
            dist, dm = r
            pp.append(p1); pk.append(dist.get(al[i], 0.0)); db.append(dbucket(dm)); kb.append(kbucket(mk))
    return tuple(map(np.array, (pp, pk, db, kb)))


def mixp(pp, pk, lam):
    return (1 - lam) * pp + lam * (0.95 * pk + 0.05 / V)


def main():
    variant = sys.argv[1]; gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    t0 = time.time()
    tr, va, te = corpus()
    print('モジュール数 訓練%d 検証%d テスト%d / 全単語(訓練側)=%d' % (len(tr), len(va), len(te), sum(map(len, tr))), flush=True)
    tr = take(tr, N_TRAIN); va = take(va, N_VAL); te = take(te, N_TEST)
    from collections import Counter
    cnt = Counter(t for m in tr for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}   # 0 = <unk>
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr_i, va_i, te_i = conv(tr), conv(va), conv(te)
    unk = np.mean(np.concatenate(te_i) == 0)
    print('訓練%d語 検証%d語 テスト%d語 <unk>率(テスト)=%.1f%%' % (sum(map(len, tr_i)), sum(map(len, va_i)), sum(map(len, te_i)), 100 * unk), flush=True)
    rng = np.random.default_rng(1)
    ng = NGram()
    for a in tr_i:
        ng.train(a)
    Vec = make_codes(variant, gamma, tr_i, rng)
    enc = Encoder(Vec, rng)
    Ms, ys = zip(*[enc.encode(a) for a in tr_i])
    M = np.concatenate(Ms); nxt = np.concatenate(ys)
    print('変種=%s γ=%.2f 記憶=%d件 準備%.0fs' % (variant, gamma, len(M), time.time() - t0), flush=True)
    cache = {}
    val = collect(va_i, ng, enc, M, nxt, cache)
    print('検証終了 %.0fs' % (time.time() - t0), flush=True)
    table = {}
    for d in range(len(DEDGES) + 1):
        for k in range(3):
            m = (val[2] == d) & (val[3] == k)
            table[(d, k)] = 0.0 if m.sum() < 30 else max(LAMS, key=lambda l: np.log2(mixp(val[0][m], val[1][m], l)).sum())
    pp, pk, db, kb = collect(te_i, ng, enc, M, nxt, cache)
    lam = np.array([table[(d, k)] for d, k in zip(db, kb)])
    b_ng = -np.log2(pp); b_kn = -np.log2(0.95 * pk + 0.05 / V); b_mx = -np.log2(mixp(pp, pk, lam))
    print('テスト終了 %.0fs' % (time.time() - t0))
    for d in range(len(DEDGES) + 1):
        print('   ', [table[(d, k)] for k in range(3)])
    print('テスト n=%d | 1語あたりbit: 4-gram=%.3f kNN単独=%.3f 混合=%.3f (改善 %.2f%%)'
          % (len(pp), b_ng.mean(), b_kn.mean(), b_mx.mean(), 100 * (1 - b_mx.mean() / b_ng.mean())))
    for name, m in (('文脈が初見(最長一致<=1語)', kb == 0), ('一致2語', kb == 1), ('一致3語', kb == 2)):
        if m.sum():
            print('   %-22s 割合%4.0f%%  4-gram=%.3f kNN=%.3f 混合=%.3f' % (name, 100 * m.mean(), b_ng[m].mean(), b_kn[m].mean(), b_mx[m].mean()))


if __name__ == '__main__':
    main()
