"""
実験5: データを増やすと、小さなニューラル(勾配学習)は 4-gram(数え上げ)との差を広げるか。
コーパス: 標準ライブラリ+インストール済みライブラリのdocstring(英語の技術文書)約400万語。
  モジュール単位で 訓練/検証/テスト に分け、検証・テストと同じ文章が訓練側にあれば訓練側から除く(重複リーク対策)。
  語彙は全体共通(訓練全体の上位5000語)なので、テストの単語列はどのNでも同一。
使い方: python3 exp_scale.py <訓練語数N> <エポック数>
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import ast, hashlib, re, sys, time, warnings, copy, pickle
import numpy as np
from collections import Counter
import sysconfig
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/hdc')
import exp_nn2 as nn2

V = 5000; CTX = 3; DELTA = 0.75
TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
N_VAL, N_TEST = 15_000, 40_000
CACHE = '/home/claude/hdc/corpus_cache.pkl'


def build_corpus():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, 'rb'))
    roots = [sysconfig.get_paths()['stdlib'], sysconfig.get_paths()['purelib']]
    files = []
    for r in roots:
        for dp, dn, fn in os.walk(r):
            dn.sort()
            for f in sorted(fn):
                if f.endswith('.py'):
                    files.append(os.path.join(dp, f))
    split = {0: [], 1: [], 2: []}   # 0=テスト 1=検証 2=訓練
    for p in files:
        try:
            tree = ast.parse(open(p, encoding='utf-8').read())
        except Exception:
            continue
        ds = [ast.get_docstring(n) for n in ast.walk(tree)
              if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        ds = [d.lower() for d in ds if d and len(d) > 40]
        if not ds:
            continue
        h = int(hashlib.md5(p.encode()).hexdigest(), 16)
        split[0 if h % 10 == 0 else 1 if h % 10 == 1 else 2].append((h, ds))
    held = set()
    for s in (0, 1):
        for _, ds in split[s]:
            held.update(hashlib.md5(d.encode()).hexdigest() for d in ds)
    seen = set()
    out = {}
    for s in (0, 1, 2):
        mods = []
        for h, ds in sorted(split[s]):
            keep = []
            for d in ds:
                k = hashlib.md5(d.encode()).hexdigest()
                if s == 2 and k in held:
                    continue
                if k in seen:        # 同じ文章の重複は1回だけ
                    continue
                seen.add(k); keep.append(d)
            toks = TOK.findall('\n'.join(keep))
            if len(toks) >= 200:
                mods.append(toks)
        out[s] = mods
    pickle.dump(out, open(CACHE, 'wb'))
    return out


def windows(mods, limit=None):
    X, y, tot = [], [], 0
    for a in mods:
        n = len(a)
        if n <= CTX:
            continue
        X.append(np.stack([a[j:n - CTX + j] for j in range(CTX)], axis=1)); y.append(a[CTX:])
        tot += n - CTX
        if limit and tot >= limit:
            break
    X = np.concatenate(X); y = np.concatenate(y)
    return (X[:limit], y[:limit]) if limit else (X, y)


def ctxcode(X, k):
    c = np.zeros(len(X), np.int64)
    for j in range(k):
        c += X[:, CTX - 1 - j].astype(np.int64) * (V ** j)
    return c


class NG:
    def __init__(s, X, y):
        s.t = []
        for k in range(CTX + 1):
            ck = ctxcode(X, k)
            up, pc = np.unique(ck * V + y, return_counts=True)
            uc, tot = np.unique(ck, return_counts=True)
            _, typ = np.unique(up // V, return_counts=True)
            s.t.append((uc, tot, typ, up, pc))

    def probs(s, X, y):
        p = np.full(len(y), 1.0 / V); alive = np.ones(len(y), bool); mk = np.zeros(len(y), int)
        for k in range(CTX + 1):
            uc, tot, typ, up, pc = s.t[k]
            cq = ctxcode(X, k)
            i = np.minimum(np.searchsorted(uc, cq), len(uc) - 1)
            found = alive & (uc[i] == cq)
            pq = cq * V + y
            j = np.minimum(np.searchsorted(up, pq), len(up) - 1)
            c = np.where(up[j] == pq, pc[j], 0)
            newp = np.maximum(c - DELTA, 0) / tot[i] + DELTA * typ[i] / tot[i] * p
            p = np.where(found, newp, p); mk = np.where(found, k, mk); alive = found
        return p, mk


def main():
    N = int(sys.argv[1]); epochs = float(sys.argv[2])
    t0 = time.time()
    corp = build_corpus()
    print('モジュール数 テスト%d 検証%d 訓練%d / 訓練側の総語数=%d (%.0fs)' % (
        len(corp[0]), len(corp[1]), len(corp[2]), sum(map(len, corp[2])), time.time() - t0), flush=True)
    cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr_i = conv(corp[2]); va_i = conv(corp[1]); te_i = conv(corp[0])
    Xva, yva = windows(va_i, N_VAL); Xte, yte = windows(te_i, N_TEST)
    Xtr, ytr = windows(tr_i, N)
    print('訓練%d 検証%d テスト%d 語 / テストの<unk>率=%.1f%%' % (len(Xtr), len(Xva), len(Xte), 100 * np.mean(yte == 0)), flush=True)
    ng = NG(Xtr, ytr)
    pva_ng, _ = ng.probs(Xva, yva); pte_ng, mk_te = ng.probs(Xte, yte)
    print('4-gram bit/語: 検証%.3f テスト%.3f (準備%.0fs)' % (-np.log2(pva_ng).mean(), -np.log2(pte_ng).mean(), time.time() - t0), flush=True)
    rng = np.random.default_rng(0)
    net = nn2.Net(rng, np.zeros(V)); net.qprob = np.ones(V) / V
    B = nn2.B
    spe = len(Xtr) // B
    total = int(epochs * spe)
    eval_every = max(1, spe // 4)
    best = (9e9, None, 0, 0, 0)
    step = 0
    while step < total:
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm) - B + 1, B):
            ix = perm[i:i + B]
            lr = 2e-3 if step < 0.6 * total else 1e-3
            net.step(Xtr[ix], ytr[ix], rng, False, lr)
            step += 1
            if step % eval_every == 0 or step == total:
                pv = net.probs_true(Xva, yva)
                lam = max([0, .05, .1, .2, .3, .5, .7, .9], key=lambda l: np.log2((1 - l) * pva_ng + l * pv).sum())
                bm = -np.log2((1 - lam) * pva_ng + lam * pv).mean()
                print('  step %d (%.2f epoch) 計算量=%.2f TMAC 検証: NN単独=%.3f 混合=%.3f(λ=%.2f) [%.0fs]' % (
                    step, step / spe, net.macs / 1e12, -np.log2(pv + 1e-12).mean(), bm, lam, time.time() - t0), flush=True)
                if bm < best[0]:
                    best = (bm, copy.deepcopy(net.p), lam, net.macs, step / spe)
            if step >= total:
                break
    net.p = best[1]; lam = best[2]
    pt = net.probs_true(Xte, yte)
    b_ng = -np.log2(pte_ng); b_nn = -np.log2(pt + 1e-12); b_mx = -np.log2((1 - lam) * pte_ng + lam * pt)
    print('\n[N=%d] テスト bit/語: 4-gram=%.3f NN単独=%.3f 混合=%.3f (混合の改善 %.1f%%, NN単独の改善 %.1f%%) | 選んだ時点: %.2f epoch, %.2f TMAC, λ=%.2f'
          % (len(Xtr), b_ng.mean(), b_nn.mean(), b_mx.mean(), 100 * (1 - b_mx.mean() / b_ng.mean()),
             100 * (1 - b_nn.mean() / b_ng.mean()), best[4], best[3] / 1e12, lam), flush=True)
    for k in range(CTX + 1):
        m = mk_te == k
        if m.sum():
            print('   4-gram最長一致=%d語: 割合%4.0f%%  4-gram=%.3f NN=%.3f 混合=%.3f' % (k, 100 * m.mean(), b_ng[m].mean(), b_nn[m].mean(), b_mx[m].mean()))


if __name__ == '__main__':
    main()
