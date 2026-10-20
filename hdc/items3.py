"""
第4陣: 「0.1Bで商用LLM並み」に近づくための、突破の問題ごとの試験。
 問題1 知識と技能の分離      #61 知識の容量(モデルに覚えさせる vs 索引に置く)
 問題2 正しい資料を引く       #50 学習データ全体への最長一致
 問題3 学習の効率            #53 順序  #56 出力の初期値  #62 先の語もまとめて予測
 問題4 考える時間と検証       #60 検証つき探索
 問題5 密な表現              #52 入出力の重み共有  #51 同じ重みの繰り返し(考える回数)
 問題6 物差し               #64 文法の最小対(BLiMP)
 問題7 検証できる課題での自己改善  #60(後半)
 問題9 柔らかい規則           #63 つづりの揺れに強い規則
"""
import os, time, math, random, itertools
import numpy as np
import torch
import lab
from lab import *
import items as I


def bits_net(net, ctx):
    return bits(forward_all(net, ctx, 'te')['p'])


# ================= #50 学習データ全体への最長一致 =================
def _hash(X, k):
    h = np.zeros(len(X), np.int64)
    for j in range(k):
        h = h * 1000003 + X[:, X.shape[1] - k + j] + 1
    return h


def i50_global_match(ctx):
    net, rt, b0, _ = baseline(ctx, 256)
    rv = forward_all(net, ctx, 'va')
    ks = (12, 8, 6, 4)
    tables = []
    for k in ks:
        key = _hash(ctx.Xtr, k); uk, inv, ck = np.unique(key, return_inverse=True, return_counts=True)
        pairs, pc = np.unique(inv.astype(np.int64) * V + ctx.ytr, return_counts=True)
        tables.append((uk, ck, pairs, pc))
    def prep(X, y):
        n = len(y); best = np.full(n, -1); pm = np.zeros(n)
        for i, (k, (uk, ck, pairs, pc)) in enumerate(zip(ks, tables)):
            key = _hash(X, k); pos = np.minimum(np.searchsorted(uk, key), len(uk) - 1)
            present = uk[pos] == key; tot = ck[pos]
            pair = pos.astype(np.int64) * V + y; pp = np.minimum(np.searchsorted(pairs, pair), len(pairs) - 1)
            cnt = np.where(pairs[pp] == pair, pc[pp], 0)
            sel = present & (best < 0)
            best[sel] = i; pm[sel] = (cnt / (tot + 0.5))[sel]
        return best, pm
    bv, pmv = prep(ctx.Xva, ctx.yva); bt, pmt = prep(ctx.Xte, ctx.yte)
    p = rt['p'].copy(); share = {}
    for b in range(-1, len(ks)):
        mv, mt = bv == b, bt == b
        share['none' if b < 0 else 'k=%d' % ks[b]] = float(mt.mean())
        if b < 0 or mv.sum() < 50:
            continue
        best = None
        for lam in np.linspace(0.05, 0.95, 19):
            bb = bits(lam * rv['p'][mv] + (1 - lam) * pmv[mv])
            if best is None or bb < best[0]:
                best = (bb, lam)
        p[mt] = best[1] * rt['p'][mt] + (1 - best[1]) * pmt[mt]
    return dict(base=b0, with_global_match=bits(p), coverage=share)


# ================= #51 同じ重みを繰り返す =================
class LoopNet(lab.Net):
    def __init__(s, H=256, R=0, shared=True):
        super().__init__(H)
        s.R, s.shared = R, shared
        s.blocks = torch.nn.ModuleList([torch.nn.Linear(H, H) for _ in range(1 if shared else R)]) if R else torch.nn.ModuleList()

    def forward(s, X, Hd=None):
        e = s.E(X)
        h = torch.relu(s.l1(torch.cat([e[:, -NEAR:].reshape(len(X), -1), e[:, :-NEAR].mean(1)], 1)))
        for r in range(s.R):
            h = h + torch.relu((s.blocks[0] if s.shared else s.blocks[r])(h))
        return s.l2(h)

    def macs(s):
        return super().macs() + s.R * s.H * s.H


def i51_loop(ctx):
    rows = []
    for R, sh in ((0, True), (1, True), (2, True), (4, True), (2, False), (4, False)):
        torch.manual_seed(0)
        net = train(ctx, 256, net=LoopNet(256, R, sh))
        rows.append(dict(R=R, shared=sh, params=sum(p.numel() for p in net.parameters()), macs=net.macs(), bits=bits_net(net, ctx)))
    return dict(rows=rows)


# ================= #52 入出力の重み共有 =================
class TiedNet(torch.nn.Module):
    def __init__(s, H):
        super().__init__()
        s.E = torch.nn.Embedding(V, DE); s.l1 = torch.nn.Linear((NEAR + 1) * DE, H); s.l2 = torch.nn.Linear(H, DE)
        s.b = torch.nn.Parameter(torch.zeros(V)); s.H = H

    def forward(s, X, Hd=None):
        e = s.E(X)
        h = torch.relu(s.l1(torch.cat([e[:, -NEAR:].reshape(len(X), -1), e[:, :-NEAR].mean(1)], 1)))
        return s.l2(h) @ s.E.weight.T + s.b

    def macs(s):
        return s.l1.in_features * s.H + s.H * DE + DE * V


def i52_tied(ctx):
    rows = []
    net = train(ctx, 256)
    rows.append(dict(name='通常 H=256', params=sum(p.numel() for p in net.parameters()), macs=net.macs(), bits=bits_net(net, ctx)))
    for H in (256, 1024, 2048):
        torch.manual_seed(0)
        net = train(ctx, H, net=TiedNet(H))
        rows.append(dict(name='共有 H=%d' % H, params=sum(p.numel() for p in net.parameters()), macs=net.macs(), bits=bits_net(net, ctx)))
    return dict(rows=rows)


# ================= #56 出力の初期値 =================
def i56_bias_init(ctx):
    freq = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1; lp = np.log(freq / freq.sum())
    rows = []
    for ep in (0.1, 0.25, 1.0):
        a = bits_net(train(ctx, 256, ep=ep), ctx)
        torch.manual_seed(0); n = lab.Net(256); n.l2.bias.data = torch.from_numpy(lp).float()
        b = bits_net(train(ctx, 256, ep=ep, net=n), ctx)
        rows.append(dict(epochs=ep, plain=a, unigram_bias_init=b))
    return dict(rows=rows)


# ================= #53 やさしい順・難しい順 =================
def i53_curriculum(ctx):
    freq = np.bincount(ctx.ytr, minlength=V); score = freq[ctx.ytr]
    order = np.argsort(-score, kind='stable')
    res = {}
    for name, idx in (('random', np.random.default_rng(0).permutation(len(order))), ('easy_first', order), ('hard_first', order[::-1])):
        torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
        net = lab.Net(256); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
        steps = len(X) // 512; st = 0; rng = np.random.default_rng(1)
        for blk in np.array_split(idx, 10):
            blk = rng.permutation(blk)
            for j in range(0, len(blk) - 511, 512):
                i = torch.from_numpy(blk[j:j + 512])
                for g in opt.param_groups: g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
                loss = torch.nn.functional.cross_entropy(net(X[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step(); st += 1
        res[name] = bits_net(net, ctx)
    return res


# ================= #62 先の語もまとめて予測 =================
def i62_multitoken(ctx):
    n = len(ctx.ytr)
    y2 = np.full(n, -100, np.int64); y3 = np.full(n, -100, np.int64)
    same = (ctx.mtr[1:] == ctx.mtr[:-1]) & (ctx.ptr[1:] == ctx.ptr[:-1] + 1)
    y2[:-1] = np.where(same, ctx.ytr[1:], -100)
    same2 = same[1:] & same[:-1]
    y3[:-2] = np.where(same2, ctx.ytr[2:], -100)
    res = {}
    for name, w in (('next_only', (0, 0)), ('plus_next2', (0.5, 0)), ('plus_next2_next3', (0.5, 0.25))):
        torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
        net = lab.Net(256); h2 = torch.nn.Linear(256, V); h3 = torch.nn.Linear(256, V)
        opt = torch.optim.Adam(list(net.parameters()) + list(h2.parameters()) + list(h3.parameters()), lr=2e-3)
        X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr); Y2 = torch.from_numpy(y2); Y3 = torch.from_numpy(y3)
        steps = len(X) // 512; st = 0
        for i in torch.randperm(len(X)).split(512):
            if len(i) < 512: break
            for g in opt.param_groups: g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
            e = net.E(X[i]); h = torch.relu(net.l1(torch.cat([e[:, -NEAR:].reshape(len(i), -1), e[:, :-NEAR].mean(1)], 1)))
            loss = torch.nn.functional.cross_entropy(net.l2(h), y[i])
            if w[0]: loss = loss + w[0] * torch.nn.functional.cross_entropy(h2(h), Y2[i], ignore_index=-100)
            if w[1]: loss = loss + w[1] * torch.nn.functional.cross_entropy(h3(h), Y3[i], ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
        res[name] = bits_net(net, ctx)
    return res


# ================= #60 検証つき探索と、自己改善 =================
MAXL = 5
OPS = ['reverse', 'sort', 'inc', 'dec', 'dbl', 'drop_first', 'drop_last', 'evens', 'cumsum', 'sort_desc']


def _apply(op, l):
    if op == 0: return l[::-1]
    if op == 1: return sorted(l)
    if op == 2: return [(x + 1) % 10 for x in l]
    if op == 3: return [(x - 1) % 10 for x in l]
    if op == 4: return [(2 * x) % 10 for x in l]
    if op == 5: return l[1:]
    if op == 6: return l[:-1]
    if op == 7: return [x for x in l if x % 2 == 0]
    if op == 8:
        out, s = [], 0
        for x in l: s = (s + x) % 10; out.append(s)
        return out
    return sorted(l, reverse=True)


def _run(prog, l):
    for op in prog:
        l = _apply(op, l)
    return l


def _make_tasks(n, rng, Ls=(2, 3, 4, 5), n_ex=4):
    tasks = []
    for _ in range(n):
        L = rng.choice(Ls); prog = [int(rng.integers(10)) for _ in range(L)]
        ins = [[int(x) for x in rng.integers(0, 10, rng.integers(4, 9))] for _ in range(2 * n_ex)]
        outs = [_run(prog, l) for l in ins]
        tasks.append((prog, ins, outs))
    return tasks


def _enc(ins, outs):
    # 4つの例(入力・出力)を長さ8、値11種(0-9と空き)のワンホットにして並べる
    v = np.zeros((4, 2, 8, 11), np.float32)
    for i in range(4):
        for j, l in enumerate((ins[i], outs[i])):
            for p in range(8):
                v[i, j, p, l[p] if p < len(l) else 10] = 1
    return v.reshape(-1)


class Proposer(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.f = torch.nn.Sequential(torch.nn.Linear(4 * 2 * 8 * 11, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, MAXL * 11))

    def forward(s, x):
        return s.f(x).view(len(x), MAXL, 11)


def _fit(model, tasks_prog, epochs, lr=2e-3):
    # tasks_prog: [(ins, outs, prog)]
    X = torch.from_numpy(np.stack([_enc(a, b) for a, b, _ in tasks_prog]))
    Y = torch.full((len(tasks_prog), MAXL), 10, dtype=torch.long)
    for i, (_, _, p) in enumerate(tasks_prog):
        for j, op in enumerate(p): Y[i, j] = op
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for i in torch.randperm(len(X)).split(128):
            loss = torch.nn.functional.cross_entropy(model(X[i]).reshape(-1, 11), Y[i].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()


def _sample(model, ins, outs, k, rng):
    with torch.no_grad():
        pr = torch.softmax(model(torch.from_numpy(_enc(ins, outs))[None])[0], 1).numpy().astype(np.float64)
    pr /= pr.sum(1, keepdims=True)
    draws = np.stack([rng.choice(11, size=k, p=pr[j]) for j in range(MAXL)], 1)
    progs = []
    for row in draws:
        p = []
        for c in row:
            if c == 10: break
            p.append(int(c))
        progs.append(p)
    return progs


def _solve(model, task, k, rng):
    """k個の案を出し、見せた4例を満たすものを探す。(見つかった案, それが隠した4例も満たすか)"""
    _, ins, outs = task
    for p in _sample(model, ins[:4], outs[:4], k, rng):
        if p and all(_run(p, ins[i]) == outs[i] for i in range(4)):
            return p, all(_run(p, ins[i]) == outs[i] for i in range(4, 8))
    return None, False


def i60_search_verify(ctx=None):
    torch.manual_seed(0); rng = np.random.default_rng(0); torch.set_num_threads(os.cpu_count() or 2)
    test = _make_tasks(300, rng)
    labeled = _make_tasks(20000, rng)
    unl = _make_tasks(20000, rng)
    def evalm(m):
        r = {}
        for k in (1, 16, 256, 1024):
            r['k=%d' % k] = float(np.mean([_solve(m, t, k, rng)[1] for t in test]))
        return r
    m = Proposer(); _fit(m, [(t[1][:4], t[2][:4], t[0]) for t in labeled], 30)
    res = {'labeled3000_start': evalm(m)}
    # ランダムに案を出すだけの基準
    rnd = Proposer()
    res['untrained(random proposals)'] = {'k=%d' % k: float(np.mean([_solve(rnd, t, k, rng)[1] for t in test])) for k in (1, 16, 256, 1024)}
    # 自己改善: 答え(プログラム)を見ずに、実行で確かめられた案だけで学び直す
    pool = [(t[1][:4], t[2][:4], t[0]) for t in labeled]
    for rnd_i in range(1, 5):
        found = []
        for t in unl:
            p, ok = _solve(m, t, 64, rng)
            if p is not None:
                found.append((t[1][:4], t[2][:4], p))
        m = Proposer(); _fit(m, pool + found, 30)
        res['self_improve_round%d(found %d/%d)' % (rnd_i, len(found), len(unl))] = evalm(m)
    return res


# ================= #61 知識の容量: モデルに覚えさせる vs 索引に置く =================
def i61_capacity(ctx=None):
    """鍵は固定の乱数ベクトル(学習しない・数えない)。MLPの重みだけが記憶の場所。何ビット覚えられるかを測る"""
    torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
    C = 64; bits_per_fact = math.log2(C)
    rows = []
    for H in (64, 256, 1024):
        for N in (1000, 4000, 16000, 64000):
            keys = torch.randn(N, 32); vals = torch.randint(0, C, (N,))
            m = torch.nn.Sequential(torch.nn.Linear(32, H), torch.nn.ReLU(), torch.nn.Linear(H, C))
            params = sum(p.numel() for p in m.parameters())
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            for ep in range(400 if N <= 16000 else 250):
                for i in torch.randperm(N).split(2048):
                    loss = torch.nn.functional.cross_entropy(m(keys[i]), vals[i]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                acc = float((m(keys).argmax(1) == vals).float().mean())
            rows.append(dict(H=H, facts=N, params=params, acc=acc, bits_per_param=N * bits_per_fact * acc / params))
    return dict(rows=rows, index_lookup_acc=1.0, note='索引に置いた事実は、モデルの大きさに関係なく正答率100%(1件ずつ引くだけ)')


# ================= #63 つづりの揺れに強い規則 =================
def _edit(a, b, cap=3):
    if abs(len(a) - len(b)) > cap: return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def i63_soft_rules(ctx=None):
    import scan_lift as SL
    tr = SL.load('tasks_train_simple.txt'); te = SL.load('tasks_test_simple.txt')[:3000]
    S = SL.learn(tr, log=lambda *a: None); SL.tune_ranks(S, tr, log=lambda *a: None)
    known = sorted(set(S.lex) | {c for r in S.rules for c in r.consts})
    rng = np.random.default_rng(0)
    def typo(w):
        if len(w) < 3: return w
        i = int(rng.integers(len(w)))
        t = rng.integers(3)
        if t == 0: return w[:i] + w[i + 1:]
        if t == 1: return w[:i] + chr(97 + int(rng.integers(26))) + w[i + 1:]
        j = min(i + 1, len(w) - 1); l = list(w); l[i], l[j] = l[j], l[i]; return ''.join(l)
    def snap(w):
        if w in known: return w
        d = sorted((_edit(w, k), k) for k in known)
        return d[0][1] if d[0][0] <= 2 and (len(d) < 2 or d[0][0] < d[1][0]) else w
    res = {}
    for p in (0.0, 0.05, 0.1, 0.2):
        a1 = a2 = 0
        for cmd, out in te:
            noisy = tuple(typo(w) if rng.random() < p else w for w in cmd)
            a1 += S.run(noisy) == out
            a2 += S.run(tuple(snap(w) for w in noisy)) == out
        res['typo%d%%' % int(p * 100)] = dict(exact_rules=a1 / len(te), snapped_to_nearest=a2 / len(te))
    return res


# ================= #64 文法の物差し(BLiMP) =================
def i64_blimp(ctx):
    try:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'], check=True)
        from datasets import load_dataset
        from datasets import get_dataset_config_names
        allc = get_dataset_config_names('nyu-mll/blimp')
        confs = allc[::5][:13]
        data = {c: load_dataset('nyu-mll/blimp', c, split='train') for c in confs}
    except Exception as e:
        return dict(unavailable=str(e)[:200])
    from collections import Counter
    import re
    TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
    # 語彙: ctx と同じ(上位V語)。ctx.tr_i は ID 列なので、語→IDの対応を取り直す
    import exp_scale as es
    corp = es.build_corpus(); cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    net, _, b0, _ = baseline(ctx, 256)
    def score(sent):
        ids = [vocab.get(t, 0) for t in TOK.findall(sent.lower())]
        X, y = [], []
        for i in range(len(ids)):
            w = [0] * HIST + ids[:i]; X.append(w[-HIST:]); y.append(ids[i])
        with torch.no_grad():
            lp = torch.log_softmax(net(torch.tensor(X)), 1)
            return float(lp[torch.arange(len(y)), torch.tensor(y)].sum())
    res = {}
    for c, d in data.items():
        n = min(len(d), 300); ok = 0
        for r in list(d)[:n]:
            ok += score(r['sentence_good']) > score(r['sentence_bad'])
        res[c] = ok / n
    res['mean'] = float(np.mean(list(res.values())))
    return res


ITEMS3 = {'50': i50_global_match, '51': i51_loop, '52': i52_tied, '53': i53_curriculum, '56': i56_bias_init,
          '60': i60_search_verify, '61': i61_capacity, '62': i62_multitoken, '63': i63_soft_rules, '64': i64_blimp}
