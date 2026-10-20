"""
施策の試験(第3陣)。第1・2陣は items.py。ここでは残りを、手元のデータで測れる「代理の試験」で埋める。
代理であることは各関数の説明に書く(本物の試験には、会話・コード実行の環境が要るものがある)。
"""
import os, sys, time, math, re, copy, sysconfig
import numpy as np
import torch
import lab
from lab import *
import items as I
from items import hidden, knn_probs


# ---------------- 共通 ----------------
def split_AB(ctx):
    cut = np.median(np.unique(ctx.mtr))
    A = np.where(ctx.mtr <= cut)[0].astype(np.int64); B = np.where(ctx.mtr > cut)[0].astype(np.int64)
    return A, B


def bits_te(net, ctx):
    return bits(forward_all(net, ctx, 'te')['p'])


def knn_bits(net, ctx, Xk, yk, rv, rt, k=32):
    with torch.no_grad():
        keys = torch.cat([hidden(net, torch.from_numpy(Xk[i:min(i + 8192, len(Xk))])) for i in range(0, len(Xk), 8192)])
        vals = torch.from_numpy(yk)
        q = lambda X: torch.cat([hidden(net, torch.from_numpy(X[i:i + 8192])) for i in range(0, len(X), 8192)])
        pv = knn_probs(keys, vals, q(ctx.Xva), ctx.yva, k); pt = knn_probs(keys, vals, q(ctx.Xte), ctx.yte, k)
    best = None
    for T in pv:
        for lam in (0.5, 0.6, 0.7, 0.8, 0.9):
            bv = bits(lam * rv['p'] + (1 - lam) * pv[T])
            if best is None or bv < best[0]:
                best = (bv, T, lam)
    _, T, lam = best
    return bits(lam * rt['p'] + (1 - lam) * pt[T])


# ---------------- #4 局所の更新(勾配の一部だけで更新) ----------------
def i04_sparse_update(ctx):
    """#4 毎回の更新を、勾配の大きい上位10%の重みだけにしても学習できるか"""
    out = dict(base=bits_te(train(ctx, 256), ctx))
    for frac in (0.1, 0.02):
        torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
        net = lab.Net(256); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
        steps = len(X) // 512; st = 0
        for i in torch.randperm(len(X)).split(512):
            if len(i) < 512:
                break
            loss = torch.nn.functional.cross_entropy(net(X[i]), y[i]); opt.zero_grad(); loss.backward()
            for p in net.parameters():
                g = p.grad.abs().flatten()
                s = g[torch.randint(len(g), (min(len(g), 100000),))]
                thr = torch.quantile(s, 1 - frac)
                p.grad *= (p.grad.abs() >= thr)
            opt.step(); st += 1
        out['top%d%%' % int(frac * 100)] = bits_te(net, ctx)
    return out


# ---------------- #5 知識を足しても学習し直さない ----------------
def i05_add_knowledge(ctx):
    """#5 モデルは前半の文書だけで学習。後半の文書を、(a)索引に足す (b)追加学習する (c)最初から全部で学習する"""
    A, B = split_AB(ctx)
    net = train(ctx, 256, idx=A)
    rv, rt = forward_all(net, ctx, 'va'), forward_all(net, ctx, 'te')
    M = 500_000
    res = dict(A_only=bits(rt['p']))
    res['A_plus_index_A'] = knn_bits(net, ctx, ctx.Xtr[A][:M], ctx.ytr[A][:M], rv, rt)
    Xab = np.concatenate([ctx.Xtr[A][:M], ctx.Xtr[B][:M]]); yab = np.concatenate([ctx.ytr[A][:M], ctx.ytr[B][:M]])
    res['A_plus_index_AB(再学習なし)'] = knn_bits(net, ctx, Xab, yab, rv, rt)
    ft = lab.Net(256); ft.load_state_dict({k: v.clone() for k, v in net.state_dict().items()})
    ft = train(ctx, 256, idx=B, net=ft, lr=1e-3)
    res['A_then_finetune_B'] = bits_te(ft, ctx)
    res['joint_AB(全部で学習)'] = bits_te(train(ctx, 256), ctx)
    return res


# ---------------- #8 棚(学習結果)の合体 ----------------
def i08_merge(ctx):
    """#8 別々に学習した2つのモデルを、重みの平均で合体できるか(共通の出発点がある場合/ない場合)"""
    A, B = split_AB(ctx)
    def avg(a, b):
        n = lab.Net(256); n.load_state_dict({k: (a.state_dict()[k] + b.state_dict()[k]) / 2 for k in a.state_dict()}); return n
    a = train(ctx, 256, idx=A, seed=0); b = train(ctx, 256, idx=B, seed=0)
    b2 = train(ctx, 256, idx=B, seed=1)
    res = dict(A=bits_te(a, ctx), B=bits_te(b, ctx), avg_same_init_seed=bits_te(avg(a, b), ctx), avg_different_init=bits_te(avg(a, b2), ctx))
    warm = train(ctx, 256, ep=0.2)
    wa = train(ctx, 256, idx=A, net=copy.deepcopy(warm), lr=1e-3); wb = train(ctx, 256, idx=B, net=copy.deepcopy(warm), lr=1e-3)
    res['avg_after_shared_warmup'] = bits_te(avg(wa, wb), ctx)
    res['joint'] = bits_te(train(ctx, 256), ctx)
    return res


# ---------------- #12,#13,#27 規則(SCAN) ----------------
def _scan(name):
    import scan_lift as SL
    return SL, SL.load('tasks_train_%s.txt' % name), SL.load('tasks_test_%s.txt' % name)


def i12_rule_scope(ctx=None):
    """#12 規則が「どこまで通用するか」を自分で分かるか: 規則の裏づけ数と、使った時の正答率"""
    SL, tr, te = _scan('simple')
    S = SL.learn(tr, log=lambda *a: None); SL.tune_ranks(S, tr, log=lambda *a: None)
    rows = []
    for r in S.rules:
        used = [(a, b) for a, b in te[:3000] if any(it[0] == 'c' and it[1] in r.consts for it in r.pat) and r.consts and r.consts[0] in a]
        if used:
            rows.append(dict(rule=r.show(), n=len(used), acc=float(np.mean([S.run(a) == b for a, b in used]))))
    return dict(test_acc=SL.accuracy(S, te), rules=len(S.rules), rows=rows[:12])


def i13_specific_first(ctx=None):
    """#13 例外(具体的)を規則(一般的)より先に使う優先順位は効くか"""
    SL, tr, te = _scan('length')
    S = SL.learn(tr, log=lambda *a: None)
    res = {}
    for name, key in (('学習順(既定)', None), ('具体的な規則を先', lambda r, i: (len(r.consts), -i)), ('一般的な規則を先', lambda r, i: (-len(r.consts), -i))):
        S2 = copy.copy(S)
        if key:
            S2.ranks = None
            S2.rank = (lambda k: (lambda r, i: (0, *k(r, i))))(key)
        res[name] = SL.accuracy(S2, te[:4000])
    return res


def i27_noise(ctx=None):
    """#27 訓練データに間違い(ラベルのノイズ)が混じっても、実行での確認と採用基準で規則を守れるか"""
    SL, tr, te = _scan('simple')
    rng = np.random.default_rng(0)
    res = {}
    for p in (0.0, 0.05, 0.1, 0.2):
        tr2 = []
        for a, b in tr:
            if rng.random() < p and len(b) > 1:
                b = list(b); j = rng.integers(len(b)); b[j] = b[(j + 1) % len(b)] if rng.random() < .5 else 'I_WALK'; b = tuple(b)
            tr2.append((a, b))
        S = SL.learn(tr2, log=lambda *a: None); SL.tune_ranks(S, tr2, log=lambda *a: None)
        res['noise%d%%' % int(p * 100)] = dict(test_acc=SL.accuracy(S, te[:4000]), rules=len(S.rules))
    return res


# ---------------- #16 価値で刈る vs 回数で刈る ----------------
def i16_value_prune(ctx):
    """#16 同じ記憶量で、「回数が少ないものから」捨てるのと「節約ビットが小さいものから」捨てるのを比べる"""
    import exp_scale as es, exp_budget as EB, exp_ladder as L
    w3 = lambda X: X[:, HIST - 3:]
    A = es.NG(w3(ctx.Xtr), ctx.ytr)
    pu = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1.0; pu /= pu.sum()
    def bits_of(m):
        pv = m.probs(w3(ctx.Xva), ctx.yva); pt = m.probs(w3(ctx.Xte), ctx.yte)
        pv = pv[0] if isinstance(pv, tuple) else pv; pt = pt[0] if isinstance(pt, tuple) else pt
        w = L.em(np.array([pv, pu[ctx.yva]])); return bits(w[0] * pt + w[1] * pu[ctx.yte])
    def prune_by(model, score_fn, keep_n):
        sc = [score_fn(t) for t in model.t]; allsc = np.concatenate(sc)
        thr = np.sort(allsc)[::-1][min(keep_n, len(allsc)) - 1]
        new = []
        for t, s in zip(model.t, sc):
            uc, tot, typ, up, pc = t
            keep = s >= thr
            if not keep.any(): keep[np.argmax(s)] = True
            up2, pc2 = up[keep], pc[keep]; uc2, inv = np.unique(up2 // V, return_inverse=True)
            new.append((uc2, np.bincount(inv, weights=pc2).astype(np.float64), np.bincount(inv).astype(np.float64), up2, pc2))
        m = copy.copy(model); m.t = new; return m
    def value(t):
        uc, tot, typ, up, pc = t
        ctxs = up // V; y = up % V
        i = np.searchsorted(uc, ctxs); tc = tot[np.minimum(i, len(tot) - 1)]
        return pc * np.log2(np.maximum(pc / np.maximum(tc, 1), 1e-9) / pu[y])
    full = EB.entries(A); res = dict(full_entries=full, full_bits=bits_of(A), rows=[])
    for frac in (0.5, 0.2, 0.1, 0.05):
        n = int(full * frac)
        res['rows'].append(dict(keep=frac, by_count=bits_of(prune_by(A, lambda t: t[4].astype(np.float64) + np.random.default_rng(0).random(len(t[4])) * 0.01, n)),
                                by_value=bits_of(prune_by(A, value, n))))
    return res


# ---------------- #23 長さに対する計算量 ----------------
def i23_length_cost(ctx=None):
    """#23 文脈が長くなった時の計算時間: 平均を使う軽い方式 と Transformer"""
    torch.set_num_threads(os.cpu_count() or 2)
    class TFL(I.TinyTF):
        def __init__(s, L):
            super().__init__(); s.pos = torch.nn.Parameter(torch.zeros(L, s.d))
    rows = []
    for L in (12, 48, 192, 768):
        X = torch.randint(0, V, (128, L)); y = torch.randint(0, V, (128,))
        mlp = lab.Net(256); tf = TFL(L)
        r = dict(length=L)
        for name, m in (('mlp_mean', mlp), ('transformer', tf)):
            t0 = time.time()
            for _ in range(3):
                loss = torch.nn.functional.cross_entropy(m(X), y); loss.backward()
            r[name + '_ms_per_sample'] = (time.time() - t0) / 3 / 128 * 1000
        rows.append(r)
    return dict(rows=rows)


# ---------------- #26 #36 #37 計算の厳密さと道具 ----------------
def _arith(n, rng, ops=2):
    a = rng.integers(0, 1000, (n, ops + 0)); return a


def i26_exact_math(ctx=None):
    """#26/#36/#37 足し算を、NNに学習させた場合と、道具(電卓)を呼ぶ場合の正答率。2項と3項の連鎖"""
    torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
    rng = np.random.default_rng(0)
    def enc(a, b):  # 3桁+3桁 → 6桁のワンホット
        d = np.stack([a // 100, a // 10 % 10, a % 10, b // 100, b // 10 % 10, b % 10], 1)
        return torch.from_numpy(np.eye(10, dtype=np.float32)[d].reshape(len(a), -1))
    def target(s):  # 4桁
        return torch.from_numpy(np.stack([s // 1000, s // 100 % 10, s // 10 % 10, s % 10], 1))
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.f = torch.nn.Sequential(torch.nn.Linear(60, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 40))
        def forward(s, x): return s.f(x).view(len(x), 4, 10)
    m = M(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for step in range(6000):
        a = rng.integers(0, 1000, 256); b = rng.integers(0, 1000, 256)
        loss = torch.nn.functional.cross_entropy(m(enc(a, b)).reshape(-1, 10), target(a + b).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    a = rng.integers(0, 1000, 5000); b = rng.integers(0, 1000, 5000)
    with torch.no_grad():
        pred = m(enc(a, b)).argmax(2).numpy()
    pv = pred[:, 0] * 1000 + pred[:, 1] * 100 + pred[:, 2] * 10 + pred[:, 3]
    acc2 = float((pv == a + b).mean())
    # 連鎖: (a+b)+c を、学習したNNを2回使って
    c = rng.integers(0, 1000, 5000); ok = pv.copy(); ok = np.clip(ok, 0, 999)  # NNは3桁入力しか受けない
    with torch.no_grad():
        p2 = m(enc(ok, c)).argmax(2).numpy()
    pv2 = p2[:, 0] * 1000 + p2[:, 1] * 100 + p2[:, 2] * 10 + p2[:, 3]
    small = (a + b) < 1000
    acc3 = float((pv2 == (a + b) + c)[small].mean())
    return dict(nn_two_operand_acc=acc2, nn_chained_three_operand_acc=acc3, tool_acc=1.0,
                note='道具(電卓)は実行するので正解率は構成上100%。NN側の正答率は、桁数・学習量を増やして外れる所を探す対象')


# ---------------- #30 弱点の発見 ----------------
def i30_weakness(ctx):
    """#30 自分の損失から弱点(頻度帯)を見つけ、そこを増やして学習すると全体が良くなるか"""
    net, r, b0, _ = baseline(ctx, 256)
    freq = np.bincount(ctx.ytr, minlength=V); rank = np.argsort(np.argsort(-freq))
    dec = np.minimum(rank[ctx.yva] * 10 // V, 9)
    rv = forward_all(net, ctx, 'va')
    per = [float(rv['loss'][dec == d].mean()) if (dec == d).any() else float('nan') for d in range(10)]
    tr_dec = np.minimum(rank[ctx.ytr] * 10 // V, 9)
    have = [d for d in range(10) if (tr_dec == d).sum() > 100 and not np.isnan(per[d])]
    worst = max(have, key=lambda d: per[d])
    extra = np.where(tr_dec == worst)[0].astype(np.int64)
    rng = np.random.default_rng(0)
    n_extra = int(0.2 * len(ctx.Xtr))
    ex = rng.choice(extra, n_extra); rd = rng.choice(len(ctx.Xtr), n_extra)
    out = dict(base=b0, loss_by_freq_decile=per, worst_decile=worst)
    for name, add in (('targeted', ex), ('random', rd)):
        idx = np.concatenate([np.arange(len(ctx.Xtr)), add]).astype(np.int64)
        out[name] = bits_te(train(ctx, 256, ep=1.0, idx=idx), ctx)
    return out


# ---------------- #33 中断と再開 ----------------
def i33_resume(ctx):
    """#33 途中で止めて(モデル・最適化の状態・順番を)保存し、読み込み直して続けても、止めずに学習した場合と同じ結果になるか"""
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
    steps = len(X) // 512
    def run(stop=None, state=None):
        torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
        net = lab.Net(256); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        g = torch.Generator().manual_seed(1); perm = torch.randperm(len(X), generator=g)
        st = 0
        if state is not None:
            net.load_state_dict(state['m']); opt.load_state_dict(state['o']); st = state['st']
        while st < (stop or steps):
            i = perm[(st * 512) % (len(perm) - 512):][:512]
            for gr in opt.param_groups: gr['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
            loss = torch.nn.functional.cross_entropy(net(X[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step(); st += 1
        return net, opt, st
    full, _, _ = run()
    half, opt, st = run(stop=steps // 2)
    torch.save(dict(m=half.state_dict(), o=opt.state_dict(), st=st), '/tmp/ck.pt')
    n2, _, _ = run(state=torch.load('/tmp/ck.pt'))
    return dict(uninterrupted=bits_te(full, ctx), saved_and_resumed=bits_te(n2, ctx))


# ---------------- #34 同じ部品で複数の種類(文章+コード) ----------------
def _code_docs(limit_tokens):
    root = sysconfig.get_paths()['stdlib']; TOK = re.compile(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]")
    docs, tot = [], 0
    for dp, dn, fn in os.walk(root):
        if 'site-packages' in dp or 'test' in dp:
            continue
        dn.sort()
        for f in sorted(fn):
            if f.endswith('.py'):
                try: t = TOK.findall(open(os.path.join(dp, f), encoding='utf-8').read())
                except Exception: continue
                if len(t) > 200:
                    docs.append(t); tot += len(t)
        if tot > limit_tokens:
            break
    return docs


def i34_one_mechanism(ctx):
    """#34 文章とコードを1つのモデルで学習して、それぞれ別々に学習したモデルと比べる"""
    from collections import Counter
    import exp_far as XF
    code = _code_docs(3_000_000)
    k = int(len(code) * 0.8)
    wiki_tr = [d for d in ctx.tr_i if len(d) > 60]
    cnt = Counter()
    for d in code[:k]: cnt.update(d)
    cvocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V // 2 - 1))}
    # 結合語彙: 文章側(既存ID 0..V/2-1 に再割り当て)+ コード側(V/2..)
    wiki_ids = lambda a: np.where(a < V // 2, a, 0)
    cod = lambda d: np.array([cvocab.get(t, 0) + V // 2 for t in d], np.int64)
    mk = lambda docs, lim: XF.lwindows_meta(docs, lim)
    n = min(ctx.N, 2_000_000)
    Xw, yw, _, _ = mk([wiki_ids(d) for d in ctx.tr_i], n); Xwt, ywt, _, _ = mk([wiki_ids(d) for d in ctx.te_i], 40000)
    Xc, yc, _, _ = mk([cod(d) for d in code[:k]], n); Xct, yct, _, _ = mk([cod(d) for d in code[k:]], 40000)
    c2 = lab.Ctx(); c2.N = n
    def fit(X, y):
        c2.Xtr, c2.ytr, c2.mtr = X, y, np.zeros(len(y), np.int32)
        return train(c2, 256)
    def ev(net, X, y):
        c3 = lab.Ctx(); c3.Xte, c3.yte = X, y; c3.Xtr = c3.Xva = X; c3.ytr = c3.yva = y; return bits(forward_all(net, c3, 'te')['p'])
    mix = fit(np.concatenate([Xw, Xc]), np.concatenate([yw, yc]))
    w_only, c_only = fit(Xw, yw), fit(Xc, yc)
    return dict(wiki=dict(wiki_only=ev(w_only, Xwt, ywt), mixed=ev(mix, Xwt, ywt)),
                code=dict(code_only=ev(c_only, Xct, yct), mixed=ev(mix, Xct, yct)))


# ---------------- #35 質問から答え(文脈からの引き出し) ----------------
def i35_qa(ctx=None):
    """#35 「事実を述べた後、質問されたら答える」型を学べるか(合成データ: 人→町)。窓つきNN と 文脈の索引(コピー)"""
    torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
    rng = np.random.default_rng(0)
    NP, NC = 300, 50  # 人の数、町の数
    # 語彙: 0..NP-1 人, NP..NP+NC-1 町, 特別語: lives, where, ?, .,
    LIVES, WHERE, Q, DOT = NP + NC, NP + NC + 1, NP + NC + 2, NP + NC + 3
    def make(n, k=3):
        X, y = [], []
        for _ in range(n):
            ps = rng.choice(NP, k, replace=False); cs = rng.integers(0, NC, k)
            seq = []
            for p, c in zip(ps, cs): seq += [p, LIVES, NP + c, DOT]
            j = rng.integers(k); seq += [WHERE, ps[j], Q]
            X.append(seq); y.append(NP + cs[j])
        return np.array(X), np.array(y)
    Xtr, ytr = make(60000); Xte, yte = make(5000)
    VV = NP + NC + 4
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.E = torch.nn.Embedding(VV, 64)
            L = Xtr.shape[1]; s.f = torch.nn.Sequential(torch.nn.Linear(L * 64, 512), torch.nn.ReLU(), torch.nn.Linear(512, VV))
        def forward(s, x): return s.f(s.E(x).reshape(len(x), -1))
    m = M(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    for ep in range(8):
        for i in torch.randperm(len(Xt)).split(256):
            loss = torch.nn.functional.cross_entropy(m(Xt[i]), yt[i]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc_nn = float((m(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean())
    # 文脈の索引(コピー規則): 質問の人の直後の「lives 町」を引く
    ok = 0
    for x, t in zip(Xte, yte):
        p = x[-2]; pos = np.where(x[:-3] == p)[0]
        ok += (len(pos) > 0 and x[pos[0] + 2] == t)
    return dict(window_nn_acc=acc_nn, context_index_acc=float(ok / len(yte)), chance=1 / NC)


# ---------------- #39 物差し(スコアカード) ----------------
def i39_scorecard(ctx):
    """#39 基準モデルを、複数の物差し(文章・計算・文脈からの答え・不確かさ)で一度に測る"""
    net, r, b0, _ = baseline(ctx, 256)
    o = {'wiki_bits': b0, 'top1_acc': float(r['ok'].mean()), 'auc_confidence': I.auc(r['maxp'], r['ok'])}
    o['arith'] = i26_exact_math()['nn_two_operand_acc']
    o['qa_window_nn'] = i35_qa()['window_nn_acc']
    return o


ITEMS2 = {'04': i04_sparse_update, '05': i05_add_knowledge, '08': i08_merge, '12': i12_rule_scope, '13': i13_specific_first,
          '16': i16_value_prune, '23': i23_length_cost, '26': i26_exact_math, '27': i27_noise, '30': i30_weakness,
          '33': i33_resume, '34': i34_one_mechanism, '35': i35_qa, '39': i39_scorecard}
