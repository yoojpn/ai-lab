"""
G4: 文法 (入れ子) の深さの一般化。 学習は深さ <= 3 だけ。 テストは 深さ 3 / 6 / 10 まで。
  言語 A: Dyck-2 (( ) [ ] と葉 x)。 閉じ括弧の種類は 対応する開き括弧で決まる。 END は 括弧が全部閉じたときだけ。
  言語 B: 算術式  expr := term (('+'|'*') term)*,  term := n | '(' expr ')'。
  特徴: off1..off8 (直前8語), stk1..3 (まだ閉じていない開き括弧の種類, 内側から), depth0 (括弧が全部閉じているか), depth (0..5に丸め)。
  選び方: 特徴の部分集合 (大きさ <=3) を総当たりし、 検証の対数損失 (学習と同じ深さ) で選ぶ。 (a) 損失が最小のもの (b) 最小から 1% 以内で最も小さいもの。
  評価: (1) 正しい文法で見たときの 次語の「合法な語に置いた確率」 (深さ別, 教師強制)  (2) モデルから語を引いて文を自動生成したときの、 完全に合法な文の割合。
  「スタック」 (stk, depth0) は 私が与えた原始操作。 ここで測るのは、 それがあれば数え上げの表で深さを超えて一般化するか。
使い方: python3 exp_g4.py '{"ntrain":4000,"ntest":600}'
"""
import sys, json, itertools
import numpy as np

BOS, END = 6, 5
V = 7


def gen_dyck(rng, dmax, bias=0.5, L=70):
    seq = []; st = []
    while len(seq) < L or st:
        d = len(st)
        if len(seq) >= L:
            seq.append(st.pop() + 1); continue
        r = rng.random()
        if d < dmax and r < bias:
            k = 0 if rng.random() < 0.5 else 2; seq.append(k); st.append(k)
        elif d > 0 and r < bias + 0.25:
            seq.append(st.pop() + 1)
        else:
            seq.append(4)
    return seq + [END]


def gen_arith(rng, dmax):
    def expr(d):
        out = term(d)
        for _ in range(int(rng.integers(0, 3))):
            out += [3 if rng.random() < 0.5 else 4] + term(d)
        return out
    def term(d):
        if d < dmax and rng.random() < (0.65 if d < dmax else 0):
            return [0] + expr(d + 1) + [1]
        return [2]
    return expr(0) + [END]


def legal_set(lang, st, last):
    """st: 開き括弧のスタック (Dyck は 0/2, 算術は 0), last: 直前の語"""
    if lang == 'dyck':
        s = {0, 2, 4}
        if st:
            s.add(st[-1] + 1)
        else:
            s.add(END)
        return s
    if last in (BOS, 0, 3, 4):
        return {0, 2}
    s = {3, 4}
    if st:
        s.add(1)
    else:
        s.add(END)
    return s


def step_stack(lang, st, t):
    if t in (0, 2) and (lang == 'dyck' or t == 0):
        st.append(t)
    elif t in (1, 3) and lang == 'dyck' and st:
        st.pop()
    elif lang == 'arith' and t == 1 and st:
        st.pop()


def feats(lang, seq):
    """各位置 (BOS の次から END まで) の特徴ベクトル。 位置 i の特徴は seq[:i] だけから作る"""
    st = []; hist = [BOS] * 8; rows = []
    for t in seq:
        o = hist[::-1][:8]; s = [x + 1 for x in st[::-1][:3]] + [0] * (3 - min(3, len(st)))
        rows.append(o + s + [int(len(st) == 0), min(len(st), 5)])
        hist.append(t); hist = hist[-8:]; step_stack(lang, st, t)
    return rows


NAMES = ['off%d' % k for k in range(1, 9)] + ['stk1', 'stk2', 'stk3', 'depth0', 'depth']


def make(lang, rng, n, dmax):
    X = []; Y = []; M = []
    for _ in range(n):
        seq = gen_dyck(rng, dmax) if lang == 'dyck' else gen_arith(rng, dmax)
        X += feats(lang, seq); Y += seq
        st = []; last = BOS
        for t in seq:
            m = np.zeros(V, bool); m[list(legal_set(lang, st, last))] = True; M.append(m)
            step_stack(lang, st, t); last = t
    return np.array(X, np.int64), np.array(Y, np.int64), np.array(M)


def table(X, Y, sub):
    key = np.zeros(len(X), np.int64)
    for f in sub:
        key = key * 16 + X[:, f]
    u, inv = np.unique(key, return_inverse=True)
    C = np.zeros((len(u), V)); np.add.at(C, (inv, Y), 1)
    return u, C


def probs(tab, X, sub, a=0.1):
    u, C = tab; key = np.zeros(len(X), np.int64)
    for f in sub:
        key = key * 16 + X[:, f]
    i = np.minimum(np.searchsorted(u, key), len(u) - 1); hit = u[i] == key
    P = np.where(hit[:, None], (C[i] + a) / (C[i].sum(1, keepdims=True) + a * V), 1.0 / V)
    return P, hit


def loss(tab, X, Y, sub):
    P, _ = probs(tab, X, sub); return float(-np.log(P[np.arange(len(Y)), Y]).mean())


def select(Xt, Yt, Xv, Yv, maxk=3):
    res = []
    for k in range(1, maxk + 1):
        for sub in itertools.combinations(range(len(NAMES)), k):
            tab = table(Xt, Yt, sub); res.append((loss(tab, Xv, Yv, sub), sub))
    res.sort(); best = res[0]
    occam = min([r for r in res if r[0] <= best[0] * 1.01 + 1e-9], key=lambda r: (len(r[1]), r[0]))
    return best, occam


def generate_valid(lang, tab, sub, rng, n=500, maxlen=300):
    ok = 0; depths = []
    for _ in range(n):
        st = []; hist = [BOS] * 8; valid = True; mx = 0
        for step in range(maxlen):
            o = hist[::-1][:8]; s = [x + 1 for x in st[::-1][:3]] + [0] * (3 - min(3, len(st)))
            row = np.array([o + s + [int(len(st) == 0), min(len(st), 5)]])
            P, _ = probs(tab, row, sub); t = int(rng.choice(V, p=P[0] / P[0].sum()))
            if t not in legal_set(lang, st, hist[-1]):
                valid = False; break
            if t == END:
                break
            step_stack(lang, st, t); hist.append(t); hist = hist[-8:]; mx = max(mx, len(st))
        else:
            valid = False
        ok += valid; depths.append(mx)
    return ok / n, float(np.mean(depths))


def main(cfg):
    rng = np.random.default_rng(0); out = {}
    for lang in ('dyck', 'arith'):
        Xt, Yt, _ = make(lang, rng, cfg['ntrain'], 3); Xv, Yv, _ = make(lang, rng, cfg['ntrain'] // 4, 3)
        best, occam = select(Xt, Yt, Xv, Yv)
        X2, Y2, _ = make(lang, rng, cfg['ntrain'], 2); X3, Y3, _ = make(lang, rng, cfg['ntrain'] // 4, 3)       # 深さ<=2 で表を作り、 深さ3 で検証 (外挿で選ぶ)
        ood, _ = select(X2, Y2, X3, Y3)
        base = min(((loss(table(Xt, Yt, s), Xv, Yv, s), s) for k in (2, 3) for s in itertools.combinations(range(8), k)))
        tests = {dm: make(lang, rng, cfg['ntest'], dm) for dm in cfg.get('depths', [3, 6, 10])}
        out[lang] = {}
        for nm, (l, sub) in (('損失最小(同じ深さで検証)', best), ('最小から1%以内で最小', occam), ('外挿で検証して選ぶ(深さ<=2で作り3で検証)', ood), ('直前8語のみ', base)):
            tab = table(Xt, Yt, sub); rec = dict(feat=[NAMES[f] for f in sub], val_loss=l, legal_mass={}, ctx_seen={})
            for dm, (Xe, Ye, Me) in tests.items():
                P, hit = probs(tab, Xe, sub); rec['legal_mass'][dm] = float((P * Me).sum(1).mean()); rec['ctx_seen'][dm] = float(hit.mean())
            rec['valid_gen'], rec['gen_depth'] = generate_valid(lang, tab, sub, rng, cfg.get('ngen', 400))
            out[lang][nm] = rec
            print(lang, nm, rec['feat'], 'val %.3f' % l, 'legal_mass', {k: round(v, 4) for k, v in rec['legal_mass'].items()}, 'ctx_seen', {k: round(v, 3) for k, v in rec['ctx_seen'].items()}, 'valid_gen %.3f depth %.1f' % (rec['valid_gen'], rec['gen_depth']), flush=True)
    print('RESULT_G4 ' + json.dumps(dict(cfg=cfg, res=out)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
