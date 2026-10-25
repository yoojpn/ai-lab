"""
K: 文脈内の対応づけ(キーと値)。「A→1, B→2, ... A→?」と聞かれて、文脈の中から答えを引けるか。
  LLMが「教えていないこと」をその場で覚えて使う力(文脈内学習)の最小版。毎回、対応はランダムに作り直す(学習では覚えられない)。
  比較: 数えるn-gram(直前1〜3語) / 文脈内のn-gram head(引くだけ・勾配なし) / MLP(勾配あり) / 小さなTransformer(勾配あり)
使い方: python3 exp_k.py
"""
import json, time, math
import numpy as np, torch
NK = NV = 50; L = 13   # トークン: キー 1..50, 値 51..100, 0=詰め物


def gen(n, m, rng):
    ks = np.array([rng.permutation(NK)[:m] for _ in range(n)]) + 1
    vs = rng.integers(0, NV, (n, m)) + 1 + NK
    q = rng.integers(0, m, n)
    seq = np.zeros((n, L), np.int64)
    for j in range(m):
        seq[:, 2 * j] = ks[:, j]; seq[:, 2 * j + 1] = vs[:, j]
    seq[np.arange(n), 2 * m] = ks[np.arange(n), q]
    X = np.roll(seq, L - (2 * m + 1), 1)        # 右詰め(最後がクエリ)
    return X, vs[np.arange(n), q]


def head(X, y):
    """引くだけ: 最後のトークンと同じトークンが前にあれば、その次を答える"""
    pred = np.zeros(len(X), np.int64)
    for i, x in enumerate(X):
        idx = np.nonzero(x[:-1] == x[-1])[0]
        pred[i] = x[idx[-1] + 1] if len(idx) else 0
    return float((pred == y).mean())


def ngram(Xtr, ytr, X, y, k):
    from collections import Counter, defaultdict
    t = defaultdict(Counter)
    for x, v in zip(Xtr, ytr): t[tuple(x[-k:])][v] += 1
    ok = 0
    for x, v in zip(X, y):
        c = t.get(tuple(x[-k:]))
        ok += (c is not None and c.most_common(1)[0][0] == v)
    return ok / len(X)


class MLP(torch.nn.Module):
    def __init__(s, H=256):
        super().__init__(); s.E = torch.nn.Embedding(NK + NV + 1, 32)
        s.f = torch.nn.Sequential(torch.nn.Linear(L * 32, H), torch.nn.ReLU(), torch.nn.Linear(H, H), torch.nn.ReLU(), torch.nn.Linear(H, NK + NV + 1))
    def forward(s, x): return s.f(s.E(x).reshape(len(x), -1))


class TF(torch.nn.Module):
    def __init__(s, d=64, nl=2):
        super().__init__(); s.E = torch.nn.Embedding(NK + NV + 1, d); s.P = torch.nn.Parameter(torch.zeros(L, d))
        s.enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True), nl)
        s.o = torch.nn.Linear(d, NK + NV + 1)
    def forward(s, x): return s.o(s.enc(s.E(x) + s.P)[:, -1])


def train(model, steps, rng, ms=(2, 3, 4, 5, 6)):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for st in range(steps):
        m = ms[st % len(ms)]
        X, y = gen(256, m, rng)
        for g in opt.param_groups: g['lr'] = 2e-3 * min(1, (st + 1) / 200) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.9 + 0.1)
        loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(X)), torch.from_numpy(y))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def acc(model, m, rng):
    X, y = gen(2000, m, rng)
    with torch.no_grad(): return float((model(torch.from_numpy(X)).argmax(1).numpy() == y).mean())


if __name__ == '__main__':
    t0 = time.time(); torch.set_num_threads(4); rng = np.random.default_rng(0)
    res = {'chance': 1 / NV, 'head': {}, 'ngram1': {}, 'ngram2': {}, 'ngram3': {}, 'mlp': {}, 'tf': {}}
    for m in (2, 3, 4, 5, 6):
        X, y = gen(2000, m, rng); Xt, yt = gen(20000, m, rng)
        res['head'][m] = head(X, y)
        for k in (1, 2, 3): res['ngram%d' % k][m] = ngram(Xt, yt, X, y, k)
    for nm, mk, steps in (('mlp', MLP, 6000), ('tf', TF, 6000)):
        torch.manual_seed(0); model = train(mk(), steps, rng)
        res[nm] = {m: acc(model, m, rng) for m in (2, 3, 4, 5, 6)}
        res[nm + '_params'] = int(sum(p.numel() for p in model.parameters()))
        print(nm, res[nm], '%.0fs' % (time.time() - t0), flush=True)
    print('RESULT ' + json.dumps(res))
