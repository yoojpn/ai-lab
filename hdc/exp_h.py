"""
H群: 「賢さ」を言語で直接測る。単語予測のbit/語だけでなく、モデルが何を使えているかを調べる。
  T1 文脈への敏感さ: 直前から d 語前の語を別の語に置き換えたとき、正しい文脈の方を選べるか(2択, 偶然=50%)。
       d を変えて、モデルが「どこまで遠くを見ているか」を測る。
  T2 紛らわしい誤答との2択: 正解の語と、同じ頻度帯のランダムな語のどちらが高確率か(偶然=50%)。
  T3 次の語の当たり率: 1位/5位以内。
  T4 文章の生成: 実際にサンプリングして続きを書かせ、多様さ(distinct-n)・繰り返し・自分の驚きの量を、本物の文章と比べる。
モデル = 数えるモデル(n-gram+1語飛ばし, 重みはEM) と/または 極小NN(勾配あり)。
使い方: LADDER_CORPUS=... python3 exp_h.py '<json>'
  json: N, count(1/0), main(3|4), delta, nn_h(0なら無し), nn_n, nn_init(ppmi|rand), nq
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab, items
from lab import V, HIST, bits, log
import exp_scale as es, exp_ladder as L
from exp_g import Chain

es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


def chain_dist(ch, x):
    """1つの窓 x(HIST語)に対する、全語の確率分布(V,)"""
    p = np.full(V, 1.0 / V)
    for k, lv in enumerate(ch.levels):
        uc, tot, typ, up, pc = ch.t[k]
        cq = ch.code(x[None], lv)[0]
        i = int(np.searchsorted(uc, cq))
        if i >= len(uc) or uc[i] != cq:
            break
        lo = int(np.searchsorted(up, cq * V)); hi = int(np.searchsorted(up, (cq + 1) * V))
        new = ch.delta * typ[i] / tot[i] * p
        new[up[lo:hi] - cq * V] += np.maximum(pc[lo:hi] - ch.delta, 0) / tot[i]
        p = new
    return p


class Model:
    def __init__(s, chains, net, w):
        s.chains, s.net, s.w = chains, net, w

    def nn_probs(s, X, y=None):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                pr = torch.softmax(s.net(torch.from_numpy(X[i:i + 4096])), 1)
                out.append(pr.numpy() if y is None else pr[torch.arange(len(pr)), torch.from_numpy(y[i:i + 4096])].numpy())
        return np.concatenate(out).astype(np.float64)

    def prob(s, X, y):
        P = [c.probs(X, y)[0] for c in s.chains]
        if s.net is not None: P.append(s.nn_probs(X, y))
        return (s.w[:, None] * np.array(P)).sum(0)

    def dist(s, Xb):
        out = np.zeros((len(Xb), V)); k = len(s.chains)
        for r in range(len(Xb)):
            for j, c in enumerate(s.chains):
                out[r] += s.w[j] * chain_dist(c, Xb[r])
        if s.net is not None:
            out += s.w[k] * s.nn_probs(Xb)
        return out / out.sum(1, keepdims=True)


def dn(seqs, n):
    g = [tuple(q[i:i + n]) for q in seqs for i in range(len(q) - n + 1)]
    return len(set(g)) / max(1, len(g))


def rep4(seqs):
    r = t = 0
    for q in seqs:
        g = [tuple(q[i:i + 4]) for i in range(len(q) - 3)]
        t += len(g); r += len(g) - len(set(g))
    return r / max(1, t)


def run(cfg):
    t0 = time.time(); torch.set_num_threads(os.cpu_count() or 2)
    N = cfg['N']; nq = cfg.get('nq', 5000); delta = cfg.get('delta', 0.9); main_len = cfg.get('main', 3)
    nn_h = cfg.get('nn_h', 0); nn_n = cfg.get('nn_n', 1000000); use_count = cfg.get('count', 1)
    ctx = lab.load(N); N = len(ctx.Xtr)
    inv = {i: w for w, i in ctx.vocab.items()}; inv[0] = '<unk>'
    rng = np.random.default_rng(0)
    res = dict(cfg=cfg, N=N); ops = {}
    chains = []
    if use_count:
        for lvs in ([tuple(range(1, k + 1)) for k in range(main_len + 1)], [(), (2,), (2, 3), (2, 3, 4)], [(), (1,), (1, 3), (1, 3, 4)]):
            ch = Chain(ctx.Xtr, ctx.ytr, lvs, delta); chains.append(ch); ops['count'] = ops.get('count', 0) + ch.ops(N)
    net = None
    if nn_h:
        E0 = None
        if cfg.get('nn_init', 'ppmi') == 'ppmi':
            E0 = items.ppmi_embeddings(ctx, lab.DE); ops['svd'] = float(2 * V * V * lab.DE * 7 + 22 * N + 10 * V * V)
        idx = rng.choice(len(ctx.Xtr), min(nn_n, len(ctx.Xtr)), replace=False)
        net = lab.train(ctx, nn_h, 1.0, E0=E0, idx=idx); net.eval()
        ops['nn_train'] = float(6 * 5576 * nn_h * len(idx))
    log('学習完了 [%.0fs]' % (time.time() - t0))
    # 重み(検証)
    nv = 20000
    m0 = Model(chains, net, None)
    Pv = [c.probs(ctx.Xva[:nv], ctx.yva[:nv])[0] for c in chains]
    if net is not None: Pv.append(m0.nn_probs(ctx.Xva[:nv], ctx.yva[:nv]))
    w = L.em(np.array(Pv)); M = Model(chains, net, w)
    res['weights'] = [float(x) for x in w]; res['ops'] = ops; res['ops_total'] = float(sum(ops.values()))
    X, y = ctx.Xte[:nq], ctx.yte[:nq]
    res['bits'] = bits(M.prob(X, y))
    log('bit/語 %.4f' % res['bits'])
    # T1 文脈への敏感さ
    pt = M.prob(X, y); uni = np.bincount(ctx.ytr[:2000000], minlength=V).astype(np.float64); uni /= uni.sum()
    res['T1'] = {}
    for d in (1, 2, 3, 4, 6, 8, 12):
        Xc = X.copy(); Xc[:, HIST - d] = rng.choice(V, nq, p=uni)
        pc = M.prob(Xc, y)
        res['T1'][str(d)] = float(((pt > pc) + 0.5 * (pt == pc)).mean())
    log('T1 ' + json.dumps(res['T1']))
    # T2 紛らわしい誤答
    y2 = np.clip(np.round(y * (1 + rng.uniform(-0.1, 0.1, nq))).astype(np.int64), 1, V - 1)
    y2 = np.where(y2 == y, np.clip(y2 + 1, 1, V - 1), y2)
    p2 = M.prob(X, y2)
    res['T2'] = float(((pt > p2) + 0.5 * (pt == p2)).mean())
    log('T2 %.4f' % res['T2'])
    # T3 次の語の当たり率
    n3 = min(2000, nq); D = M.dist(X[:n3]); top = np.argsort(-D, 1)[:, :5]
    res['T3_top1'] = float((top[:, 0] == y[:n3]).mean()); res['T3_top5'] = float((top == y[:n3, None]).any(1).mean())
    log('T3 top1 %.4f top5 %.4f' % (res['T3_top1'], res['T3_top5']))
    # T4 生成(40語)。本物の続きと比べる
    L40, B = 40, 100
    real, pref = [], []
    for i in range(0, len(ctx.Xte), max(1, len(ctx.Xte) // 2000)):
        d = ctx.te_i[ctx.mte[i]]; p = ctx.pte[i]
        if p + L40 <= len(d):
            real.append(list(d[p:p + L40])); pref.append(ctx.Xte[i])
        if len(real) == B: break
    cur = np.array(pref); gen = [[] for _ in range(len(cur))]; surp = []
    for t in range(L40):
        D = M.dist(cur); cdf = np.cumsum(D, 1); u = rng.random((len(cur), 1))
        tok = (cdf < u).sum(1).clip(0, V - 1)
        surp.append(-np.log2(D[np.arange(len(cur)), tok] + 1e-12))
        for i, tk in enumerate(tok): gen[i].append(int(tk))
        cur = np.concatenate([cur[:, 1:], tok[:, None]], 1)
    res['T4'] = dict(n=len(gen), gen_distinct1=dn(gen, 1), gen_distinct2=dn(gen, 2), gen_distinct3=dn(gen, 3), gen_rep4=rep4(gen),
                     real_distinct1=dn(real, 1), real_distinct2=dn(real, 2), real_distinct3=dn(real, 3), real_rep4=rep4(real),
                     gen_mean_surprisal=float(np.mean(surp)))
    res['T4_samples'] = [' '.join(inv.get(int(w_), '?') for w_ in list(pref[i]) + gen[i]) for i in range(3)]
    log('T4 ' + json.dumps(res['T4']))
    for s_ in res['T4_samples']: log('サンプル: ' + s_)
    res['sec'] = time.time() - t0
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
