"""
CB2: CB の「データを増やしたときの伸びの鈍り」を直す版。
 鈍りの見立て: 距離の束のPMI(2語の関係の足し算)と3-gramは、データが増えても早く頭打ちになる(表現の幅が足りない)。
 足すもの (どれも数え上げだけ。 重みは L-BFGS):
  g4,g5      : 4-gram, 5-gram (文脈3語, 4語)。 データが増えるほど文脈が見つかる
  s13,s23,s14: 飛び越し (w-1とw-3, w-2とw-3, w-1とw-4)。 語順の揺れに強い
  c123,wc,cc4: 語のクラス(200)で文脈を作る 3語クラス, (w-1+クラスw-2,w-3), 4語クラス。 見たことのない並びに、仲間の統計を貸す
  cache16/64 : 直近16/64語に出た語 (文書内の話題)
  ゲート     : 4-gram・5-gram の文脈が何回見つかったか(確かさ)で混ぜる重みを切り替える
 各 n-gram系の専門家は 「log((c+α·p(b))/(文脈の合計+α)) − log p(b)」。 文脈が一度も無ければ 0 (何も言わない)。
使い方: LADDER_CORPUS=... python3 exp_cb2.py '{"N":16000000,"ntune":5000,"ntest":20000}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
from exp_cb import V, SEQ, LN2, BUCKETS, log

K = 200
# 文脈の定義: 各要素 (何語前, 'w' か 'c')
SPECS = {
    'g4': [(1, 'w'), (2, 'w'), (3, 'w')], 'g5': [(1, 'w'), (2, 'w'), (3, 'w'), (4, 'w')],
    's13': [(1, 'w'), (3, 'w')], 's23': [(2, 'w'), (3, 'w')], 's14': [(1, 'w'), (4, 'w')],
    'c123': [(1, 'c'), (2, 'c'), (3, 'c')], 'wc': [(1, 'w'), (2, 'c'), (3, 'c')], 'cc4': [(1, 'c'), (2, 'c'), (3, 'c'), (4, 'c')],
}
MINCNT = {'g4': 1, 'g5': 2, 'cc4': 1}
CIRC = [(1, 1, 'exact'), (1, 1, 'class'), (2, 1, 'class'), (2, 1, 'exact'), (3, 1, 'class'), (3, 1, 'exact')]
ALPHA = 1.0


def ctx_ids_stream(x, cls, spec):
    L = max(j for j, _ in spec); n = len(x)
    ids = np.zeros(n - L, np.int64)
    for j, kind in spec:
        v = x[L - j:n - j]
        if kind == 'c':
            v = cls[v]; r = K
        else:
            r = V
        ids = ids * r + v
    return ids, L


def ctx_ids_win(X, cls, spec):
    ids = torch.zeros(len(X), dtype=torch.long)
    for j, kind in spec:
        v = X[:, SEQ - j]
        if kind == 'c':
            v = cls[v]; r = K
        else:
            r = V
        ids = ids * r + v
    return ids.numpy()


def build_sparse(x, cls, spec, mincnt):
    ids, L = ctx_ids_stream(x, cls, spec)
    key = ids * V + x[L:]
    del ids
    uk, cn = np.unique(key, return_counts=True)
    del key
    keep = cn >= mincnt
    uk, cn = uk[keep], cn[keep].astype(np.float32)
    ctx = uk // V
    starts = np.r_[0, np.flatnonzero(ctx[1:] != ctx[:-1]) + 1]
    tot = np.add.reduceat(cn, starts)
    lens = np.diff(np.r_[starts, len(ctx)])
    return uk, cn, np.repeat(tot, lens).astype(np.float32)


def feat_sparse(X, cls, spec, T, pu):
    keys, cn, tot = T
    n = len(X); ids = ctx_ids_win(X, cls, spec)
    lo = np.searchsorted(keys, ids * V); hi = np.searchsorted(keys, ids * V + V)
    lens = hi - lo; out = torch.zeros(n, V)
    totrow = np.zeros(n, np.float32)
    if lens.sum() == 0:
        return out, totrow
    rows = np.repeat(np.arange(n), lens)
    pos = np.repeat(lo, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    bb = (keys[pos] - ids[rows] * V).astype(np.int64)
    c = torch.from_numpy(cn[pos]); tt = torch.from_numpy(tot[pos])
    hit = lens > 0
    totrow[hit] = tot[lo[hit]]
    base = torch.log(ALPHA / (torch.from_numpy(totrow) + ALPHA))
    out[torch.from_numpy(hit)] = base[torch.from_numpy(hit)][:, None]
    rr = torch.from_numpy(rows); bt = torch.from_numpy(bb)
    out[rr, bt] = torch.log((c + ALPHA * pu[bt]) / (tt + ALPHA)) - torch.log(pu[bt])
    return out, totrow


def feat_cache(X, w):
    n = X.shape[0]; out = torch.zeros(n, V)
    out.scatter_add_(1, X[:, SEQ - w:], torch.ones(n, w))
    return torch.log1p(out)


def all_feats(X, G):
    F = {'uni': G.logpu[None, :].expand(len(X), V).contiguous()}
    for k, Bk in enumerate(BUCKETS):
        F['bF_%d' % k] = G.pmF[k][X[:, [SEQ - j for j in Bk]]].mean(1)
    F['tri'] = B.feat_tri(X, G.tri, G.C1, G.rs1, G.n1, G.pu)
    for (m, v, kind) in CIRC:
        F['ci_%d_%d_%s' % (m, v, kind)] = B.feat_circuit(X, m, v, kind, G.cls)
    aux = {}
    for nm, spec in SPECS.items():
        F[nm], tr = feat_sparse(X, G.cls, spec, G.sp[nm], G.pu)
        aux[nm] = tr
    F['cache16'] = feat_cache(X, 16); F['cache64'] = feat_cache(X, 64)
    t4, t5 = aux['g4'], aux['g5']
    g4 = np.digitize(t4, [0.5, 2.5, 9.5]); g5 = np.digitize(t5, [0.5, 2.5])
    gate = torch.from_numpy((g4 * 3 + g5).astype(np.int64))
    return F, gate


def collect(X, G, bs=250):
    parts = {}; gates = []
    for a in range(0, len(X), bs):
        F, g = all_feats(X[a:a + bs], G); gates.append(g)
        for k, t in F.items():
            parts.setdefault(k, []).append(t)
    return {k: torch.cat(v) for k, v in parts.items()}, torch.cat(gates)


NG = 12


def logits(F, names, w, gate=None, D=None):
    z = None
    for i, n in enumerate(names):
        wi = w[i] if D is None else (w[i] + D[gate, i])[:, None]
        t = wi * F[n]; z = t if z is None else z + t
    return z


def fit(F, names, y, gate=None, gated=False, lam=2e-3, iters=80):
    w = torch.tensor([1.0 if n == 'uni' else 0.3 for n in names], requires_grad=True)
    D = torch.zeros(NG, len(names), requires_grad=True) if gated else None
    params = [w] + ([D] if gated else [])
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = Fn.cross_entropy(logits(F, names, w, gate, D), y)
        if gated:
            loss = loss + lam * (D ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        b = Fn.cross_entropy(logits(F, names, w, gate, D), y).item() / LN2
    return w.detach(), (D.detach() if gated else None), b


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 5000); ntest = cfg.get('ntest', 20000)
    x, Xva, yva, Xte, yte = B.load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d' % (len(x), len(yva), len(yte)))
    G = B.Model()
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = B.build_counts(x); log('距離の束ごとの共起を数えた')
    G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float()
    G.tri = B.build_tri(x); log('3-gram: %d 種類' % len(G.tri[0]))
    best = None
    for beta in cfg.get('betas', [80.0, 320.0, 1280.0]):
        pm = [B.pmi_from_counts(Ck, G.pu, beta) for Ck in C]
        Fv = {'uni': G.logpu[None, :].expand(len(Xva), V).contiguous()}
        for k, Bk in enumerate(BUCKETS):
            Fv['bF_%d' % k] = torch.cat([pm[k][Xva[a:a + 250][:, [SEQ - j for j in Bk]]].mean(1) for a in range(0, len(Xva), 250)])
        w, _, b = fit(Fv, ['uni'] + ['bF_%d' % k for k in range(8)], yva, iters=40); log('beta=%g 調整窓 %.4f' % (beta, b))
        if best is None or b < best[0]:
            best = (b, beta)
        del pm
    beta = best[1]; log('beta=%g を採用' % beta)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, beta) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, S, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :64] * S[:64].sqrt()[None, :]; E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    G.cls = B.kmeans(E, K); del C, ppmi; log('語のクラス %d' % K)
    cls_np = G.cls.numpy()
    G.sp = {}
    for nm, spec in SPECS.items():
        G.sp[nm] = build_sparse(x, cls_np, spec, MINCNT.get(nm, 1)); log('%s: 項目 %d' % (nm, len(G.sp[nm][0])))
    del x
    t = time.time(); Fv, gv = collect(Xva, G); log('調整窓の特徴 %.0f 秒, 専門家 %d 個' % (time.time() - t, len(Fv)))
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c for c in CIRC]
    base = ['uni'] + bF + ['tri'] + ci
    variants = {
        'base': (base, False), '+cache': (base + ['cache16', 'cache64'], False),
        '+ngram': (base + ['g4', 'g5'], False), '+skip': (base + ['s13', 's23', 's14'], False),
        '+class': (base + ['c123', 'wc', 'cc4'], False),
    }
    allx = base + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    variants['+all'] = (allx, False); variants['+all_gated'] = (allx, True)
    fitted = {}
    for vn, (nm, gated) in variants.items():
        w, D, b = fit(Fv, nm, yva, gate=gv, gated=gated); fitted[vn] = (nm, w, D, gated); log('%-12s 調整窓 %.4f' % (vn, b))
    tot = {vn: 0.0 for vn in fitted}; n_te = len(yte)
    for a in range(0, n_te, 250):
        Fc, gc = all_feats(Xte[a:a + 250], G); yc = yte[a:a + 250]
        with torch.no_grad():
            for vn, (nm, w, D, gated) in fitted.items():
                z = logits(Fc, nm, w, gc, D); tot[vn] += Fn.cross_entropy(z, yc, reduction='sum').item()
        if (a // 250) % 20 == 0:
            log('テスト %d / %d' % (a, n_te))
    res = dict(cfg=cfg, beta=beta, test_bits={vn: tot[vn] / n_te / LN2 for vn in tot},
               weights={vn: dict(zip(fitted[vn][0], [round(float(v), 3) for v in fitted[vn][1]])) for vn in ('+all',)},
               entries={nm: int(len(G.sp[nm][0])) for nm in SPECS})
    for vn in tot:
        log('テスト %-12s %.4f bit/語' % (vn, res['test_bits'][vn]))
    log('RESULT_CB2 ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
