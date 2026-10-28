"""
CB: 勾配で大きな表を学習しない言語モデルの最小版 (数え上げ + SVD + ごく少数の混ぜる重み)。
  部品
   uni   : 1語の出現確率 (log)
   bF_k  : 距離の束k (1,2,3,4,5-8,9-16,17-32,33-64) ごとの「前の語 -> 次の語」の PMI を、束の中の語で平均 (Attentionの全語を等しく見る版)
   bR*_k : 同じ PMI 行列を SVD で階数rに圧縮したもの (座標 = 低ランク因子)
   tri   : 3-gram (絶対割引の補間) の log 確率
   ci_*  : 回路の自動探索。 「直前のm語が文脈内で前に出た位置」の v 語先の語を写す (m,v,一致の種類 exact/class を総当たり)
   at_*  : Attention相当。 直前の語と、前の各語の関連度 (窓つき共起の PMI) で softmax の重みをつけて、各語の PMI 行を混ぜる
  混ぜる重み (数十個) だけ L-BFGS で決める。表と座標は数え上げ + SVD のみ。
  評価: テスト先頭 ntest 窓 (exp_y と同じ), 直前64語 -> 次の1語。 bit/語。
使い方: LADDER_CORPUS=... python3 exp_cb.py '{"N":16000000,"ntune":2500,"ntest":20000}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
import numpy as np, torch
import torch.nn.functional as Fn
import exp_scale as es
import exp_far as XF
from exp_t3 import split_windows
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)
V = es.V; SEQ = 64; LN2 = math.log(2)
BUCKETS = [[1], [2], [3], [4], [5, 6, 7, 8], list(range(9, 17)), list(range(17, 33)), list(range(33, 65))]
T0 = time.time()


def log(s):
    print('[%6.0fs] %s' % (time.time() - T0, s), flush=True)


def load(N, ntune, ntest):
    corp = es.build_corpus()
    cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr, va, te = conv(corp[2]), conv(corp[1]), conv(corp[0])
    x = np.concatenate(tr)[:N]
    _, yte, mte, pte = XF.lwindows_meta(te, es.N_TEST)
    yte = yte[:ntest]; Xte = split_windows(te, mte, pte, len(yte), SEQ)
    _, yva, mva, pva = XF.lwindows_meta(va, es.N_VAL)
    yva = yva[:ntune]; Xva = split_windows(va, mva, pva, len(yva), SEQ)
    return x, torch.from_numpy(Xva), torch.from_numpy(yva), torch.from_numpy(Xte), torch.from_numpy(yte)


def build_counts(x):
    C = []
    for B in BUCKETS:
        M = torch.zeros(V * V, dtype=torch.float32)
        for j in B:
            k = x[:-j] * V + x[j:]                     # a=x[i-j] (前), b=x[i] (次)
            M += torch.from_numpy(np.bincount(k, minlength=V * V).astype(np.float32))
        C.append(M.view(V, V))
    return C


def pmi_from_counts(Ck, pu, beta):
    rs = Ck.sum(1, keepdim=True)
    P = (Ck + beta * pu[None, :]) / (rs + beta)
    return torch.log(P) - torch.log(pu)[None, :]


def lowrank(M, r, q):
    U, S, Vv = torch.svd_lowrank(M, q=q, niter=3)
    return (U[:, :r] * S[:r]) @ Vv[:, :r].T


def kmeans(E, K=200, it=25):
    g = torch.Generator().manual_seed(0)
    C = E[torch.randperm(len(E), generator=g)[:K]].clone()
    for _ in range(it):
        a = torch.cdist(E, C).argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                C[k] = E[m].mean(0)
    return torch.cdist(E, C).argmin(1)


def build_tri(x):
    k = x[:-2] * (V * V) + x[1:-1] * V + x[2:]
    keys, cnts = np.unique(k, return_counts=True)
    return keys, cnts.astype(np.float32)


def feat_tri(X, tri, C1, rs1, n1, pu, D=0.75):
    keys, cnts = tri; n = len(X)
    a = X[:, -2].numpy(); b = X[:, -1].numpy(); out = torch.empty(n, V)
    for q in range(n):
        bq = int(b[q])
        Pbi = (torch.clamp(C1[bq] - D, min=0) + D * n1[bq] * pu) / rs1[bq] if rs1[bq] > 0 else pu
        base = int(a[q]) * V * V + bq * V
        lo = np.searchsorted(keys, base); hi = np.searchsorted(keys, base + V)
        if hi > lo:
            cs = torch.from_numpy((keys[lo:hi] - base).astype(np.int64)); c3 = torch.from_numpy(cnts[lo:hi]); c2 = c3.sum()
            p = Pbi * (D * len(cs) / c2); p[cs] += torch.clamp(c3 - D, min=0) / c2
        else:
            p = Pbi
        out[q] = torch.log(p + 1e-12)
    return out


def feat_circuit(X, m, v, kind, cls):
    n = X.shape[0]; out = torch.zeros(n, V)
    Xc = X if kind == 'exact' else cls[X]
    q = [Xc[:, SEQ - 1 - r] for r in range(m)]
    base_ok = (X[:, SEQ - 1] != 0) if kind == 'exact' else torch.ones(n, dtype=torch.bool)
    for i in range(m - 1, SEQ - 1 - v + 1):
        if i + v > SEQ - 1 or i >= SEQ - 1:
            break
        ok = base_ok.clone()
        for r in range(m):
            ok &= (Xc[:, i - r] == q[r])
        out.scatter_add_(1, X[:, i + v][:, None], ok.float()[:, None])
    return torch.log1p(out)


def feat_attn(X, S, tau, pm):
    n = X.shape[0]; a = X[:, -1]
    js = list(range(2, SEQ + 1))
    U = X[:, [SEQ - j for j in js]]
    w = torch.softmax(tau * S[a[:, None], U], 1)
    out = torch.zeros(n, V)
    for B, M in zip(BUCKETS, pm):
        sel = [js.index(j) for j in B if j >= 2]
        if sel:
            out += (w[:, sel, None] * M[U[:, sel]]).sum(1)
    return out


class Model:
    pass


def all_feats(X, G):
    """chunk X -> {専門家の名前: (n,V) の特徴}"""
    F = {'uni': G.logpu[None, :].expand(len(X), V).contiguous()}
    for tag, pms in (('F', G.pmF), ('R128', G.pmR128), ('R32', G.pmR32)):
        for k, B in enumerate(BUCKETS):
            F['b%s_%d' % (tag, k)] = pms[k][X[:, [SEQ - j for j in B]]].mean(1)
    F['tri'] = feat_tri(X, G.tri, G.C1, G.rs1, G.n1, G.pu)
    for (m, v, kind) in G.circuits:
        F['ci_%d_%d_%s' % (m, v, kind)] = feat_circuit(X, m, v, kind, G.cls)
    for head, S in (('syn', G.Ssyn), ('top', G.Stop)):
        for tau in (1.0, 3.0):
            F['at_%s_%g' % (head, tau)] = feat_attn(X, S, tau, G.pmF)
    return F


def collect(X, G, bs=250):
    parts = {}
    for a in range(0, len(X), bs):
        for k, t in all_feats(X[a:a + bs], G).items():
            parts.setdefault(k, []).append(t)
    return {k: torch.cat(v) for k, v in parts.items()}


def fit(F, names, y, init=None, iters=60):
    w = torch.tensor([(init or {}).get(n, 1.0 if n == 'uni' else 0.3) for n in names], requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=iters, line_search_fn='strong_wolfe')

    def logits(w):
        z = None
        for i, n in enumerate(names):
            t = w[i] * F[n]; z = t if z is None else z + t
        return z

    def closure():
        opt.zero_grad(); loss = Fn.cross_entropy(logits(w), y); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        loss = Fn.cross_entropy(logits(w), y).item()
    return w.detach(), loss / LN2


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 2500); ntest = cfg.get('ntest', 20000)
    x, Xva, yva, Xte, yte = load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d' % (len(x), len(yva), len(yte)))
    G = Model()
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = build_counts(x); log('距離の束ごとの共起を数えた')
    G.C1 = C[0]; G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float()
    G.tri = build_tri(x); log('3-gram: %d 種類' % len(G.tri[0]))
    # 窓つき共起の関連度 (対称)
    def sym_pmi(idx, beta):
        Cs = sum(C[i] for i in idx); Cs = Cs + Cs.T
        return pmi_from_counts(Cs, G.pu, beta)
    G.circuits = [(m, v, kind) for m in (1, 2, 3) for v in (1, 2) for kind in ('exact', 'class')]
    # beta を調整窓で選ぶ
    best = None
    for beta in cfg.get('betas', [5.0, 20.0, 80.0]):
        G.pmF = [pmi_from_counts(Ck, G.pu, beta) for Ck in C]
        G.Ssyn = G.Stop = None; G.pmR128 = G.pmR32 = G.pmF; G.cls = None
        Fv = {'uni': G.logpu[None, :].expand(len(Xva), V).contiguous()}
        for k, B in enumerate(BUCKETS):
            parts = [G.pmF[k][Xva[a:a + 250][:, [SEQ - j for j in B]]].mean(1) for a in range(0, len(Xva), 250)]
            Fv['bF_%d' % k] = torch.cat(parts)
        names = ['uni'] + ['bF_%d' % k for k in range(8)]
        w, b = fit(Fv, names, yva); log('beta=%g  調整窓 bit/語 %.4f' % (beta, b))
        if best is None or b < best[0]:
            best = (b, beta)
    beta = best[1]; log('beta=%g を採用' % beta)
    G.pmF = [pmi_from_counts(Ck, G.pu, beta) for Ck in C]
    # 座標 (SVD) と階数圧縮
    G.pmR128 = [lowrank(M, 128, 144) for M in G.pmF]; G.pmR32 = [lowrank(M, 32, 48) for M in G.pmF]; log('SVD 圧縮 (r=128, 32)')
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, S, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :64] * S[:64].sqrt()[None, :]; E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    G.cls = kmeans(E, 200); log('語の座標 (64次元) -> 200 クラス')
    G.Ssyn = sym_pmi([0, 1, 2, 3], beta); G.Stop = sym_pmi(range(8), beta)
    del C

    t = time.time(); Fv = collect(Xva, G); log('調整窓の特徴 %.0f 秒, 専門家 %d 個' % (time.time() - t, len(Fv)))
    names_of = lambda tag: ['b%s_%d' % (tag, k) for k in range(8)]
    variants = {
        'uni': ['uni'],
        'tri': ['tri'],
        'pmi_full': ['uni'] + names_of('F'),
        'pmi_full+tri': ['uni'] + names_of('F') + ['tri'],
        'pmi_r128+tri': ['uni'] + names_of('R128') + ['tri'],
        'pmi_r32+tri': ['uni'] + names_of('R32') + ['tri'],
    }
    fitted = {}
    for vn, nm in variants.items():
        w, b = fit(Fv, nm, yva); fitted[vn] = (nm, w); log('%-16s 調整窓 %.4f bit/語' % (vn, b))
    # 回路の自動探索: pmi_full+tri に1個ずつ足して、得を測る
    nm0, w0 = fitted['pmi_full+tri']
    with torch.no_grad():
        z0 = sum(w0[i] * Fv[n] for i, n in enumerate(nm0))
    base_b = Fn.cross_entropy(z0, yva).item() / LN2
    mus = [0.25, 0.5, 1.0, 2.0, 4.0]
    cand = {}
    for name in [k for k in Fv if k.startswith('ci_') or k.startswith('at_')]:
        bb, bm = min((Fn.cross_entropy(z0 + m * Fv[name], yva).item() / LN2, m) for m in mus)
        cand[name] = (base_b - bb, bm)
    rank = sorted(cand.items(), key=lambda kv: -kv[1][0])
    for nme, (g, m) in rank:
        log('候補 %-16s 得 %+.4f bit  (重み %g)' % (nme, g, m))
    thr = cfg.get('thr', 0.003)
    adopt_ci = [n for n, (g, m) in rank if n.startswith('ci_') and g > thr]
    bestat = {}
    for n, (g, m) in rank:
        if n.startswith('at_') and g > thr:
            bestat.setdefault(n.split('_')[1], n)
    adopt_at = list(bestat.values())
    log('採用した回路: %s / Attention: %s' % (adopt_ci, adopt_at))
    variants2 = {'+circuits': nm0 + adopt_ci, '+attn': nm0 + adopt_at, '+circuits+attn': nm0 + adopt_ci + adopt_at}
    for vn, nm in variants2.items():
        w, b = fit(Fv, nm, yva); fitted[vn] = (nm, w); log('%-16s 調整窓 %.4f bit/語' % (vn, b))
    # テスト: 1回の通しで全部の変種を評価
    tot = {vn: 0.0 for vn in fitted}; totc = {n: 0.0 for n in cand}
    zb = fitted['pmi_full+tri']
    t = time.time(); n_te = len(yte)
    for a in range(0, n_te, 250):
        Fc = all_feats(Xte[a:a + 250], G); yc = yte[a:a + 250]
        with torch.no_grad():
            for vn, (nm, w) in fitted.items():
                z = sum(w[i] * Fc[n] for i, n in enumerate(nm)); tot[vn] += Fn.cross_entropy(z, yc, reduction='sum').item()
            zz = sum(zb[1][i] * Fc[n] for i, n in enumerate(zb[0]))
            for n, (g, m) in cand.items():
                totc[n] += Fn.cross_entropy(zz + m * Fc[n], yc, reduction='sum').item()
        if (a // 250) % 20 == 0:
            log('テスト %d / %d' % (a, n_te))
    res = dict(cfg=cfg, beta=beta, test_bits={vn: tot[vn] / n_te / LN2 for vn in tot},
               circuit_gain_tune={n: g for n, (g, m) in cand.items()},
               circuit_gain_test={n: tot['pmi_full+tri'] / n_te / LN2 - totc[n] / n_te / LN2 for n in cand},
               adopted=adopt_ci + adopt_at,
               macs_per_token=dict(pmi_full=64 * V, pmi_r128=128 * (64 + 8 * V), pmi_r32=32 * (64 + 8 * V)),
               entries=dict(pmi_full=8 * V * V, pmi_r128=8 * 2 * V * 128, pmi_r32=8 * 2 * V * 32, tri=int(len(G.tri[0]))))
    for vn in tot:
        log('テスト %-16s %.4f bit/語' % (vn, res['test_bits'][vn]))
    log('RESULT_CB ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
