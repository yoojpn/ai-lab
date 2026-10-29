"""
EM: 「次の語を予測しやすいように、文脈をグループに分け直す」を周回する (勾配降下・逆伝播なし)。
  文脈ベクトル x = 前の64語の座標を、位置の束8個で平均して並べたもの (512次元, 標準化)。 3つの見方: all(512), loc(束1-5, 320), far(束6-8, 192)。
  モデル(混合): p(y|x) = sum_k r_k(x) P(y|k),  r_k(x) = softmax_k( -s*|x-mu_k|^2/tau0 + log pi_k )
  KM(対照, 周回なし): k-means で mu を決め、各グループの次の語の回数 P(y|k) を数える。
  EM(周回あり)      : 各周回で
      E: 学習データの各位置が、グループ k に属する確率 r_nk ∝ exp(-s|x_n-mu_k|^2/tau0) pi_k P(y_n|k)   (正解 y_n を使う = 目的に向けて育てる)
      M: mu_k = r で重みづけした x の平均, pi_k = r の合計, P(y|k) = r で重みづけした回数 (+ 平滑化)
    予測時は y を使わず r_k(x) だけで混ぜる。
  専門家の値: log p(y|x) - log pu(y)  (表の専門家と同じ形)。 表(CB2 +all_gated)に足し、混ぜる重みだけ L-BFGS。
使い方: LADDER_CORPUS=... python3 exp_em.py '{"N":16000000,"K":1024,"nfit":800000,"em_iters":8}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
from exp_cb import V, SEQ, LN2, BUCKETS, log

NF = 64; NB = len(BUCKETS); DIN = NB * NF
VIEWS = {'all': slice(0, DIN), 'loc': slice(0, 5 * NF), 'far': slice(5 * NF, DIN)}
SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]
ALPHA = 2.0


def pooled(X, E):
    return torch.stack([E[X[:, [SEQ - j for j in Bk]]].mean(1) for Bk in BUCKETS], 1)


def d2(X, Mu, mm):
    return (X ** 2).sum(1)[:, None] - 2 * X @ Mu.T + mm[None, :]


def kmeans_fit(Xs, K, iters, g, c=20000):
    n, D = Xs.shape; Mu = Xs[torch.randperm(n, generator=g)[:K]].clone()
    for it in range(iters):
        mm = (Mu ** 2).sum(1); S = torch.zeros(K, D); Nk = torch.zeros(K)
        for a in range(0, n, c):
            X = Xs[a:a + c]; ai = d2(X, Mu, mm).argmin(1)
            S.index_add_(0, ai, X); Nk.index_add_(0, ai, torch.ones(len(X)))
        dead = Nk < 1
        Mu = torch.where(dead[:, None], Xs[torch.randperm(n, generator=g)[:K]], S / Nk.clamp(min=1)[:, None])
    return Mu


class Mix:
    def __init__(self, Mu, logpi, P, tau0):
        self.Mu, self.logpi, self.P, self.tau0 = Mu, logpi, P, tau0; self.mm = (Mu ** 2).sum(1)

    def logp(self, X, s, c=2000):
        out = []
        for a in range(0, len(X), c):
            g = -s * d2(X[a:a + c], self.Mu, self.mm) / self.tau0 + self.logpi
            out.append(torch.log(torch.softmax(g, 1) @ self.P + 1e-12))
        return torch.cat(out)


def hard_stats(Xs, ys, Mu, pu, c=20000):
    K = len(Mu); mm = (Mu ** 2).sum(1); Nk = torch.zeros(K); CntT = torch.zeros(V, K); dsum = 0.0
    for a in range(0, len(Xs), c):
        dd = d2(Xs[a:a + c], Mu, mm); mn, ai = dd.min(1); dsum += mn.sum().item()
        Nk.index_add_(0, ai, torch.ones(len(ai))); CntT.index_put_((ys[a:a + c], ai), torch.ones(len(ai)), accumulate=True)
    P = (CntT.T + ALPHA * pu[None, :]) / (Nk[:, None] + ALPHA)
    return Nk, P, dsum / len(Xs)


def em_step(Xs, ys, mix, s, pu, c=20000):
    K, D = mix.Mu.shape; n = len(Xs); logPT = torch.log(mix.P).T.contiguous()
    Nk = torch.zeros(K); S = torch.zeros(K, D); CntT = torch.zeros(V, K); ll = 0.0
    for a in range(0, n, c):
        X, y = Xs[a:a + c], ys[a:a + c]
        g = -s * d2(X, mix.Mu, mix.mm) / mix.tau0 + mix.logpi + logPT[y]
        lse = torch.logsumexp(g, 1); ll += lse.sum().item(); r = torch.exp(g - lse[:, None])
        Nk += r.sum(0); S += r.T @ X; CntT.index_add_(0, y, r)
    Mu = torch.where((Nk < 1)[:, None], mix.Mu, S / Nk.clamp(min=1e-6)[:, None])
    logpi = torch.log(Nk / Nk.sum() + 1e-9); P = (CntT.T + ALPHA * pu[None, :]) / (Nk[:, None] + ALPHA)
    return Mix(Mu, logpi, P, mix.tau0), ll / n


def bits(lp, y):
    return -lp[torch.arange(len(y)), y].mean().item() / LN2


def pick_scale(mix, Pv, yva):
    best = None
    for s in SCALES:
        b = bits(mix.logp(Pv, s), yva)
        if best is None or b < best[0]:
            best = (b, s)
    return best[1], best[0]


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000); nfit = cfg.get('nfit', 800000)
    K = cfg.get('K', 1024); km_iters = cfg.get('km_iters', 8); em_iters = cfg.get('em_iters', 8); c = 20000
    x, Xva, yva, Xte, yte = B.load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d, K=%d' % (len(x), len(yva), len(yte), K))
    G = B.Model()
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = B.build_counts(x); G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float()
    G.tri = B.build_tri(x)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, cfg.get('beta', 1280.0)) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, Sv, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    Eraw = U[:, :NF] * Sv[:NF].sqrt()[None, :]
    En = Eraw / (Eraw.norm(dim=1, keepdim=True) + 1e-9)
    G.cls = B.kmeans(En, C2.K); del C, ppmi
    cls_np = G.cls.numpy(); G.sp = {}
    for nm, spec in C2.SPECS.items():
        G.sp[nm] = C2.build_sparse(x, cls_np, spec, C2.MINCNT.get(nm, 1))
    log('表の構築 完了')
    rng = np.random.default_rng(0); pos = np.sort(rng.choice(len(x) - SEQ, nfit, replace=False)) + SEQ
    xs = torch.from_numpy(x.astype(np.int64)); ar = torch.arange(-SEQ, 0); del x
    X0 = xs[torch.from_numpy(pos[:c])[:, None] + ar[None, :]]; P0 = pooled(X0, Eraw); mu0 = P0.mean(0); sd0 = P0.std(0) + 1e-6
    norm = lambda X: ((pooled(X, Eraw) - mu0) / sd0).reshape(len(X), DIN)
    Pf = torch.empty(nfit, DIN); yf = xs[torch.from_numpy(pos)]
    for a in range(0, nfit, c):
        pp = torch.from_numpy(pos[a:a + c]); Pf[a:a + c] = norm(xs[pp[:, None] + ar[None, :]])
    del xs
    Pv = norm(Xva); log('文脈ベクトル 作成 (学習 %d 位置)' % nfit)
    gen = torch.Generator().manual_seed(1)
    models = {}; hist = {}
    for vn, sl in VIEWS.items():
        Xs = Pf[:, sl].contiguous(); Xv = Pv[:, sl].contiguous(); t0 = time.time()
        Mu = kmeans_fit(Xs, K, km_iters, gen)
        Nk, P, tau0 = hard_stats(Xs, yf, Mu, G.pu)
        mix = Mix(Mu, torch.log(Nk / Nk.sum() + 1e-9), P, tau0)
        s, b = pick_scale(mix, Xv, yva); models['km_' + vn] = (mix, s); hist[vn] = [dict(it=0, tune=b, s=s)]
        log('[%s] KM(周回なし): 調整窓 %.4f bit (s=%g, tau0=%.1f) %.0f 秒' % (vn, b, s, tau0, time.time() - t0))
        for it in range(1, em_iters + 1):
            mix, ll = em_step(Xs, yf, mix, s, G.pu)
            s2, b = pick_scale(mix, Xv, yva); hist[vn].append(dict(it=it, tune=b, s=s2, ll=-ll / LN2))
            log('[%s] EM 周回%d: 学習の対数尤度 %.4f bit/語, 調整窓の専門家だけ %.4f bit (s=%g)' % (vn, it, -ll / LN2, b, s2))
        models['em_' + vn] = (mix, s2)
    del Pf
    t = time.time(); Fv, gv = C2.collect(Xva, G); log('調整窓の特徴 %.0f 秒' % (time.time() - t))
    for nm, (mix, s) in models.items():
        vn = nm.split('_')[1]; Fv[nm] = mix.logp(Pv[:, VIEWS[vn]].contiguous(), s) - G.logpu[None, :]
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    kmn = ['km_' + v for v in VIEWS]; emn = ['em_' + v for v in VIEWS]
    variants = {'T': (allx, True), 'T+KM': (allx + kmn, True), 'T+EM': (allx + emn, True), 'KM_only': (['uni'] + kmn, False), 'EM_only': (['uni'] + emn, False)}
    fitted = {}
    for vn, (nms, gated) in variants.items():
        w, D, b = C2.fit(Fv, nms, yva, gate=gv, gated=gated); fitted[vn] = (nms, w, D, gated); log('%-8s 調整窓 %.4f' % (vn, b))
    tot = {vn: 0.0 for vn in fitted}; n_te = len(yte)
    for a in range(0, n_te, 250):
        Xb = Xte[a:a + 250]; Fc, gc = C2.all_feats(Xb, G); yc = yte[a:a + 250]; Pb = norm(Xb)
        for nm, (mix, s) in models.items():
            vn = nm.split('_')[1]; Fc[nm] = mix.logp(Pb[:, VIEWS[vn]].contiguous(), s) - G.logpu[None, :]
        with torch.no_grad():
            for vn, (nms, w, D, gated) in fitted.items():
                z = C2.logits(Fc, nms, w, gc, D); tot[vn] += Fn.cross_entropy(z, yc, reduction='sum').item()
    bits_ = {vn: tot[vn] / n_te / LN2 for vn in tot}
    macs = sum(K * (VIEWS[v].stop - VIEWS[v].start) + 8 * V for v in VIEWS)
    log('--- 結果 (テスト, bit/語) ---')
    for vn in variants:
        log('%-8s %.4f' % (vn, bits_[vn]))
    log('専門家3つの推論MAC/語(上位8グループだけ使う場合) %.2e' % macs)
    log('RESULT_EM ' + json.dumps(dict(cfg=cfg, test_bits=bits_, hist=hist, macs=macs)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
