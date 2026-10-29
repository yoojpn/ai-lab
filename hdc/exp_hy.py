"""
HY: 表(CB2) + 小さな骨格(座標 + 位置の束プーリング + ランダムReLU層 + 最小二乗の読み出し)。 勾配降下なし。
 骨格 (全部「数える・SVD・最小二乗」だけで作る。 重みの微調整は混ぜる数十個だけ)
   座標     : 語の座標 E = PPMI(距離1,2) の SVD (V x 64)。 使う次元 dE を変える
   Attention: 位置の束(1,2,3,4,5-8,9-16,17-32,33-64)ごとに、束の中の語の座標を平均 -> 8 x dE 個の特徴 (位置で見る場所を決める版)
   層       : 'lin'  = 特徴をそのまま最小二乗で次の語に読み出す
              'relu' = ランダム射影 + ReLU (幅 m) を挟んでから最小二乗で読み出す
   読み出し : 次の語の one-hot への最小二乗(リッジ)。 閉形式 W = (A^T A + λ n I)^-1 A^T Y。 勾配降下なし。
 比べるもの (同じテスト窓、bit/語)
   T      : 表だけ (CB2 の +all_gated)
   S_i    : 骨格 i だけ (uni + 骨格の2特徴)
   TS_i   : 表 + 骨格 i (ゲートつき)
 費用: 推論の計算(語あたり MAC)と学習の計算(FLOP, 秒)を式で出す。 密LLM(exp_y, 4層Transformer)の参照値は PLAN.md の測定値。
使い方: LADDER_CORPUS=... python3 exp_hy.py '{"N":16000000,"ntune":4000,"ntest":20000,"nfit":1500000}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
from exp_cb import V, SEQ, LN2, BUCKETS, log

NF = 64
NB = len(BUCKETS)
SIZES = [('lin', 16, 0), ('lin', 32, 0), ('lin', 64, 0), ('relu', 32, 256), ('relu', 64, 512), ('relu', 64, 1024)]
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]          # 相対リッジ (n を掛ける)
# 参照: 密LLM (4層Transformer, seq64, 2エポック, PLAN.md の測定値)
DENSE_BITS = {'d128': {2000000: 6.634, 4000000: 5.936, 8000000: 5.435, 16000000: 5.130}, 'd64': {16000000: 5.548}, 'd256': {16000000: 4.872}}


def dense_macs(d, nl=4, seq=64):
    return nl * (12 * d * d + seq * d) + d * V


def name(sz):
    return ('lin%d' % sz[1]) if sz[0] == 'lin' else ('relu%dx%d' % (sz[1], sz[2]))


def skel_macs(sz):
    k, dE, m = sz
    pool = 64 * dE
    if k == 'lin':
        return pool + NB * dE * V
    return pool + NB * dE * m + m * V


def pooled(X, E):
    return torch.stack([E[X[:, [SEQ - j for j in Bk]]].mean(1) for Bk in BUCKETS], 1)      # [n, 8, 64]


class Skel:
    def __init__(self, E, mu, sd):
        self.E, self.mu, self.sd = E, mu, sd
        self.W = {}; self.W1 = {}; self.b1 = {}; self.sc = {}

    def norm_pool(self, X):
        return (pooled(X, self.E) - self.mu) / self.sd

    def hidden(self, P, sz):
        k, dE, m = sz
        z = P[:, :, :dE].reshape(len(P), NB * dE)
        if k == 'lin':
            h = z
        else:
            h = torch.relu(z @ self.W1[name(sz)] + self.b1[name(sz)])
        return torch.cat([h, torch.ones(len(h), 1)], 1)

    def scores(self, X, szs, lamsel):
        P = self.norm_pool(X); out = {}
        for sz in szs:
            out[name(sz)] = self.hidden(P, sz) @ self.W[(name(sz), lamsel[name(sz)])]
        return out


def fit_skeleton(x, E, nfit, rng):
    n = len(x); pos = np.sort(rng.choice(n - SEQ, nfit, replace=False)) + SEQ
    xs = torch.from_numpy(x.astype(np.int64)); ar = torch.arange(-SEQ, 0)
    c = 20000
    # 標準化の統計 (最初の塊)
    X0 = xs[torch.from_numpy(pos[:c])[:, None] + ar[None, :]]
    P0 = pooled(X0, E); mu = P0.mean(0); sd = P0.std(0) + 1e-6
    S = Skel(E, mu, sd)
    g = torch.Generator().manual_seed(1)
    for sz in SIZES:
        if sz[0] == 'relu':
            S.W1[name(sz)] = torch.randn(NB * sz[1], sz[2], generator=g) / math.sqrt(NB * sz[1])
            S.b1[name(sz)] = torch.randn(sz[2], generator=g) * 0.3
    dims = {'lin': NB * NF + 1}
    G = {'lin': torch.zeros(dims['lin'], dims['lin'], dtype=torch.float64)}
    Y = {'lin': torch.zeros(dims['lin'], V, dtype=torch.float64)}
    for sz in SIZES:
        if sz[0] == 'relu':
            G[name(sz)] = torch.zeros(sz[2] + 1, sz[2] + 1, dtype=torch.float64); Y[name(sz)] = torch.zeros(sz[2] + 1, V, dtype=torch.float64)
    flops = 0.0
    for a in range(0, nfit, c):
        pp = torch.from_numpy(pos[a:a + c]); cc = len(pp)
        X = xs[pp[:, None] + ar[None, :]]; y = xs[pp]
        P = (pooled(X, E) - mu) / sd
        flops += cc * (SEQ * NF)
        A = torch.cat([P.reshape(cc, NB * NF), torch.ones(cc, 1)], 1)
        G['lin'] += (A.T @ A).double(); Yc = torch.zeros(A.shape[1], V); Yc.index_add_(1, y, A.T.contiguous()); Y['lin'] += Yc.double()
        flops += cc * (2 * A.shape[1] ** 2 + A.shape[1])
        for sz in SIZES:
            if sz[0] != 'relu':
                continue
            Hh = S.hidden(P, sz); nm = name(sz)
            G[nm] += (Hh.T @ Hh).double(); Yc = torch.zeros(Hh.shape[1], V); Yc.index_add_(1, y, Hh.T.contiguous()); Y[nm] += Yc.double()
            flops += cc * (2 * NB * sz[1] * sz[2] + 2 * Hh.shape[1] ** 2 + Hh.shape[1])
        if (a // c) % 10 == 0:
            log('骨格の最小二乗 %d / %d' % (a, nfit))
    # 解く
    for sz in SIZES:
        nm = name(sz)
        if sz[0] == 'lin':
            idx = [b * NF + j for b in range(NB) for j in range(sz[1])] + [NB * NF]
            Gs = G['lin'][idx][:, idx]; Ys = Y['lin'][idx]
        else:
            Gs, Ys = G[nm], Y[nm]
        for lam in LAMS:
            R = Gs + lam * nfit * torch.eye(len(Gs), dtype=torch.float64)
            S.W[(nm, lam)] = torch.linalg.solve(R, Ys).float()
            flops += (len(Gs) ** 3) * 2 / 3 + len(Gs) ** 2 * V * 2
    return S, flops


def main(cfg):
    global SIZES
    if cfg.get('sizes'):
        SIZES = [tuple(z) for z in cfg['sizes']]
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000); nfit = cfg.get('nfit', 1500000)
    x, Xva, yva, Xte, yte = B.load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d' % (len(x), len(yva), len(yte)))
    G = B.Model()
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    t0 = time.time()
    C = B.build_counts(x); log('距離の束ごとの共起を数えた')
    G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float()
    G.tri = B.build_tri(x); log('3-gram: %d 種類' % len(G.tri[0]))
    beta = cfg.get('beta', 1280.0)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, beta) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, Sv, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    Eraw = U[:, :NF] * Sv[:NF].sqrt()[None, :]
    En = Eraw / (Eraw.norm(dim=1, keepdim=True) + 1e-9)
    G.cls = B.kmeans(En, C2.K); del C, ppmi; log('語のクラス %d' % C2.K)
    cls_np = G.cls.numpy(); G.sp = {}
    for nm, spec in C2.SPECS.items():
        G.sp[nm] = C2.build_sparse(x, cls_np, spec, C2.MINCNT.get(nm, 1)); log('%s: 項目 %d' % (nm, len(G.sp[nm][0])))
    t_table = time.time() - t0
    t1 = time.time()
    S, fl_fit = fit_skeleton(x, Eraw, nfit, np.random.default_rng(0)); t_skel = time.time() - t1
    log('骨格の最小二乗 完了 %.0f 秒, %.2e FLOP' % (t_skel, fl_fit))
    del x
    t = time.time(); Fv, gv = C2.collect(Xva, G); log('調整窓の特徴 %.0f 秒, 専門家 %d 個' % (time.time() - t, len(Fv)))
    # λ の選択 (骨格だけの調整窓 bit で)
    lamsel = {}; scale = {}
    for sz in SIZES:
        nm = name(sz); best = None
        P = S.norm_pool(Xva); Hh = S.hidden(P, sz)
        for lam in LAMS:
            s = Hh @ S.W[(nm, lam)]; sc = s.std().item() + 1e-9
            Ft = {'uni': Fv['uni'], 'sk': s / sc, 'skl': torch.log(torch.clamp(s, min=1e-4))}
            _, _, b = C2.fit(Ft, ['uni', 'sk', 'skl'], yva, iters=40)
            log('%-11s λ=%g 骨格だけ 調整窓 %.4f' % (nm, lam, b))
            if best is None or b < best[0]:
                best = (b, lam, sc)
        lamsel[nm] = best[1]; scale[nm] = best[2]
        s = Hh @ S.W[(nm, best[1])]
        Fv['sk_' + nm] = s / best[2]; Fv['skl_' + nm] = torch.log(torch.clamp(s, min=1e-4))
        del s
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    fitted = {}
    w, D, b = C2.fit(Fv, allx, yva, gate=gv, gated=True); fitted['T'] = (allx, w, D, True); log('%-14s 調整窓 %.4f' % ('T', b))
    for sz in SIZES:
        nm = name(sz); sk = ['sk_' + nm, 'skl_' + nm]
        w, D, b = C2.fit(Fv, ['uni'] + sk, yva, iters=60); fitted['S_' + nm] = (['uni'] + sk, w, D, False); log('%-14s 調整窓 %.4f' % ('S_' + nm, b))
        w, D, b = C2.fit(Fv, allx + sk, yva, gate=gv, gated=True); fitted['TS_' + nm] = (allx + sk, w, D, True); log('%-14s 調整窓 %.4f' % ('TS_' + nm, b))
    tot = {vn: 0.0 for vn in fitted}; n_te = len(yte)
    for a in range(0, n_te, 250):
        Fc, gc = C2.all_feats(Xte[a:a + 250], G); yc = yte[a:a + 250]
        sc_ = S.scores(Xte[a:a + 250], SIZES, lamsel)
        for sz in SIZES:
            nm = name(sz); s = sc_[nm]
            Fc['sk_' + nm] = s / scale[nm]; Fc['skl_' + nm] = torch.log(torch.clamp(s, min=1e-4))
        with torch.no_grad():
            for vn, (nms, w, D, gated) in fitted.items():
                z = C2.logits(Fc, nms, w, gc, D); tot[vn] += Fn.cross_entropy(z, yc, reduction='sum').item()
        if (a // 250) % 20 == 0:
            log('テスト %d / %d' % (a, n_te))
    bits = {vn: tot[vn] / n_te / LN2 for vn in tot}
    tab_macs = 72 * V      # 距離の束のPMI行 64 + 重みづけ和 8 の V次元ベクトル + 疎な表の引き(少量)
    cores = os.cpu_count() or 1
    res = dict(cfg=cfg, test_bits=bits, lam=lamsel, tab_macs=tab_macs, skel_macs={name(s): skel_macs(s) for s in SIZES},
               t_table=t_table, t_skel=t_skel, flops_fit=fl_fit, cores=cores)
    log('--- 結果 (bit/語, 語あたり推論MAC, 学習) ---')
    log('T (表だけ)            %.4f  MAC %.2e  表の構築 %.0f 秒 x %d コア' % (bits['T'], tab_macs, t_table, cores))
    for sz in SIZES:
        nm = name(sz)
        log('S_%-11s 骨格だけ %.4f  MAC %.2e' % (nm, bits['S_' + nm], skel_macs(sz)))
        log('TS_%-10s 表+骨格  %.4f  MAC %.2e' % (nm, bits['TS_' + nm], tab_macs + skel_macs(sz)))
    log('骨格の学習: %.0f 秒 x %d コア, %.2e FLOP' % (t_skel, cores, fl_fit))
    for dn, dd in (('d64', 64), ('d128', 128), ('d256', 256)):
        if N in DENSE_BITS[dn]:
            log('密LLM %-5s %.3f  MAC %.2e  学習 %.2e FLOP(2エポック)' % (dn, DENSE_BITS[dn][N], dense_macs(dd), 6 * dense_macs(dd) * N * 2))
    log('RESULT_HY ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
