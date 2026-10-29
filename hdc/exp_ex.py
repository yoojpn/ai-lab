"""
EX: 表(CB2) に、勾配なしの専門家を足していく。 どれも「数える・近傍・EM・木」だけ。
  A  : 文脈グループ(EM周回)  3つの見方 all/loc/far   (exp_em.py と同じ)
  E  : 近傍検索 (kNN-LM型)。 文脈ベクトルが近い学習位置を引き、その次の語を使う
  D  : 話題 (PLSA)。 窓ごとの話題の混合比をEMで推定し、話題ごとの語の確率を足す
  C  : 文脈グループの組を数える。 (loc,far) / (all,直前語) / (far,直前語) をキーにした疎な表 (グループ番号を新しい語とみなす階層)
  B  : 再ランク。 全部を混ぜた確率の上位50候補を、決定木(LightGBM)で並べ直す。 木は混ぜた確率の「補正」だけを学ぶ
 評価: 表(T)に足した各変種と、全部(ALL)、ALL+再ランクの bit/語 (テスト窓)。
使い方: LADDER_CORPUS=... python3 exp_ex.py '{"N":16000000,"K":1024,"nfit":800000,"nidx":1500000}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
import exp_em as EM
from exp_cb import V, SEQ, LN2, BUCKETS, log

NF, NB, DIN, VIEWS = EM.NF, EM.NB, EM.DIN, EM.VIEWS
NC = 50


def build_sparse_ids(ids, y, mincnt):
    key = ids * V + y
    uk, cn = np.unique(key, return_counts=True)
    keep = cn >= mincnt; uk, cn = uk[keep], cn[keep].astype(np.float32)
    ctx = uk // V
    starts = np.r_[0, np.flatnonzero(ctx[1:] != ctx[:-1]) + 1]
    tot = np.add.reduceat(cn, starts); lens = np.diff(np.r_[starts, len(ctx)])
    return uk, cn, np.repeat(tot, lens).astype(np.float32)


def feat_sparse_ids(ids, T, pu):
    keys, cn, tot = T; n = len(ids)
    lo = np.searchsorted(keys, ids * V); hi = np.searchsorted(keys, ids * V + V)
    lens = hi - lo; out = torch.zeros(n, V); totrow = np.zeros(n, np.float32)
    if lens.sum() == 0:
        return out, totrow
    rows = np.repeat(np.arange(n), lens)
    pos = np.repeat(lo, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    bb = (keys[pos] - ids[rows] * V).astype(np.int64)
    c = torch.from_numpy(cn[pos]); tt = torch.from_numpy(tot[pos]); hit = lens > 0
    totrow[hit] = tot[lo[hit]]
    base = torch.log(C2.ALPHA / (torch.from_numpy(totrow) + C2.ALPHA))
    out[torch.from_numpy(hit)] = base[torch.from_numpy(hit)][:, None]
    rr = torch.from_numpy(rows); bt = torch.from_numpy(bb)
    out[rr, bt] = torch.log((c + C2.ALPHA * pu[bt]) / (tt + C2.ALPHA)) - torch.log(pu[bt])
    return out, totrow


class KNN:
    def __init__(self, Xi, yi):
        self.Xi, self.yi = Xi, yi; self.xx = torch.cat([(Xi[a:a + 200000].float() ** 2).sum(1) for a in range(0, len(Xi), 200000)])

    def search(self, Q, k=32, ic=200000):
        qq = (Q ** 2).sum(1)[:, None]; bd = None
        for a in range(0, len(self.Xi), ic):
            Xc = self.Xi[a:a + ic].float(); dd = qq - 2 * Q @ Xc.T + self.xx[a:a + ic][None, :]
            d, ix = torch.topk(dd, min(k, dd.shape[1]), dim=1, largest=False); ix = ix + a
            if bd is None:
                bd, bi = d, ix
            else:
                d2_, i2 = torch.topk(torch.cat([bd, d], 1), k, dim=1, largest=False); bd, bi = d2_, torch.gather(torch.cat([bi, ix], 1), 1, i2)
        return bd.clamp(min=0), bi

    def feat(self, d, ix, T, alpha, pu):
        w = torch.softmax(-d / (T * d.mean(1, keepdim=True) + 1e-9), 1); out = torch.zeros(len(d), V)
        out.scatter_add_(1, self.yi[ix], w)
        return torch.log((out + alpha * pu[None, :]) / (1 + alpha)) - torch.log(pu)[None, :]


def plsa_fit(W, Kz, iters, g, c=20000):
    n, L = W.shape
    P = torch.rand(V, Kz, generator=g) + 0.5; P = P / P.sum(0, keepdim=True)
    Th = torch.full((n, Kz), 1.0 / Kz) + 0.1 * torch.rand(n, Kz, generator=g); Th = Th / Th.sum(1, keepdim=True)
    for it in range(iters):
        Pa = torch.zeros(V, Kz)
        for a in range(0, n, c):
            Wc = W[a:a + c]; post = Th[a:a + c][:, None, :] * P[Wc]; post = post / (post.sum(-1, keepdim=True) + 1e-12)
            Th[a:a + c] = post.sum(1) / L; Pa.index_add_(0, Wc.reshape(-1), post.reshape(-1, Kz))
        P = (Pa + 1e-3) / (Pa + 1e-3).sum(0, keepdim=True)
    return P


def plsa_logp(Wc, P, iters=10):
    Kz = P.shape[1]; Th = torch.full((len(Wc), Kz), 1.0 / Kz); Pw = P[Wc]
    for _ in range(iters):
        post = Th[:, None, :] * Pw; post = post / (post.sum(-1, keepdim=True) + 1e-12); Th = post.mean(1)
    return torch.log(Th @ P.T + 1e-9)


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); nrank = cfg.get('nrank', 8000); ntest = cfg.get('ntest', 20000); nfit = cfg.get('nfit', 800000)
    K = cfg.get('K', 1024); km_iters = cfg.get('km_iters', 8); em_iters = cfg.get('em_iters', 8); nidx = cfg.get('nidx', 1500000)
    ntopic = cfg.get('ntopic', 300000); Kz = cfg.get('Kz', 128); c = 20000
    x, Xva, yva, Xte, yte = B.load(N, ntune + nrank, ntest)
    Xrk, yrk = Xva[ntune:], yva[ntune:]; Xva, yva = Xva[:ntune], yva[:ntune]
    log('学習 %d 語, 調整窓 %d, 再ランク学習窓 %d, テスト窓 %d' % (len(x), len(yva), len(yrk), len(yte)))
    G = B.Model(); cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = B.build_counts(x); G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float(); G.tri = B.build_tri(x)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, cfg.get('beta', 1280.0)) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, Sv, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    Eraw = U[:, :NF] * Sv[:NF].sqrt()[None, :]; En = Eraw / (Eraw.norm(dim=1, keepdim=True) + 1e-9)
    G.cls = B.kmeans(En, C2.K); del C, ppmi
    cls_np = G.cls.numpy(); G.sp = {}
    for nm, spec in C2.SPECS.items():
        G.sp[nm] = C2.build_sparse(x, cls_np, spec, C2.MINCNT.get(nm, 1))
    log('表の構築 完了')
    xs = torch.from_numpy(x.astype(np.int64)); ar = torch.arange(-SEQ, 0); n = len(xs)
    rng = np.random.default_rng(0); pos = np.sort(rng.choice(n - SEQ, nfit, replace=False)) + SEQ
    X0 = xs[torch.from_numpy(pos[:c])[:, None] + ar[None, :]]; P0 = EM.pooled(X0, Eraw); mu0 = P0.mean(0); sd0 = P0.std(0) + 1e-6
    norm = lambda X: ((EM.pooled(X, Eraw) - mu0) / sd0).reshape(len(X), DIN)
    Pf = torch.empty(nfit, DIN); yf = xs[torch.from_numpy(pos)]
    for a in range(0, nfit, c):
        pp = torch.from_numpy(pos[a:a + c]); Pf[a:a + c] = norm(xs[pp[:, None] + ar[None, :]])
    Pv = norm(Xva); gen = torch.Generator().manual_seed(1)
    # ---- A: 文脈グループ (EM周回)
    models = {}
    for vn, sl in VIEWS.items():
        Xs = Pf[:, sl].contiguous(); Xv = Pv[:, sl].contiguous()
        Mu = EM.kmeans_fit(Xs, K, km_iters, gen); Nk, P, tau0 = EM.hard_stats(Xs, yf, Mu, G.pu)
        mix = EM.Mix(Mu, torch.log(Nk / Nk.sum() + 1e-9), P, tau0); s, b = EM.pick_scale(mix, Xv, yva)
        for it in range(em_iters):
            mix, ll = EM.em_step(Xs, yf, mix, s, G.pu)
        s, b = EM.pick_scale(mix, Xv, yva); models[vn] = (mix, s); log('[A:%s] EM %d周 調整窓の専門家だけ %.4f bit (s=%g)' % (vn, em_iters, b, s))
    del Pf
    # ---- E: 近傍検索の索引
    posi = np.sort(np.random.default_rng(2).choice(n - SEQ, nidx, replace=False)) + SEQ
    Xi = torch.empty(nidx, DIN, dtype=torch.float16)
    for a in range(0, nidx, c):
        pp = torch.from_numpy(posi[a:a + c]); Xi[a:a + c] = norm(xs[pp[:, None] + ar[None, :]]).half()
    knn = KNN(Xi, xs[torch.from_numpy(posi)]); t = time.time(); dq, iq = knn.search(Pv); best = None
    for T in (0.02, 0.05, 0.1, 0.3):
        for al in (0.05, 0.2, 1.0):
            b = EM.bits(knn.feat(dq, iq, T, al, G.pu) + G.logpu[None, :], yva)
            if best is None or b < best[0]:
                best = (b, T, al)
    _, kT, kA = best; log('[E] kNN 索引 %d 位置, 調整窓の専門家だけ %.4f bit (T=%g, α=%g), 検索 %.0f 秒' % (nidx, best[0], kT, kA, time.time() - t))
    # ---- D: 話題 (PLSA)
    Wd = xs[torch.from_numpy(pos[:ntopic])[:, None] + torch.arange(-SEQ, 1)[None, :]]
    t = time.time(); Ptop = plsa_fit(Wd, Kz, cfg.get('topic_iters', 15), gen); del Wd
    bt = EM.bits(plsa_logp(Xva, Ptop), yva); log('[D] 話題 %d個, 調整窓の専門家だけ %.4f bit, %.0f 秒' % (Kz, bt, time.time() - t))
    # ---- C: 文脈グループの組を数える (全位置のグループ番号 -> 疎な表)
    t = time.time(); asg = {vn: np.zeros(n - SEQ, np.int64) for vn in VIEWS}
    for a in range(SEQ, n, 100000):
        p = torch.arange(a, min(a + 100000, n)); Pc = norm(xs[p[:, None] + ar[None, :]])
        for vn, sl in VIEWS.items():
            mix, s = models[vn]; g = -s * EM.d2(Pc[:, sl], mix.Mu, mix.mm) / mix.tau0 + mix.logpi
            asg[vn][a - SEQ:a - SEQ + len(p)] = g.argmax(1).numpy()
    xn = xs.numpy(); yt = xn[SEQ:]; w1 = xn[SEQ - 1:n - 1]
    CT = {'ctLF': build_sparse_ids(asg['loc'] * K + asg['far'], yt, 1), 'ctAW': build_sparse_ids(asg['all'] * V + w1, yt, 1),
          'ctFW': build_sparse_ids(asg['far'] * V + w1, yt, 1)}
    log('[C] グループの組の表 項目数 %s, %.0f 秒' % ({k_: len(v[0]) for k_, v in CT.items()}, time.time() - t)); del asg

    def view_logp(Pb):
        return {'em_' + vn: models[vn][0].logp(Pb[:, sl].contiguous(), models[vn][1]) - G.logpu[None, :] for vn, sl in VIEWS.items()}

    def feats(Xb):
        Fc, gc = C2.all_feats(Xb, G); Pb = norm(Xb); Fc.update(view_logp(Pb))
        d, ix = knn.search(Pb); Fc['knn'] = knn.feat(d, ix, kT, kA, G.pu)
        Fc['topic'] = plsa_logp(Xb, Ptop) - G.logpu[None, :]
        w1b = Xb[:, SEQ - 1].numpy(); a_ = {}
        for vn, sl in VIEWS.items():
            mix, s = models[vn]; a_[vn] = (-s * EM.d2(Pb[:, sl], mix.Mu, mix.mm) / mix.tau0 + mix.logpi).argmax(1).numpy().astype(np.int64)
        ids = {'ctLF': a_['loc'] * K + a_['far'], 'ctAW': a_['all'] * V + w1b, 'ctFW': a_['far'] * V + w1b}
        for nm_ in ids:
            Fc[nm_], _ = feat_sparse_ids(ids[nm_], CT[nm_], G.pu)
        return Fc, gc

    t = time.time(); parts = {}; gates = []
    for a in range(0, len(Xva), 250):
        Fc, gc = feats(Xva[a:a + 250]); gates.append(gc)
        for k_, v_ in Fc.items():
            parts.setdefault(k_, []).append(v_)
    Fv = {k_: torch.cat(v_) for k_, v_ in parts.items()}; gv = torch.cat(gates); del parts
    log('調整窓の特徴 %.0f 秒, 専門家 %d 個' % (time.time() - t, len(Fv)))
    bF = ['bF_%d' % k_ for k_ in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    Aa = ['em_all', 'em_loc', 'em_far']; Ee = ['knn']; Dd = ['topic']; Cc = ['ctLF', 'ctAW', 'ctFW']
    variants = {'T': allx, 'T+A': allx + Aa, 'T+E': allx + Ee, 'T+D': allx + Dd, 'T+C': allx + Cc, 'ALL': allx + Aa + Ee + Dd + Cc}
    fitted = {}
    for vn, nms in variants.items():
        w, D, b = C2.fit(Fv, nms, yva, gate=gv, gated=True); fitted[vn] = (nms, w, D, True); log('%-6s 調整窓 %.4f' % (vn, b))
    del Fv
    names_all = variants['ALL']

    def run_batches(X, y, tot=None, cand=None):
        for a in range(0, len(X), 250):
            Fc, gc = feats(X[a:a + 250]); yc = y[a:a + 250]
            with torch.no_grad():
                if tot is not None:
                    for vn, (nms, w, D, gated) in fitted.items():
                        z = C2.logits(Fc, nms, w, gc, D); tot[vn] += Fn.cross_entropy(z, yc, reduction='sum').item()
                zA = C2.logits(Fc, names_all, fitted['ALL'][1], gc, fitted['ALL'][2])
                lp = torch.log_softmax(zA, 1); lv, ci_ = lp.topk(NC, 1)
                cols = [lv.reshape(-1)]
                for nm_ in names_all[1:]:
                    cols.append(Fc[nm_].gather(1, ci_).reshape(-1))
                ent = -(lp.exp() * lp).sum(1)
                rk = torch.arange(NC).repeat(len(lv)).float(); gap = (lv[:, :1] - lv).reshape(-1)
                ex = [rk, gap, ent.repeat_interleave(NC), gc.float().repeat_interleave(NC)]
                M = torch.stack(cols + ex, 1).numpy()
                lab = (ci_ == yc[:, None]).float().reshape(-1).numpy()
                cand['M'].append(M); cand['lab'].append(lab); cand['lp'].append(lv.reshape(-1).numpy())
                cand['ltrue'].append(lp[torch.arange(len(yc)), yc].numpy()); cand['hit'].append((ci_ == yc[:, None]).any(1).numpy())
            if (a // 250) % 20 == 0:
                log('  バッチ %d / %d' % (a, len(X)))

    def newc():
        return {'M': [], 'lab': [], 'lp': [], 'ltrue': [], 'hit': []}
    t = time.time(); crk = newc(); run_batches(Xrk, yrk, None, crk); log('再ランク学習データ %.0f 秒' % (time.time() - t))
    tot = {vn: 0.0 for vn in fitted}; cte = newc(); t = time.time(); run_batches(Xte, yte, tot, cte); log('テスト %.0f 秒' % (time.time() - t))
    bits_ = {vn: tot[vn] / len(yte) / LN2 for vn in tot}
    # ---- B: 再ランク (LightGBM, 候補50個の中のソフトマックス交差エントロピーで、混ぜた確率への補正 raw を学ぶ)
    import lightgbm as lgb
    Mr = np.concatenate(crk['M']); lr_ = np.concatenate(crk['lab']); lpr = np.concatenate(crk['lp']); hitr = np.concatenate(crk['hit'])
    hw = np.flatnonzero(hitr); idx = (hw[:, None] * NC + np.arange(NC)[None, :]).reshape(-1)
    M = Mr[idx]; L = lr_[idx].reshape(-1, NC); LP = lpr[idx].reshape(-1, NC); ntr = int(len(hw) * 0.85)

    def lse_(s_):
        m_ = s_.max(1, keepdims=True); return (m_ + np.log(np.exp(s_ - m_).sum(1, keepdims=True)))[:, 0]
    LPtr, Ltr, LPva, Lva = LP[:ntr], L[:ntr], LP[ntr:], L[ntr:]

    def fobj(preds, ds):
        s_ = LPtr + preds.reshape(-1, NC); pr = np.exp(s_ - lse_(s_)[:, None])
        return (pr - Ltr).reshape(-1), np.ones(pr.size)      # 一定のヘシアン: 確率が極小の正解に対する更新の暴走を防ぐ

    def feval(preds, ds):
        s_ = LPva + preds.reshape(-1, NC); return 'ce', float((lse_(s_) - (s_ * Lva).sum(1)).mean()), False
    dtr = lgb.Dataset(M[:ntr * NC], label=np.zeros(ntr * NC)); dva = lgb.Dataset(M[ntr * NC:], label=np.zeros(len(M) - ntr * NC), reference=dtr)
    bst = lgb.train(dict(objective=fobj, metric='None', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, verbose=-1, num_threads=os.cpu_count() or 2),
                    dtr, num_boost_round=cfg.get('gbm_rounds', 400), valid_sets=[dva], feval=feval, callbacks=[lgb.early_stopping(30, verbose=False)])
    log('[B] LightGBM 木 %d 本, 再ランク学習窓の被覆率(正解が上位%d候補に入る) %.3f' % (bst.current_iteration(), NC, hitr.mean()))
    Mt = np.concatenate(cte['M']); lpt = np.concatenate(cte['lp']); hit = np.concatenate(cte['hit']); ltrue = np.concatenate(cte['ltrue']); labt = np.concatenate(cte['lab'])
    raw = bst.predict(Mt, raw_score=True); s = (lpt + raw).reshape(-1, NC); lpc = lpt.reshape(-1, NC)
    top = lse_(lpc); lsm = s - lse_(s)[:, None]
    pos_ = labt.reshape(-1, NC).argmax(1); ltr = np.where(hit, top + lsm[np.arange(len(lsm)), pos_], ltrue)
    bits_['ALL+B'] = float(-ltr.mean() / LN2); bits_['ALL_check'] = float(-ltrue.mean() / LN2); bits_['coverage'] = float(hit.mean())
    log('--- 結果 (テスト, bit/語) ---')
    for vn, v_ in bits_.items():
        log('%-10s %.4f' % (vn, v_))
    log('RESULT_EX ' + json.dumps(dict(cfg=cfg, test_bits=bits_)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
