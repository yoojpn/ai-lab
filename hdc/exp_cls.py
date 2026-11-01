"""刷新の実験 7 (クラスの n-gram): 1 層目の潜在クラス (exp_lat の保存モデル) で 各語にクラスを付け、 前の 1〜3 語の クラスの列 を文脈にした 次の語の予測 (後ろ盾つき平滑化)。\n  (元の説明) 潜在クラスの確率モデルを EM 法 (期待回数を数えて割る; 勾配なし) で最尤学習する。
  P(w | 文脈) = sum_k pi_k * sum_z A_k(z | k 語前の語) * B(w | z)       (k = 1..K, z = Z 個の潜在クラス)
  目的は 「次の語の確率を当てる」 (対数尤度) そのもの。 PPMI-SVD (共起の要約) や 座標の二乗誤差とは 違う。
  n-gram (1000万語 5-gram; /tmp/lr) には 確率の線形補間 (1-g) P_ngram + g P_lat、 g は n-gram の自信度の区分ごとに 検証で決める。
使い方: python3 exp_lat.py '{"nidx":10000000,"ntr":3000000,"K":6,"Z":256,"iters":15,"nval":100000,"ntest":200000}'
"""
import sys, json, time
import numpy as np
from scipy import sparse
import engine as E

V = 32768


def main(cfg):
    t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); W = '/tmp/tok'; D = '/tmp/lr'
    nidx = cfg['nidx']; K = cfg.get('K', 6); Z = cfg.get('Z', 256); Vc = cfg.get('Vc', 20000); ntr = cfg['ntr']; iters = cfg.get('iters', 0); rng = np.random.default_rng(0)
    alpha = cfg.get('alpha', 0.1); beta = cfg.get('beta', 0.01)
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = nidx // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); N = len(S)
    if cfg.get('full'):
        lam = {int(k): v for k, v in json.load(open(W + '/lam.json')).items()}; M = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    else:
        lam = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model({n: '%s/final_o%d' % (D, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    freq = np.bincount(S.astype(np.int64), minlength=V); freq[:3] = 0; ids = np.argsort(-freq, kind='stable')[:Vc]; ids = ids[freq[ids] > 0]; Vc = len(ids)
    rank = np.full(V + 1, Vc, np.int64); rank[ids] = np.arange(Vc); r = rank[S.astype(np.int64)]        # 圏外は Vc
    Tsrc = Ttr if cfg.get('full') else S
    pos = np.sort(rng.integers(8 + K, len(Tsrc), int(ntr * 1.2))); ya = rank[np.asarray(Tsrc[pos]).astype(np.int64)]; pos = pos[ya < Vc][:ntr]; n = len(pos)
    X = np.stack([rank[np.asarray(Tsrc[pos - k]).astype(np.int64)] for k in range(1, K + 1)], axis=1); Y = rank[np.asarray(Tsrc[pos]).astype(np.int64)]; lg('学習行 %d, 語彙 %d, K=%d, Z=%d, ベース=%s' % (n, Vc, K, Z, '3.8億語 5-gram' if cfg.get('full') else '1000万語 5-gram'))
    mdl = np.load(cfg['model']); A1 = mdl['A'][0]; assert (mdl['ids'] == ids).all(); cl = np.concatenate([np.argmax(A1, axis=1)[:Vc], [Z]]).astype(np.int64); NC = Z + 1        # 語 -> クラス (圏外は Z)
    lam = cfg.get('lam', 5.0); CH = 1_000_000
    def cctx(Xc, order): return sum(cl[Xc[:, j]] * (NC ** j) for j in range(order))
    tabs = {}
    for order in (1, 2, 3):
        key = cctx(X, order) * Vc + Y; u, c = np.unique(key, return_counts=True); ctxu, inv = np.unique(u // Vc, return_inverse=True); Nc = np.bincount(inv, weights=c)
        tabs[order] = (u, c.astype(np.float64), ctxu, Nc); lg('クラス %d-gram: 文脈 %d, 項目 %d' % (order, len(ctxu), len(u)))
    pu_ = np.bincount(Y, minlength=Vc).astype(np.float64) / n
    def lookup(order, Xc, y_):
        u, c, ctxu, Nc = tabs[order]; cx = cctx(Xc, order); q = cx * Vc + y_; i = np.minimum(np.searchsorted(u, q), len(u) - 1); cnt = np.where(u[i] == q, c[i], 0.0)
        j = np.minimum(np.searchsorted(ctxu, cx), len(ctxu) - 1); N = np.where(ctxu[j] == cx, Nc[j], 0.0); return cnt, N
    pi = None
    def evalset(sp_, nn):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16); p = rng.integers(16 + K, len(T), int(nn * 1.02)); p = p[T[p] != 0][:nn]; return T, p
    def plat(T, p):
        rr = rank[T.astype(np.int64)]; yy = rr[p]; out = np.zeros(len(p))
        for a in range(0, len(p), 200000):
            sl = slice(a, a + 200000); pp = p[sl]; y_ = np.minimum(yy[sl], Vc - 1); ok = yy[sl] < Vc
            Xc = np.stack([rr[pp - k] for k in range(1, 4)], axis=1); P = pu_[y_]
            for order in (1, 2, 3):
                cnt, N = lookup(order, Xc, y_); P = (cnt + lam * P) / (N + lam)
            out[sl] = np.where(ok, P, 0.0)
        return out
    parts = {}; evalset_cache = {}
    for sp_, nn in (('val', cfg['nval']), ('test', cfg['ntest'])):
        T, p = evalset(sp_, nn); evalset_cache[sp_] = (T, p); lp, hi = M.logprob_at(T, p, return_hi=True); parts[sp_] = [2.0 ** lp, None, hi]
    G = np.r_[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    def bins(pl): return np.clip((-np.log2(np.maximum(pl, 1e-12)) // 2).astype(int), 0, 7)
    bits = lambda P, m=None: float(-np.log2(P if m is None else P[m]).mean())
    def mix_eval(lv, lt):
        pv = parts['val'][0]; bv_ = bins(pv); gbest = np.zeros(8)
        for b_ in range(8):
            mm = bv_ == b_
            if mm.sum() < 200: continue
            ll_ = [np.log2((1 - g) * pv[mm] + g * lv[mm]).mean() for g in G]; gbest[b_] = G[int(np.argmax(ll_))]
        pt = parts['test'][0]; gt = gbest[bins(pt)]; Pm = (1 - gt) * pt + gt * lt; unseen = parts['test'][2] <= 2
        return dict(ngram=bits(pt), mixed=bits(Pm), gain=bits(pt) - bits(Pm), ngram_unseen=bits(pt, unseen), mixed_unseen=bits(Pm, unseen), frac_unseen=float(unseen.mean()), g_by_bin=gbest.tolist())
    def fit_g(pv, lv):
        bv_ = bins(pv); gb = np.zeros(8)
        for b_ in range(8):
            mm = bv_ == b_
            if mm.sum() < 200: continue
            ll_ = [np.log2((1 - g) * pv[mm] + g * lv[mm]).mean() for g in G]; gb[b_] = G[int(np.argmax(ll_))]
        return gb
    def mix2(lv, lt):
        """二段階: (1) n-gram に頻度を混ぜて較正 (2) その上に専門家を混ぜる。 上乗せは (1) の後の値との差"""
        pv0, pt0 = parts['val'][0], parts['test'][0]; cv, ct = ctrl_parts['val'], ctrl_parts['test']
        g1 = fit_g(pv0, cv); pv1 = (1 - g1[bins(pv0)]) * pv0 + g1[bins(pv0)] * cv; pt1 = (1 - g1[bins(pt0)]) * pt0 + g1[bins(pt0)] * ct
        g2 = fit_g(pv1, lv); pt2 = (1 - g2[bins(pt1)]) * pt1 + g2[bins(pt1)] * lt; un = parts['test'][2] <= 2
        return dict(base=bits(pt1), final=bits(pt2), extra2=bits(pt1) - bits(pt2), base_unseen=bits(pt1, un), final_unseen=bits(pt2, un))
    pu = np.bincount(Y, minlength=Vc).astype(np.float64) / n; ctrl_parts = {}
    for sp_ in ('val', 'test'):
        T, p = evalset_cache[sp_]; yy = rank[T.astype(np.int64)][p]; ctrl_parts[sp_] = np.where(yy < Vc, pu[np.minimum(yy, Vc - 1)], 0.0)
    ctrl = mix_eval(ctrl_parts['val'], ctrl_parts['test'])
    def evaluate():
        lv = plat(*evalset_cache['val']); lt = plat(*evalset_cache['test']); r_ = mix_eval(lv, lt); r_['two_stage'] = mix2(lv, lt); lg('    二段階(頻度で較正した n-gram の上): 基準 %.4f -> %.4f 上乗せ %.4f bit | 初出文脈 %.4f -> %.4f' % (r_['two_stage']['base'], r_['two_stage']['final'], r_['two_stage']['extra2'], r_['two_stage']['base_unseen'], r_['two_stage']['final_unseen'])); r_['frac_in_vocab'] = float((lt > 0).mean()); r_['control_unigram'] = ctrl; r_['extra_over_control'] = r_['gain'] - ctrl['gain']; return r_
    lt_ = plat(*evalset_cache['test']); ut_ = ctrl_parts['test']; okk = (lt_ > 0) & (ut_ > 0)
    lg('診断: テストで 語彙内の語のみ  クラス3-gram 単独 %.4f bit / 頻度だけ %.4f bit  (件数 %d, クラス使用数 %d)' % (-np.log2(lt_[okk]).mean(), -np.log2(ut_[okk]).mean(), okk.sum(), len(np.unique(cl))))
    res = evaluate(); lg('  [検証] クラス3-gram: n-gram %.4f -> 混合 %.4f 利得 %.4f (対照 %.4f; 上乗せ %.4f) | 初出文脈 %.4f -> %.4f' % (res['ngram'], res['mixed'], res['gain'], ctrl['gain'], res['extra_over_control'], res['ngram_unseen'], res['mixed_unseen']))
    lg('RESULT_LAT ' + json.dumps(dict(cfg=cfg, res=res), ensure_ascii=False)); json.dump(dict(cfg=cfg, res=res), open('/tmp/cls_result_K%d_Z%d_n%d_%s.json' % (K, Z, ntr, 'full' if cfg.get('full') else 'sub'), 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
