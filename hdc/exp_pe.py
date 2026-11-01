"""刷新の実験 6 (掛け算の潜在クラス): 窓全体を 1 つのクラス z から生成する。 P(z|窓) ∝ pi_z prod_k P(語k|z)、 予測 P(w|窓)=sum_z P(z|窓) B(w|z)。 EM 法 (元: exp_lat.py)。\n  (元の説明) 潜在クラスの確率モデルを EM 法 (期待回数を数えて割る; 勾配なし) で最尤学習する。
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
    nidx = cfg['nidx']; K = cfg.get('K', 6); Z = cfg.get('Z', 256); Vc = cfg.get('Vc', 20000); ntr = cfg['ntr']; iters = cfg['iters']; rng = np.random.default_rng(0)
    alpha = cfg.get('alpha', 0.05); beta = cfg.get('beta', 0.01)
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
    Pk = rng.gamma(0.3, size=(K, Vc + 1, Z)).astype(np.float32) + 1e-3; Pk /= Pk.sum(1, keepdims=True)          # P(語 | z) (位置kごと; zごとに語で和=1)
    Bt = rng.gamma(0.3, size=(Vc, Z)).astype(np.float32) + 1e-3; Bt /= Bt.sum(0, keepdims=True)                 # P(次の語 | z)
    logpi = np.full(Z, -np.log(Z), np.float32); CH = 40000; lPk = np.log(Pk)
    def evalset(sp_, nn):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16); p = rng.integers(16 + K, len(T), int(nn * 1.02)); p = p[T[p] != 0][:nn]; return T, p
    def plat(T, p):
        rr = rank[T.astype(np.int64)]; yy = rr[p]; out = np.zeros(len(p))
        for a in range(0, len(p), 20000):
            sl = slice(a, a + 20000); pp = p[sl]; y_ = yy[sl]; ok = y_ < Vc
            lp = np.tile(logpi, (len(pp), 1))
            for k in range(K): lp += lPk[k][rr[pp - (k + 1)]]
            lp -= lp.max(1, keepdims=True); post = np.exp(lp); post /= post.sum(1, keepdims=True)
            out[sl] = np.where(ok, (post * Bt[np.minimum(y_, Vc - 1)]).sum(1).astype(np.float64), 0.0)
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
    for it in range(iters):
        Pacc = np.zeros_like(Pk); Bacc = np.zeros((Vc, Z), np.float32); pacc = np.zeros(Z); ll = 0.0; lBt = np.log(Bt)
        for a in range(0, n, CH):
            xs = X[a:a + CH]; ys = Y[a:a + CH]; m_ = len(ys)
            lp = logpi[None, :] + lBt[ys]
            for k in range(K): lp += lPk[k][xs[:, k]]
            mx = lp.max(1, keepdims=True); R = np.exp(lp - mx); sR = R.sum(1, keepdims=True); ll += float((np.log2(sR[:, 0]) + mx[:, 0] / np.log(2)).sum()); R /= sR
            ar = np.arange(m_); pacc += R.sum(0)
            for k in range(K): Pacc[k] += sparse.csr_matrix((np.ones(m_, np.float32), (xs[:, k], ar)), shape=(Vc + 1, m_)) @ R
            Bacc += sparse.csr_matrix((np.ones(m_, np.float32), (ys, ar)), shape=(Vc, m_)) @ R
        Pk = Pacc + alpha; Pk /= Pk.sum(1, keepdims=True); lPk = np.log(Pk); Bt = Bacc + beta; Bt /= Bt.sum(0, keepdims=True); logpi = np.log(pacc / pacc.sum() + 1e-12).astype(np.float32)
        lg('反復 %d: 学習の対数尤度 %.4f bit/行 (結合), 最大のクラスの割合 %.3f, 使われるクラス数(>0.1%%) %d' % (it + 1, -ll / n, float(np.exp(logpi).max()), int((np.exp(logpi) > 1e-3).sum())))
        if (it + 1) % cfg.get('eval_every', 5) == 0 or it + 1 == iters:
            res = evaluate(); lg('  [検証] 反復 %d: n-gram %.4f -> 混合 %.4f 利得 %.4f (対照 %.4f; 上乗せ %.4f) | 初出文脈 %.4f -> %.4f' % (it + 1, res['ngram'], res['mixed'], res['gain'], ctrl['gain'], res['extra_over_control'], res['ngram_unseen'], res['mixed_unseen']))
    if cfg.get('save'): np.savez(cfg['save'], Pk=Pk, Bt=Bt, logpi=logpi, ids=ids)
    lg('RESULT_LAT ' + json.dumps(dict(cfg=cfg, res=res), ensure_ascii=False)); json.dump(dict(cfg=cfg, res=res), open('/tmp/pe_result_K%d_Z%d_n%d_%s.json' % (K, Z, ntr, 'full' if cfg.get('full') else 'sub'), 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
