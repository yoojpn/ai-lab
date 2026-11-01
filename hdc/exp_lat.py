"""刷新の実験 5: 潜在クラスの確率モデルを EM 法 (期待回数を数えて割る; 勾配なし) で最尤学習する。
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
    alpha = cfg.get('alpha', 0.1); beta = cfg.get('beta', 0.01)
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = nidx // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); N = len(S)
    lam = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model({n: '%s/final_o%d' % (D, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    freq = np.bincount(S.astype(np.int64), minlength=V); freq[:3] = 0; ids = np.argsort(-freq, kind='stable')[:Vc]; ids = ids[freq[ids] > 0]; Vc = len(ids)
    rank = np.full(V + 1, Vc, np.int64); rank[ids] = np.arange(Vc); r = rank[S.astype(np.int64)]        # 圏外は Vc
    pos = rng.integers(8 + K, N, int(ntr * 1.2)); pos = pos[r[pos] < Vc][:ntr]; n = len(pos)
    X = np.stack([r[pos - k] for k in range(1, K + 1)], axis=1); Y = r[pos]; lg('学習行 %d, 語彙 %d, K=%d, Z=%d' % (n, Vc, K, Z))
    A = rng.random((K, Vc + 1, Z)).astype(np.float32) + 1.0; A /= A.sum(2, keepdims=True)                       # P(z | 語) ; 行の和 = 1
    Bt = rng.random((Vc, Z)).astype(np.float32) + 1.0; Bt /= Bt.sum(0, keepdims=True)                         # P(語 | z) ; 列の和 = 1
    pi = np.full(K, 1.0 / K, np.float32); CH = 60000
    for it in range(iters):
        Aacc = np.zeros_like(A); Bacc = np.zeros((Vc, Z), np.float32); pacc = np.zeros(K); ll = 0.0
        for a in range(0, n, CH):
            xs = X[a:a + CH]; ys = Y[a:a + CH]; m_ = len(ys); Bc = Bt[ys]
            R = [pi[k] * A[k][xs[:, k]] * Bc for k in range(K)]; tot = sum(x.sum(1) for x in R) + 1e-30; ll += float(np.log2(tot).sum())
            ar = np.arange(m_); Rs = np.zeros((m_, Z), np.float32)
            for k in range(K):
                Rk = R[k] / tot[:, None]; pacc[k] += float(Rk.sum()); Rs += Rk
                Aacc[k] += sparse.csr_matrix((np.ones(m_, np.float32), (xs[:, k], ar)), shape=(Vc + 1, m_)) @ Rk
            Bacc += sparse.csr_matrix((np.ones(m_, np.float32), (ys, ar)), shape=(Vc, m_)) @ Rs
        A = Aacc + alpha / Z; A /= A.sum(2, keepdims=True); Bt = Bacc + beta; Bt /= Bt.sum(0, keepdims=True); pi = (pacc / pacc.sum()).astype(np.float32)
        lg('反復 %d: 学習の対数尤度 %.4f bit/行 (語彙内の語のみ), pi=%s' % (it + 1, -ll / n, np.round(pi, 3).tolist()))
    # 評価: n-gram との線形補間
    def evalset(sp_, nn):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16); p = rng.integers(16 + K, len(T), int(nn * 1.02)); p = p[T[p] != 0][:nn]; return T, p
    def plat(T, p):
        rr = rank[T.astype(np.int64)]; yy = rr[p]; out = np.zeros(len(p))
        for a in range(0, len(p), 20000):
            sl = slice(a, a + 20000); pp = p[sl]; y_ = yy[sl]; ok = y_ < Vc; acc = np.zeros(len(pp), np.float64)
            for k in range(K): acc += pi[k] * (A[k][rr[pp - (k + 1)]] * Bt[np.minimum(y_, Vc - 1)]).sum(1)
            out[sl] = np.where(ok, acc, 0.0)
        return out
    res = {}
    parts = {}; evalset_cache = {}
    for sp_, nn in (('val', cfg['nval']), ('test', cfg['ntest'])):
        T, p = evalset(sp_, nn); evalset_cache[sp_] = (T, p); lp, hi = M.logprob_at(T, p, return_hi=True); parts[sp_] = (2.0 ** lp, plat(T, p), hi)
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
    res = mix_eval(parts['val'][1], parts['test'][1]); lt = parts['test'][1]
    res['frac_in_vocab'] = float((lt > 0).mean())
    pu = np.bincount(Y, minlength=Vc).astype(np.float64) / n
    def uni(sp_):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16); return T
    ctrl = {}
    for sp_ in ('val', 'test'):
        T = uni(sp_); ppos = None
    # 対照 (頻度だけ): 同じ位置で P(w)=頻度
    ctrl_parts = {}
    for sp_ in ('val', 'test'):
        T, p = evalset_cache[sp_]; yy = rank[T.astype(np.int64)][p]; ctrl_parts[sp_] = np.where(yy < Vc, pu[np.minimum(yy, Vc - 1)], 0.0)
    res['control_unigram'] = mix_eval(ctrl_parts['val'], ctrl_parts['test'])
    lg('RESULT_LAT ' + json.dumps(dict(cfg=cfg, res=res), ensure_ascii=False)); json.dump(dict(cfg=cfg, res=res), open('/tmp/lat_result_K%d_Z%d.json' % (K, Z), 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
