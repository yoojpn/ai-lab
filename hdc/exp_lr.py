"""刷新の実験 2: 文脈の表を 低ランク分解 (閉形式, 勾配なし) して 「教えていない組み合わせ」にも効かせる。
  行 = (何語前か k, その語 w) の特徴 K x Vc 個, 列 = 次の語 (Vc 語), 値 = 共起回数 -> PPMI -> 乱択 SVD (ランク R)
  文脈の座標 h = sum_k g^(k-1) A[(k, 語)],   スコア s(w) = h . B[w]   (A=行の座標, B=列の座標)
  予測: P(w) ∝ P_ngram(w) * exp(beta * s(w))  (beta, g は 検証で格子探索)   n-gram は 同じデータで作る (1..5)
使い方: python3 exp_lr.py '{"nidx":10000000,"R":128,"K":12,"Vc":20000,"ntest":20000}'"""
import sys, os, json, time, shutil
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
import engine as E, chat

V = 32768


def main(cfg):
    t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); W = '/tmp/tok'; D = '/tmp/lr'; os.makedirs(D, exist_ok=True)
    nidx = cfg['nidx']; R = cfg['R']; K = cfg['K']; Vc = cfg['Vc']; rng = np.random.default_rng(0)
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = nidx // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); N = len(S)
    if not os.path.exists('%s/final_o5_X.bin' % D) or cfg.get('rebuild'):
        S.tofile(D + '/sub.bin'); runs = E.spill(D + '/sub.bin', N - 8, D + '/runs', [1, 2, 3, 4, 5], 8_000_000, lambda s: None); MINC = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
        for n in (1, 2, 3, 4, 5): E.merge(runs[n], D + '/final_o%d' % n, MINC[n], 20_000_000, lambda s: None)
        shutil.rmtree(D + '/runs', ignore_errors=True)
    lam = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model({n: '%s/final_o%d' % (D, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True); lg('n-gram 済み (%d 語)' % (N - 8))
    freq = np.bincount(S.astype(np.int64), minlength=V); freq[:3] = 0; ids = np.argsort(-freq, kind='stable')[:Vc]; ids = ids[freq[ids] > 0]; Vc = len(ids)
    rank = np.full(V, -1, np.int64); rank[ids] = np.arange(Vc); r = rank[S.astype(np.int64)]
    C = None; CH = 1_500_000
    for a in range(8 + K, N, CH):
        idx = np.arange(a, min(N, a + CH)); y = r[idx]; okt = y >= 0; rows = []; cols = []
        for k in range(1, K + 1):
            x = r[idx - k]; m = okt & (x >= 0); rows.append((k - 1) * Vc + x[m]); cols.append(y[m])
        rows = np.concatenate(rows); cols = np.concatenate(cols); c = sparse.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(K * Vc, Vc)).tocsr(); C = c if C is None else C + c
    lg('共起表 %d x %d, 非零 %d' % (C.shape[0], C.shape[1], C.nnz))
    tot = C.sum(); rs = np.asarray(C.sum(1)).ravel() + 1e-9; cs = np.asarray(C.sum(0)).ravel() ** 0.75; cs = cs / cs.sum() * tot + 1e-9
    Cc = C.tocoo(); v = np.log(Cc.data * tot / (rs[Cc.row] * cs[Cc.col])); keep = v > 0; Pm = sparse.csr_matrix((v[keep].astype(np.float32), (Cc.row[keep], Cc.col[keep])), shape=C.shape); del C, Cc
    U, Sg, VT = randomized_svd(Pm, n_components=R, n_iter=3, random_state=0); A = (U * np.sqrt(Sg)).astype(np.float32); B = (VT.T * np.sqrt(Sg)).astype(np.float32); lg('SVD ランク %d 済み' % R)
    def evalset(sp, n):
        T = np.fromfile('%s/%s.bin' % (W, sp), dtype=np.uint16); p = rng.integers(16 + K, len(T), int(n * 1.02)); p = p[T[p] != 0][:n]; y = T[p].astype(np.int64); return T, p, y
    def scores(T, p, g):
        h = np.zeros((len(p), R), np.float32); w = 1.0
        for k in range(1, K + 1):
            x = rank[T[p - k].astype(np.int64)]; m = x >= 0; h[m] += w * A[(k - 1) * Vc + x[m]]; w *= g
        return h @ B.T                                                                  # (n, Vc)
    sets = {sp: evalset(sp, cfg['ntest']) for sp in ('val', 'test')}
    P0 = {sp: np.stack([chat.next_dist(M, sets[sp][0][:q]).astype(np.float16) for q in sets[sp][1]]) for sp in sets}; lg('n-gram の全語彙分布 済み')
    ids_j = ids
    def bits(sp, beta, g):
        T, p, y = sets[sp]; tot = 0.0; CHK = 1000
        for a in range(0, len(p), CHK):
            sl = slice(a, min(len(p), a + CHK)); s = scores(T, p[sl], g); Pn = P0[sp][sl].astype(np.float32); adj = np.zeros_like(Pn); adj[:, ids_j] = beta * s; adj -= adj.max(1, keepdims=True)
            Q = Pn * np.exp(adj); Q /= Q.sum(1, keepdims=True); tot += float(-np.log2(Q[np.arange(Q.shape[0]), y[sl]] + 1e-12).sum())
        return tot / len(y)
    base = {sp: bits(sp, 0.0, 0.8) for sp in sets}; best = None
    for g in (0.7, 0.85, 1.0):
        for beta in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            v_ = bits('val', beta, g)
            if best is None or v_ < best[0]: best = (v_, beta, g)
    out = dict(nidx=nidx, R=R, K=K, Vc=Vc, ngram_val=base['val'], ngram_test=base['test'], best_val=best[0], beta=best[1], g=best[2], mix_test=bits('test', best[1], best[2]))
    lg('RESULT_LR ' + json.dumps(out)); json.dump(out, open('/tmp/lr_result_%d_R%d.json' % (nidx, R), 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
