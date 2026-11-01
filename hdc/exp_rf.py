"""刷新の実験 3: 勾配なしで 非線形 (ランダムな特徴 + 閉形式の最小二乗法)。 exp_lr (線形, -0.016 bit) と同じ条件で比べる。
  語の座標 E: 共起の PPMI -> SVD (d 次元)。 文脈 x = 直前 K 語の座標を位置ごとに並べたもの。
  深さ 0: 線形 (x のまま)   深さ 1: relu(x R1)   深さ 2: relu(relu(x R1) R2)   (R は乱数で固定; 学習しない)
  出力: 特徴 Phi から 次の語 (上位 Vc 語) への 最小二乗法 (リッジ, 固有値分解で λ を検証で選ぶ)。 目標 = one-hot - 単語の頻度。
  予測: P(w) ∝ P_ngram(w) * exp(beta * s(w))  (exp_lr と同じ; n-gram は /tmp/lr の 1000 万語 5-gram)
使い方: python3 exp_rf.py '{"nidx":10000000,"ntr":1500000,"m":4096,"depth":1,"ntest":10000}'
"""
import sys, os, json, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import engine as E, chat

V = 32768


def main(cfg):
    t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); W = '/tmp/tok'; D = '/tmp/lr'
    nidx = cfg['nidx']; K = cfg.get('K', 8); d = cfg.get('d', 64); Vc = cfg.get('Vc', 20000); m = cfg['m']; depth = cfg['depth']; ntr = cfg['ntr']; rng = np.random.default_rng(0)
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = nidx // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); N = len(S)
    lam = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model({n: '%s/final_o%d' % (D, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    freq = np.bincount(S.astype(np.int64), minlength=V); freq[:3] = 0; ids = np.argsort(-freq, kind='stable')[:Vc]; ids = ids[freq[ids] > 0]; Vc = len(ids)
    rank = np.full(V, -1, np.int64); rank[ids] = np.arange(Vc); r = rank[S.astype(np.int64)]
    # 語の座標 (PPMI -> SVD)
    rows = []; cols = []
    for dd in (1, 2):
        a = r[:-dd]; b = r[dd:]; ok = (a >= 0) & (b >= 0); rows += [a[ok], b[ok]]; cols += [b[ok], a[ok]]
    rows = np.concatenate(rows); cols = np.concatenate(cols); Cm = sparse.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(Vc, Vc)).tocsr()
    tot = Cm.sum(); rs = np.asarray(Cm.sum(1)).ravel() + 1e-9; cs = rs ** 0.75; cs = cs / cs.sum() * tot; Cc = Cm.tocoo(); v = np.log(Cc.data * tot / (rs[Cc.row] * cs[Cc.col])); keep = v > 0
    Pm = sparse.csr_matrix((v[keep], (Cc.row[keep], Cc.col[keep])), shape=(Vc, Vc)); U, Sg, _ = svds(Pm, k=d, random_state=0); Emb = (U * np.sqrt(Sg)).astype(np.float32); Emb /= (Emb.std() + 1e-9)
    Emb = np.vstack([Emb, np.zeros((1, d), np.float32)]); lg('語の座標 %d x %d' % (Vc, d))
    X0 = K * d; rr = np.random.default_rng(1)
    Rs = [(rr.standard_normal((X0 if i == 0 else m, m)).astype(np.float32) / np.sqrt(X0 if i == 0 else m)) * np.sqrt(2.0) for i in range(depth)]
    bs = [rr.standard_normal(m).astype(np.float32) * 0.1 for _ in range(depth)]
    def phi(T, p):
        x = np.concatenate([Emb[np.where(rank[T[p - k].astype(np.int64)] >= 0, rank[T[p - k].astype(np.int64)], Vc)] for k in range(1, K + 1)], axis=1)
        h = x
        for i in range(depth): h = np.maximum(h @ Rs[i] + bs[i], 0.0)
        return np.hstack([h, np.ones((len(h), 1), np.float32)])
    mm = (m if depth else X0) + 1
    # 学習: Phi^T Phi と Phi^T Y を 集める
    pos = rng.integers(8 + K, N, int(ntr * 1.05)); pos = pos[r[pos] >= 0][:ntr]; G = np.zeros((mm, mm), np.float64); B = np.zeros((mm, Vc), np.float32); sp = np.zeros(mm, np.float64); CH = 20000
    for a in range(0, len(pos), CH):
        p = pos[a:a + CH]; F = phi(S, p); G += (F.T @ F).astype(np.float64); sp += F.sum(0)
        Y = sparse.csr_matrix((np.ones(len(p), np.float32), (np.arange(len(p)), r[p])), shape=(len(p), Vc)); B += np.asarray((Y.T @ F).T)
        if (a // CH) % 10 == 0: lg('集計 %d / %d' % (a, len(pos)))
    py = np.bincount(r[pos], minlength=Vc).astype(np.float64) / len(pos); B = B - np.outer(sp, py).astype(np.float32)          # 目標 = one-hot - 頻度
    w, Q = np.linalg.eigh(G); QB = (Q.T @ B.astype(np.float64)).astype(np.float32); lg('固有値分解 済み (特徴 %d 次元)' % mm)
    scale = w.max()
    def weights(lm): return (Q @ (QB / (w + lm * scale)[:, None]).astype(np.float64)).astype(np.float32)
    def evalset(sp_, n):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16); p = rng.integers(16 + K, len(T), int(n * 1.02)); p = p[T[p] != 0][:n]; y = T[p].astype(np.int64); return T, p, y
    sets = {s_: evalset(s_, cfg['ntest']) for s_ in ('val', 'test')}
    P0 = {s_: np.stack([chat.next_dist(M, sets[s_][0][:q]).astype(np.float16) for q in sets[s_][1]]) for s_ in sets}; lg('n-gram の全語彙分布 済み')
    betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    def bits(s_, Wt):
        T, p, y = sets[s_]; tot = np.zeros(len(betas)); CHK = 1000
        for a in range(0, len(p), CHK):
            sl = slice(a, min(len(p), a + CHK)); sc = phi(T, p[sl]) @ Wt; Pn = P0[s_][sl].astype(np.float32)
            for j, be in enumerate(betas):
                adj = np.zeros_like(Pn); adj[:, ids] = be * sc; adj -= adj.max(1, keepdims=True); Q_ = Pn * np.exp(adj); Q_ /= Q_.sum(1, keepdims=True)
                tot[j] += float(-np.log2(Q_[np.arange(Q_.shape[0]), y[sl]] + 1e-12).sum())
        return tot / len(y)
    best = None; lms = cfg.get('lams', [1e-6, 1e-5, 1e-4, 1e-3])
    for lm in lms:
        Wt = weights(lm); bv = bits('val', Wt); j = int(np.argmin(bv)); lg('λ=%g: 検証 bit/token  beta別 %s  (beta=0 が n-gram のみ %.4f)' % (lm, np.round(bv, 4).tolist(), bv[0]))
        if best is None or bv[j] < best[0]: best = (float(bv[j]), lm, j, float(bv[0]))
    Wt = weights(best[1]); bt = bits('test', Wt); out = dict(cfg=cfg, ngram_val=best[3], best_val=best[0], lam=best[1], beta=betas[best[2]], ngram_test=float(bt[0]), mix_test=float(bt[best[2]]), gain_test=float(bt[0] - bt[best[2]]))
    lg('RESULT_RF ' + json.dumps(out)); json.dump(out, open('/tmp/rf_result_d%d_m%d.json' % (depth, m), 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
