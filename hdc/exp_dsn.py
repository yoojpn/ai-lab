"""刷新の実験 4: 層ごとに 「次の語」 を当てる 閉形式の最小二乗法を重ねる (Deep Stacking Network 型; 勾配なし)。
  層 l の入力 = [直前 K 語の座標 x, 前の層の予測 z_{l-1} (次の語の座標)]  -> ランダム relu 展開 h -> 特徴 Phi=[h, x, 1] -> リッジで 次の語の座標 (d 次元) を予測 = z_l
  λ は 学習から取り分けた行の二乗誤差で決める。 層ごとに n-gram への混合の利得 (exp_lr と同じ形) を測る。
使い方: python3 exp_dsn.py '{"nidx":10000000,"ntr":1500000,"m":4096,"layers":3,"ntest":8000}'
"""
import sys, json, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import engine as E, chat

V = 32768


def main(cfg):
    t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); W = '/tmp/tok'; D = '/tmp/lr'
    nidx = cfg['nidx']; K = cfg.get('K', 8); d = cfg.get('d', 64); Vc = cfg.get('Vc', 20000); m = cfg['m']; NL = cfg['layers']; ntr = cfg['ntr']; nho = cfg.get('nho', 50000); rng = np.random.default_rng(0)
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = nidx // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); N = len(S)
    lam = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model({n: '%s/final_o%d' % (D, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    freq = np.bincount(S.astype(np.int64), minlength=V); freq[:3] = 0; ids = np.argsort(-freq, kind='stable')[:Vc]; ids = ids[freq[ids] > 0]; Vc = len(ids)
    rank = np.full(V, -1, np.int64); rank[ids] = np.arange(Vc); r = rank[S.astype(np.int64)]
    rows = []; cols = []
    for dd in (1, 2):
        a = r[:-dd]; b = r[dd:]; ok = (a >= 0) & (b >= 0); rows += [a[ok], b[ok]]; cols += [b[ok], a[ok]]
    rows = np.concatenate(rows); cols = np.concatenate(cols); Cm = sparse.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(Vc, Vc)).tocsr()
    tot = Cm.sum(); rs = np.asarray(Cm.sum(1)).ravel() + 1e-9; cs = rs ** 0.75; cs = cs / cs.sum() * tot; Cc = Cm.tocoo(); v = np.log(Cc.data * tot / (rs[Cc.row] * cs[Cc.col])); keep = v > 0
    Pm = sparse.csr_matrix((v[keep], (Cc.row[keep], Cc.col[keep])), shape=(Vc, Vc)); U, Sg, _ = svds(Pm, k=d, random_state=0); Eb = (U * np.sqrt(Sg)).astype(np.float32); Eb /= (Eb.std() + 1e-9)
    Emb = np.vstack([Eb, np.zeros((1, d), np.float32)]); lg('語の座標 %d x %d' % (Vc, d))
    X0 = K * d; rr = np.random.default_rng(1); layers = []

    def ctx(T, p):
        return np.concatenate([Emb[np.where(rank[T[p - k].astype(np.int64)] >= 0, rank[T[p - k].astype(np.int64)], Vc)] for k in range(1, K + 1)], axis=1)

    def run(T, p, upto):
        """upto 層ぶん通して (Phi, z) を返す; Phi は 層 upto の特徴 (その層の W は まだ無くてもよい)"""
        x = ctx(T, p); z = None
        for i in range(upto + 1):
            L = layers[i] if i < len(layers) else cur
            inp = x if z is None else np.hstack([x, z])
            h = np.maximum(inp @ L['R'] + L['b'], 0.0); Phi = np.hstack([h, x, np.ones((len(x), 1), np.float32)])
            if i == upto: return Phi
            z = (Phi @ L['W']) / L['zs']
    ids_ok = lambda p: r[p] >= 0
    allpos = rng.integers(8 + K, N, int((ntr + nho) * 1.05)); allpos = allpos[r[allpos] >= 0]; pos = allpos[:ntr]; hpos = allpos[ntr:ntr + nho]
    def evalset(sp_, n):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16); p = rng.integers(16 + K, len(T), int(n * 1.02)); p = p[T[p] != 0][:n]; y = T[p].astype(np.int64); return T, p, y
    sets = {s_: evalset(s_, cfg['ntest']) for s_ in ('val', 'test')}
    P0 = {s_: np.stack([chat.next_dist(M, sets[s_][0][:q]).astype(np.float16) for q in sets[s_][1]]) for s_ in sets}; lg('n-gram の全語彙分布 済み')
    betas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    def bits(s_, Wt, upto):
        T, p, y = sets[s_]; tot = np.zeros(len(betas)); CHK = 1000
        for a in range(0, len(p), CHK):
            sl = slice(a, min(len(p), a + CHK)); sc = run(T, p[sl], upto) @ Wt; Pn = P0[s_][sl].astype(np.float32)
            for j, be in enumerate(betas):
                adj = np.zeros_like(Pn); adj[:, ids] = be * sc; adj -= adj.max(1, keepdims=True); Q_ = Pn * np.exp(adj); Q_ /= Q_.sum(1, keepdims=True)
                tot[j] += float(-np.log2(Q_[np.arange(Q_.shape[0]), y[sl]] + 1e-12).sum())
        return tot / len(y)
    results = []; Tt = lambda p: Emb[r[p]]
    for li in range(NL):
        in_dim = X0 + (d if li else 0); R = (rr.standard_normal((in_dim, m)).astype(np.float32) / np.sqrt(in_dim)) * np.sqrt(2.0); b = rr.standard_normal(m).astype(np.float32) * 0.1
        cur = dict(R=R, b=b, W=None, zs=1.0); mm = m + X0 + 1
        G = np.zeros((mm, mm), np.float64); Bm = np.zeros((mm, d), np.float64); CH = 20000
        for a in range(0, len(pos), CH):
            p = pos[a:a + CH]; F = run(p=p, T=S, upto=li); G += (F.T @ F).astype(np.float64); Bm += (F.T @ Tt(p)).astype(np.float64)
        lg('層 %d: 集計 済み' % li)
        w, Q = np.linalg.eigh(G); QB = Q.T @ Bm; scale = w.max(); Fh = run(S, hpos, li); Th = Tt(hpos); best = None
        for lm in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            Wl = (Q @ (QB / (w + lm * scale)[:, None])).astype(np.float32); mse = float(((Fh @ Wl - Th) ** 2).mean())
            if best is None or mse < best[0]: best = (mse, lm, Wl)
        var = float(((Th - Th.mean(0)) ** 2).mean()); cur['W'] = best[2]
        zz = Fh @ best[2]; cur['zs'] = float(zz.std() + 1e-9); lg('層 %d: λ=%g 取り分け行の二乗誤差 %.4f (分散 %.4f; R2 %.4f)' % (li, best[1], best[0], var, 1 - best[0] / var))
        layers.append(cur)
        Wt = (cur['W'] @ Eb.T).astype(np.float32); bv = bits('val', Wt, li); j = int(np.argmin(bv)); bt = bits('test', Wt, li)
        res = dict(layer=li, lam=best[1], r2=1 - best[0] / var, ngram_val=float(bv[0]), best_val=float(bv[j]), beta=betas[j], ngram_test=float(bt[0]), mix_test=float(bt[j]), gain_test=float(bt[0] - bt[j])); results.append(res)
        lg('層 %d: 検証 beta別 %s | テスト 利得 %.4f bit (n-gram %.4f -> %.4f)' % (li, np.round(bv, 4).tolist(), res['gain_test'], res['ngram_test'], res['mix_test']))
    lg('RESULT_DSN ' + json.dumps(results)); json.dump(dict(cfg=cfg, results=results), open('/tmp/dsn_result.json', 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
