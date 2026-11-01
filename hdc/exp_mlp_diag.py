"""診断: 同じ入力で MLP を勾配学習したら R2 と利得は どうなるか (モデルには使わない)。 元: 刷新の実験 4: 層ごとに 「次の語」 を当てる 閉形式の最小二乗法を重ねる (Deep Stacking Network 型; 勾配なし)。
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
    nidx = cfg['nidx']; K = cfg.get('K', 8); d = cfg.get('d', 64); Vc = cfg.get('Vc', 20000); m = cfg['m']; ntr = cfg['ntr']; nho = cfg.get('nho', 50000); rng = np.random.default_rng(0)
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
    X0 = K * d
    import torch
    torch.set_num_threads(2); torch.manual_seed(0)

    def ctx(T, p):
        return np.concatenate([Emb[np.where(rank[T[p - k].astype(np.int64)] >= 0, rank[T[p - k].astype(np.int64)], Vc)] for k in range(1, K + 1)], axis=1)
    allpos = rng.integers(8 + K, N, int((ntr + nho) * 1.05)); allpos = allpos[r[allpos] >= 0]; pos = allpos[:ntr]; hpos = allpos[ntr:ntr + nho]
    def evalset(sp_, n):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16); p = rng.integers(16 + K, len(T), int(n * 1.02)); p = p[T[p] != 0][:n]; y = T[p].astype(np.int64); return T, p, y
    sets = {s_: evalset(s_, cfg['ntest']) for s_ in ('val', 'test')}
    P0 = {s_: np.stack([chat.next_dist(M, sets[s_][0][:q]).astype(np.float16) for q in sets[s_][1]]) for s_ in sets}; lg('n-gram の全語彙分布 済み')
    betas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]; EbT = torch.from_numpy(Eb.T.copy())
    Xtr = torch.from_numpy(np.vstack([ctx(S, pos[a:a + 50000]) for a in range(0, len(pos), 50000)])); Ytr = torch.from_numpy(Emb[r[pos]])
    Xh = torch.from_numpy(ctx(S, hpos)); Yh = torch.from_numpy(Emb[r[hpos]]); var = float(((Yh - Yh.mean(0)) ** 2).mean())
    def bits(s_, net):
        T, p, y = sets[s_]; tot = np.zeros(len(betas)); CHK = 1000
        for a in range(0, len(p), CHK):
            sl = slice(a, min(len(p), a + CHK))
            with torch.no_grad(): sc = (net(torch.from_numpy(ctx(T, p[sl]))) @ EbT).numpy()
            Pn = P0[s_][sl].astype(np.float32)
            for j, be in enumerate(betas):
                adj = np.zeros_like(Pn); adj[:, ids] = be * sc; adj -= adj.max(1, keepdims=True); Q_ = Pn * np.exp(adj); Q_ /= Q_.sum(1, keepdims=True)
                tot[j] += float(-np.log2(Q_[np.arange(Q_.shape[0]), y[sl]] + 1e-12).sum())
        return tot / len(y)
    results = []
    for name, hid in (('MLP 隠れ1層', [m]), ('MLP 隠れ2層', [m, m])):
        dims = [X0] + hid; layers = []
        for i in range(len(hid)): layers += [torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.ReLU()]
        net = torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], d)); nparam = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); EP = cfg.get('epochs', 3); BS = 512; steps = EP * (len(Xtr) // BS); st_ = 0
        for ep in range(EP):
            perm = torch.randperm(len(Xtr))
            for a in range(0, len(Xtr) - BS + 1, BS):
                for g in opt.param_groups: g['lr'] = 1e-3 * (1 - st_ / steps) + 1e-5
                ii = perm[a:a + BS]; loss = ((net(Xtr[ii]) - Ytr[ii]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); st_ += 1
            with torch.no_grad(): mse = float(((net(Xh) - Yh) ** 2).mean())
            lg('%s: epoch %d 取り分け行 R2 %.4f' % (name, ep + 1, 1 - mse / var))
        bv = bits('val', net); j = int(np.argmin(bv)); bt = bits('test', net)
        res = dict(name=name, params=nparam, r2=1 - mse / var, ngram_test=float(bt[0]), mix_test=float(bt[j]), gain_test=float(bt[0] - bt[j]), beta=betas[j]); results.append(res)
        lg('%s (パラメータ %d): 検証 beta別 %s | テスト 利得 %.4f bit (n-gram %.4f -> %.4f)' % (name, nparam, np.round(bv, 4).tolist(), res['gain_test'], res['ngram_test'], res['mix_test']))
    lg('RESULT_MLP ' + json.dumps(results, ensure_ascii=False)); json.dump(dict(cfg=cfg, results=results), open('/tmp/mlp_diag.json', 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
