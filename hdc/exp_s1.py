"""S1: n-gram と 網を 一緒に学習する (網は n-gram が外すところを学ぶ)。 漏れのない評価つき。
  学習データ: val.bin の 前半 (n-gram の表は train だけで作ってあるので、 val の語は n-gram が見ていない = 答えを知らない状態で予測する)
  dev (重みの当てはめ用): val.bin の 最後 ndev 語。 テスト: test.bin。
  joint=1: 損失 = -log( (1-g[hi]) * p_ngram(y) + g[hi] * p_net(y) )   g[hi] は 学習する重み、 hi = n-gram が見つけた最高次数 (予測の前に分かる)
  joint=0: 損失 = -log p_net(y)  (網だけで学習)。 あとから dev で hi ごとの重みを当てはめて混ぜる。
  どちらも テストでは 重みを hi だけで決める (正解の語の確率は使わない)。
使い方: python3 exp_s1.py '{"L":4,"feats":[1,2,3],"ntr":3000000,"joint":1}'
"""
import sys, json, time, os, math
import numpy as np
import torch
import torch.nn.functional as F
import engine as E
import exp_ctx as X

V = X.V


def main(cfg):
    t0 = time.time(); lg = lambda m: print('[%6ds] %s' % (time.time() - t0, m), flush=True)
    X.DEV = torch.device(cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')); DEV = X.DEV
    seed = cfg.get('seed', 0); torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X.MC = cfg.get('mc', X.MC); X.W2 = cfg.get('w2', X.W2)
    W = '/tmp/tok'; d = cfg.get('d', 128); c = cfg.get('c', 8); L = cfg['L']; n = cfg.get('n', 256); kk = cfg.get('kk', 4); k = cfg.get('k', 4); B = cfg.get('B', 512)
    feats = list(cfg.get('feats', [1, 2, 3])); ntr = cfg['ntr']; lr = cfg.get('lr', 2e-3); lrs = cfg.get('lrs', 5e-3); joint = int(cfg.get('joint', 1)); ndev = cfg.get('ndev', 600000)
    offs8 = np.arange(1, c + 1); o1 = [1 + (np.arange(X.NS) * w) // X.NS for w in X.W1]; offs2 = np.arange(c + 1, X.W2 + 1)
    lp_ = W + '/lam.json'; lam = {int(a): b for a, b in json.load(open(lp_)).items()} if os.path.exists(lp_) else {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}
    M = E.Model({o: '%s/final_o%d' % (W, o) for o in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    Tv = np.fromfile(W + '/val.bin', dtype=np.uint16); Dv = np.load(W + '/val_docs.npy'); VT = len(Tv) - ndev
    lg('事前確認: n-gram 表は使えます。 joint=%d feats=%s 学習に使う val は %d 語 (dev は最後の %d 語)' % (joint, feats, VT, ndev))
    nb = 40; bl = ntr // nb; starts = np.linspace(8, VT - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(c, np.uint16)] + [np.asarray(Tv[s:s + bl]) for s in starts]).astype(np.int64); N = len(S)
    u = np.arange(N); ci = np.clip((u - c) // bl, 0, nb - 1); loc = u - c - ci * bl; ptr = starts[ci] + loc
    dsT = np.maximum(Dv[np.maximum(np.searchsorted(Dv, ptr, 'right') - 1, 0)], starts[ci]); DS = c + ci * bl + (dsT - starts[ci]); DS[:c] = u[:c]; del u, ci, loc, dsT
    cnt = np.bincount(S, minlength=V).astype(np.float64); pu = (cnt + 1) / (cnt.sum() + V); rar = -np.log2(pu); rar[0] = 0.0; rar[1] = 0.0
    pos = np.arange(c, N); pos = pos[S[pos] != 0]; rng.shuffle(pos); steps = len(pos) // B
    # 学習の各位置で n-gram が 正解の語につける確率と、 見つけた最高次数 (n-gram は train だけで作ってあるので val は未見)
    PL = np.zeros(len(pos)); HI = np.zeros(len(pos), np.int8); idx_all = ptr[pos]
    for a in range(0, len(pos), 400000):
        lp, hi = M.logprob_at(Tv, idx_all[a:a + 400000], return_hi=True); PL[a:a + 400000] = lp * math.log(2); HI[a:a + 400000] = hi
    lg('学習の各位置の n-gram 確率を計算: 平均 %.4f bit/語 (これが n-gram だけの学習データでの値)' % (-PL.mean() / math.log(2)))
    net = X.Net(d, c, L, n, kk, k, feats, rar).to(DEV); npar = sum(p.numel() for p in net.parameters())
    gl = torch.nn.Parameter(torch.full((6,), -0.8, device=DEV))
    lg('L=%d B=%d 学習語 %d 歩数 %d パラメータ %.1fM' % (L, B, len(pos), steps, npar / 1e6))
    opt_s = torch.optim.SparseAdam(net.sparse_params(), lr=lrs); opt_d = torch.optim.Adam(net.dense_params() + [gl], lr=lr)

    def nn_logp(T, p, ds, bs=1024):
        out = np.zeros(len(p)); net.eval()
        with torch.no_grad():
            for a in range(0, len(p), bs):
                pp = p[a:a + bs]; inp = X.make_inputs(T, pp, ds[a:a + bs], feats, offs8, o1, offs2); y = torch.from_numpy(T[pp].astype(np.int64)).to(DEV)
                out[a:a + bs] = (net.logp(inp, y) / math.log(2)).cpu().numpy()
        net.train(); return out
    ds_of = lambda D, p: D[np.maximum(np.searchsorted(D, p, 'right') - 1, 0)]
    Tt = np.fromfile(W + '/test.bin', dtype=np.uint16); Dt = np.load(W + '/test_docs.npy')
    nval = min(cfg.get('nval', 100000), ndev - 100); pv_ = rng.choice(np.arange(VT + 16, len(Tv)), size=nval, replace=False); pv_ = pv_[Tv[pv_] != 0]
    nt = cfg.get('ntest', 200000); nl = cfg.get('nlong', 40000)
    p = rng.integers(16 + c, len(Tt), int(nt * 1.02)); p = p[Tt[p] != 0][:nt]; q = rng.integers(16 + c, len(Tt), 4000000); q = q[(Tt[q] != 0) & ((q - ds_of(Dt, q)) >= 256)][:nl]
    pt_ = np.r_[p, q]; ntm = len(p)
    curve = []; run = 0.0; cnt_ = 0; ts = time.time(); le = cfg.get('log_every', 500)
    for st in range(steps):
        f = min(1.0, (st + 1) / 200) * max(0.1, 1 - st / steps)
        for g in opt_d.param_groups: g['lr'] = lr * f
        for g in opt_s.param_groups: g['lr'] = lrs * f
        sl = slice(st * B, (st + 1) * B); pp = pos[sl]; inp = X.make_inputs(S, pp, DS[pp], feats, offs8, o1, offs2); y = torch.from_numpy(S[pp]).to(DEV)
        lpn = net.logp(inp, y)
        if joint:
            pl = torch.from_numpy(PL[sl]).float().to(DEV); g_ = torch.sigmoid(gl)[torch.from_numpy(HI[sl].astype(np.int64)).to(DEV)]
            loss = -torch.logaddexp(torch.log1p(-g_) + pl, torch.log(g_) + lpn).mean()
        else:
            loss = -lpn.mean()
        run += loss.item() / math.log(2); cnt_ += 1
        opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); opt_s.step(); opt_d.step()
        if (st + 1) % le == 0:
            curve.append((st + 1, run / cnt_)); lg('歩 %d/%d: 学習の損失 %.4f bit/語 (直近) | %.2f 歩/秒' % (st + 1, steps, run / cnt_, le / (time.time() - ts))); run = 0.0; cnt_ = 0; ts = time.time()
    lg('ONLINE_S1 ' + json.dumps(dict(joint=joint, steps=steps, curve=curve), ensure_ascii=False))
    # ---- 評価 (漏れなし: 混合の重みは hi だけで決める) ----
    lpv, hv = M.logprob_at(Tv, pv_, return_hi=True); lpt, ht = M.logprob_at(Tt, pt_, return_hi=True)
    ngv, ngt = 2.0 ** lpv, 2.0 ** lpt; hv, ht = hv.astype(int), ht.astype(int)
    nnv = 2.0 ** nn_logp(Tv.astype(np.int64), pv_, ds_of(Dv, pv_)); nnt = 2.0 ** nn_logp(Tt.astype(np.int64), pt_, ds_of(Dt, pt_))
    G = np.r_[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]; bits = lambda P: float(-np.log2(P).mean())
    gph = np.zeros(6)
    for b_ in range(1, 6):
        mm = hv == b_
        if mm.sum() >= 200: gph[b_] = G[int(np.argmax([np.log2((1 - g) * ngv[mm] + g * nnv[mm]).mean() for g in G]))]
    mix = lambda g, hh: (1 - g[hh]) * ngt + g[hh] * nnt
    mt, ml = slice(0, ntm), slice(ntm, None); gj = torch.sigmoid(gl).detach().cpu().numpy()
    res = dict(joint=joint, params_M=npar / 1e6, steps=steps, ngram=bits(ngt[mt]), net_alone=bits(nnt[mt]), posthoc=bits(mix(gph, ht)[mt]), g_posthoc=gph.tolist(),
               joint_g=bits(mix(gj, ht)[mt]), g_joint=gj.tolist(), long_ngram=bits(ngt[ml]), long_net=bits(nnt[ml]), long_posthoc=bits(mix(gph, ht)[ml]), long_joint=bits(mix(gj, ht)[ml]), train_seconds=time.time() - t0)
    lg('結果(テスト, 漏れなし): n-gram %.4f | 網だけ %.4f | あとから混合(devで重み) %.4f | 学習した重みで混合 %.4f' % (res['ngram'], res['net_alone'], res['posthoc'], res['joint_g']))
    lg('長い文脈(256語以降): n-gram %.4f | 網だけ %.4f | あとから混合 %.4f | 学習した重みで混合 %.4f' % (res['long_ngram'], res['long_net'], res['long_posthoc'], res['long_joint']))
    lg('重み g[hi=1..5]: あとから %s | 学習 %s' % (np.round(gph[1:], 3).tolist(), np.round(gj[1:], 3).tolist()))
    lg('RESULT_S1 ' + json.dumps(dict(cfg=cfg, res=res), ensure_ascii=False))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
