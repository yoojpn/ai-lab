"""E2: 表引きだけを重ねた網で、層の深さが効くかを測る。
  状態 x (長さ d) の初期値 = 直前 c 語の 「語の行 × 位置ベクトル」 の和
  各層 = 積キーの記憶 (キー表 2 つ (n 行 x d/2) と 値表 (n*n 行 x d)) から 上位 k 行を引き、 近さの softmax で重みづけして x に足す
  出力 = 全語彙の内積 (測定用の足場。 規模を上げるときは 木 / 標本化に置き換える)
  学習 = データ 1 周。 先に予測して損失を記録し (progressive validation)、 そのあと更新。 値表と語の行は 疎な更新。
  評価 = 検証/テストの bit/語 (単独) と、 n-gram (3.8億語) との二段階混合の上乗せ (頻度で較正した n-gram の上)。
使い方: python3 exp_deep.py '{"L":2,"ntr":12000000}'
"""
import sys, json, time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import engine as E

V = 32768


class Net(nn.Module):
    def __init__(s, d, c, L, n, kk, k):
        super().__init__()
        s.d, s.c, s.L, s.n, s.kk, s.k = d, c, L, n, kk, k
        s.E = nn.Embedding(V, d, sparse=True); nn.init.normal_(s.E.weight, 0, 0.1)
        s.P = nn.Parameter(torch.randn(c, d) * 0.1 + 1.0)
        s.K1 = nn.ParameterList([nn.Parameter(torch.randn(n, d // 2) * 0.5) for _ in range(L)])
        s.K2 = nn.ParameterList([nn.Parameter(torch.randn(n, d // 2) * 0.5) for _ in range(L)])
        s.Val = nn.ModuleList([nn.Embedding(n * n, d, sparse=True) for _ in range(L)])
        for v in s.Val: nn.init.normal_(v.weight, 0, 0.02)
        s.tau = nn.Parameter(torch.ones(L))
        s.U = nn.Parameter(torch.randn(V, d) * 0.05)

    def sparse_params(s):
        return [s.E.weight] + [v.weight for v in s.Val]

    def dense_params(s):
        return [s.P, s.U, s.tau] + list(s.K1) + list(s.K2)

    @staticmethod
    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def hidden(s, ctx):
        x = (s.E(ctx) * s.P[None]).sum(1)
        n, kk, k, h = s.n, s.kk, s.k, s.d // 2
        for l in range(s.L):
            q = s.rms(x); q1, q2 = q[:, :h], q[:, h:]
            s1 = q1 @ s.K1[l].t() / math.sqrt(h); s2 = q2 @ s.K2[l].t() / math.sqrt(h)
            v1, i1 = s1.topk(kk, dim=1); v2, i2 = s2.topk(kk, dim=1)
            comb = (v1[:, :, None] + v2[:, None, :]).flatten(1)
            sc, ix = comb.topk(k, dim=1)
            a = F.softmax(sc / s.tau[l].abs().clamp_min(0.05), dim=1)
            ia = i1.gather(1, ix // kk); ib = i2.gather(1, ix % kk); rows = ia * n + ib
            x = x + (a[:, :, None] * s.Val[l](rows)).sum(1)
        return s.rms(x)

    def logits(s, ctx):
        return F.linear(s.hidden(ctx), s.U)


def main(cfg):
    t0 = time.time(); lg = lambda m: print('[%6ds] %s' % (time.time() - t0, m), flush=True)
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(cfg.get('seed', 0)); rng = np.random.default_rng(cfg.get('seed', 0))
    W = '/tmp/tok'; d = cfg.get('d', 128); c = cfg.get('c', 8); L = cfg['L']; n = cfg.get('n', 256); kk = cfg.get('kk', 4); k = cfg.get('k', 4); B = cfg.get('B', 512)
    ntr = cfg['ntr']; lr = cfg.get('lr', 2e-3); lrs = cfg.get('lrs', 5e-3)
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = ntr // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(c, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]).astype(np.int64); N = len(S)
    pos = np.arange(c, N); pos = pos[S[pos] != 0]; rng.shuffle(pos); steps = len(pos) // B
    # 評価に要るものを 学習の前に確かめる (学習の最後で落ちて 数時間が無駄になるのを防ぐ)
    lp_ = W + '/lam.json'; lam = {int(a): b for a, b in json.load(open(lp_)).items()} if os.path.exists(lp_) else {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}
    M = E.Model({o: '%s/final_o%d' % (W, o) for o in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    for sp_ in ('val', 'test'):
        assert os.path.exists('%s/%s.bin' % (W, sp_)), sp_
    _T = np.fromfile(W + '/val.bin', dtype=np.uint16); _p = np.arange(100, 200); M.logprob_at(_T, _p, return_hi=True); lg('事前確認: n-gram 表と lam=%s は使えます' % lam)
    net = Net(d, c, L, n, kk, k); npar = sum(p.numel() for p in net.parameters())
    lg('L=%d d=%d n=%d (行数/層 %d) kk=%d k=%d B=%d 学習語 %d 歩数 %d パラメータ %.1fM' % (L, d, n, n * n, kk, k, B, len(pos), steps, npar / 1e6))
    opt_s = torch.optim.SparseAdam(net.sparse_params(), lr=lrs); opt_d = torch.optim.Adam(net.dense_params(), lr=lr)
    offs = np.arange(1, c + 1)
    def ctx_of(T, p): return torch.from_numpy(T[p[:, None] - offs[None]].astype(np.int64))
    def nn_logp(T, p, bs=1024):
        out = np.zeros(len(p))
        net.eval()
        with torch.no_grad():
            for a in range(0, len(p), bs):
                pp = p[a:a + bs]; lo = F.log_softmax(net.logits(ctx_of(T, pp)), dim=1); y = torch.from_numpy(T[pp].astype(np.int64)); out[a:a + bs] = (lo.gather(1, y[:, None])[:, 0] / math.log(2)).numpy()
        net.train(); return out
    def evalset(sp_, nn_):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16); p = rng.integers(16 + c, len(T), int(nn_ * 1.02)); p = p[T[p] != 0][:nn_]; return T, p
    ev = {sp_: evalset(sp_, cfg['nval'] if sp_ == 'val' else cfg['ntest']) for sp_ in ('val', 'test')}
    curve = []; run = 0.0; cnt = 0; ts = time.time()
    for st in range(steps):
        f = min(1.0, (st + 1) / 200) * max(0.1, 1 - st / steps)
        for g in opt_d.param_groups: g['lr'] = lr * f
        for g in opt_s.param_groups: g['lr'] = lrs * f
        pp = pos[st * B:(st + 1) * B]; ctx = torch.from_numpy(S[pp[:, None] - offs[None]]); y = torch.from_numpy(S[pp])
        loss = F.cross_entropy(net.logits(ctx), y)                                  # 更新の前の損失 = 先に予測した損失
        run += loss.item() / math.log(2); cnt += 1
        opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); opt_s.step(); opt_d.step()
        if (st + 1) % cfg.get('log_every', 500) == 0:
            curve.append((st + 1, run / cnt)); lg('歩 %d/%d: 先に予測した損失 %.4f bit/語 (直近) | %.2f 歩/秒' % (st + 1, steps, run / cnt, cfg.get('log_every', 500) / (time.time() - ts))); run = 0.0; cnt = 0; ts = time.time()
        if cfg.get('eval_every') and (st + 1) % cfg['eval_every'] == 0:
            T, p = ev['val']; lg('  [検証 2万語] %.4f bit/語' % (-nn_logp(T, p[:20000]).mean()))
    lg('ONLINE_DEEP ' + json.dumps(dict(L=L, n=n, steps=steps, curve=curve, last2000=curve[-1][1] if curve else None, last10000=float(np.mean([v for _, v in curve[-5:]])) if curve else None), ensure_ascii=False))
    # ---- 最終評価 ----
    parts = {}
    for sp_ in ('val', 'test'):
        T, p = ev[sp_]; lp, hi = M.logprob_at(T, p, return_hi=True); parts[sp_] = dict(ng=2.0 ** lp, hi=hi, nn=2.0 ** nn_logp(T, p), T=T, p=p)
    pu = np.bincount(S[S != 0], minlength=V).astype(np.float64); pu = (pu + 1) / (pu.sum() + V)
    for sp_ in parts: parts[sp_]['uni'] = pu[parts[sp_]['T'][parts[sp_]['p']].astype(np.int64)]
    G = np.r_[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    bins = lambda pl: np.clip((-np.log2(np.maximum(pl, 1e-12)) // 2).astype(int), 0, 7)
    bits = lambda P, m=None: float(-np.log2(P if m is None else P[m]).mean())
    def fit_g(pv, lv):
        bv = bins(pv); gb = np.zeros(8)
        for b_ in range(8):
            mm = bv == b_
            if mm.sum() < 200: continue
            gb[b_] = G[int(np.argmax([np.log2((1 - g) * pv[mm] + g * lv[mm]).mean() for g in G]))]
        return gb
    pv0, pt0 = parts['val']['ng'], parts['test']['ng']
    g1 = fit_g(pv0, parts['val']['uni']); pv1 = (1 - g1[bins(pv0)]) * pv0 + g1[bins(pv0)] * parts['val']['uni']; pt1 = (1 - g1[bins(pt0)]) * pt0 + g1[bins(pt0)] * parts['test']['uni']
    g2 = fit_g(pv1, parts['val']['nn']); pt2 = (1 - g2[bins(pt1)]) * pt1 + g2[bins(pt1)] * parts['test']['nn']; un = parts['test']['hi'] <= 2
    res = dict(L=L, params_M=npar / 1e6, steps=steps, online_curve=curve, test_nn_bits=bits(parts['test']['nn']), test_ngram_bits=bits(pt0), test_base_bits=bits(pt1), test_final_bits=bits(pt2), extra2=bits(pt1) - bits(pt2),
               unseen_frac=float(un.mean()), unseen_nn_bits=bits(parts['test']['nn'], un), unseen_base_bits=bits(pt1, un), unseen_final_bits=bits(pt2, un), g2=g2.tolist(), train_seconds=time.time() - t0)
    lg('結果: 単独 %.4f bit/語 | n-gram %.4f -> 頻度較正 %.4f -> 混合 %.4f (上乗せ %.4f) | 初出文脈(%.0f%%): 単独 %.4f, 基準 %.4f -> %.4f' % (res['test_nn_bits'], res['test_ngram_bits'], res['test_base_bits'], res['test_final_bits'], res['extra2'], 100 * res['unseen_frac'], res['unseen_nn_bits'], res['unseen_base_bits'], res['unseen_final_bits']))
    lg('RESULT_DEEP ' + json.dumps(dict(cfg=cfg, res=res), ensure_ascii=False))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
