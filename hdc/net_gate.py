"""案B: 難しい位置にだけ網を使う。 n-gram が見つけた最高次数 hi ごとに 損失を分け、 網を使う位置を hi <= h0 に絞ったときの bit と 網を動かす位置の割合を測る。 CPU。
使い方: python3 net_gate.py 重み.pt TOK_DIR [N_VAL]   (TOK_DIR に val.bin と final_o1..5 の表)"""
import os, sys, time, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_ctx as X, chat_net, engine as E
from offline_path import ds_all

W = sys.argv[1]; TOK = sys.argv[2]; NV = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
net, cfg = chat_net.load(W); dev = torch.device('cpu'); X.DEV = dev; net = net.to(dev); net.eval(); net.cap = False; torch.set_num_threads(os.cpu_count())
c = cfg.get('c', 8); feats = list(cfg.get('feats', [])); offs8 = torch.arange(1, c + 1); o1 = [(1 + (torch.arange(X.NS) * w) // X.NS) for w in X.W1]; offs2 = torch.arange(c + 1, X.W2 + 1)
Tv = np.fromfile(TOK + '/val.bin', dtype=np.uint16); rng = np.random.default_rng(777); p = rng.integers(16 + c, len(Tv), int(NV * 1.02)); p = p[Tv[p] != 0][:NV]
Tg = torch.from_numpy(Tv.astype(np.int64)); dsg = torch.from_numpy(ds_all(Tv.astype(np.int64))); lpn = np.zeros(len(p)); t1 = time.time()
with torch.no_grad():
    for a in range(0, len(p), 512):
        pp = torch.from_numpy(p[a:a + 512]); inp = X.make_inputs_t(Tg, pp, dsg[pp], feats, offs8, o1, offs2); lpn[a:a + 512] = (net.logp(inp, Tg[pp]) / math.log(2)).numpy()
        if (a // 512) % 5 == 0: lg('  網 %d/%d 位置 (%.0f 位置/秒)' % (a + 512, len(p), (a + 512) / (time.time() - t1)))
Pn = 2.0 ** lpn; lam0 = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model({n: '%s/final_o%d' % (TOK, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam0, kn_base=True)
lp, hi = M.logprob_at(Tv, p, return_hi=True); Pg = 2.0 ** lp; hi = hi.astype(int); bits = lambda P: float(-np.log2(np.maximum(P, 1e-12)).mean()); h = len(p) // 2
lg('NETGATE 全体: n-gram だけ %.4f | 網だけ %.4f  (%d 位置, 網 %.0f 位置/秒)' % (bits(Pg), bits(Pn), len(p), len(p) / (time.time() - t1)))
G = np.r_[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


def fit(sl):
    g = np.zeros(6)
    for b in range(1, 6):
        m = (hi[sl] == b)
        if m.sum() < 50: g[b] = 0.5; continue
        g[b] = G[int(np.argmin([bits((1 - x) * Pg[sl][m] + x * Pn[sl][m]) for x in G]))]
    return g


gA = fit(slice(0, h)); gB = fit(slice(h, None))
for b in range(1, 6):
    m = hi == b
    if m.sum(): lg('NETGATE hi=%d: 割合 %.3f | n-gram %.3f | 網 %.3f | 混合重み(網) 前半 %.2f 後半 %.2f' % (b, m.mean(), bits(Pg[m]), bits(Pn[m]), gA[b], gB[b]))
Pmix = np.where(np.arange(len(p)) < h, (1 - gB[hi]) * Pg + gB[hi] * Pn, (1 - gA[hi]) * Pg + gA[hi] * Pn)        # 前半は 後半で選んだ重み / 後半は 前半で選んだ重み (交差)
lg('NETGATE 全位置で混合 (交差検証): %.4f bit' % bits(Pmix))
for h0 in range(0, 6):
    use = hi <= h0; Pp = np.where(use, Pmix, Pg); lg('NETGATE 網は hi<=%d の位置だけ: 網を動かす割合 %.3f -> %.4f bit' % (h0, use.mean(), bits(Pp)))
lg('終わり')
