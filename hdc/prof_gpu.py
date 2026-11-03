"""1歩の時間を 部分ごとに測る (GPU の何が遅いかを見る)。
使い方 (Colab): !cd /content/ai-lab && git pull -q && cd hdc && python3 prof_gpu.py
"""
import time, json, numpy as np, torch, torch.nn.functional as F
import exp_ctx as X

W = '/tmp/tok'; dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tv = np.fromfile(W + '/val.bin', dtype=np.uint16); Dv = np.load(W + '/val_docs.npy')
S = Tv[:2_500_000].astype(np.int64); N = len(S); DS = np.maximum(Dv[np.maximum(np.searchsorted(Dv, np.arange(N), 'right') - 1, 0)], 0)
c = 8; B = 512; feats = [1, 2, 3]; offs8 = np.arange(1, c + 1); o1 = [1 + (np.arange(X.NS) * w) // X.NS for w in X.W1]; offs2 = np.arange(c + 1, X.W2 + 1)
cnt = np.bincount(S, minlength=X.V).astype(np.float64); pu = (cnt + 1) / (cnt.sum() + X.V); rar = -np.log2(pu); rar[0] = 0; rar[1] = 0
net = X.Net(128, c, 4, 256, 4, 4, feats, rar).to(dev)
opt_s = torch.optim.SparseAdam(net.sparse_params(), lr=5e-3); opt_d = torch.optim.Adam(net.dense_params(), lr=2e-3)
rng = np.random.default_rng(0); pos = rng.integers(5000, N, B * 60)
sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)
T = {k: 0.0 for k in ('入力の準備(CPU)', 'GPUへ送る', '順方向', '逆方向', '疎な更新(SparseAdam)', '密な更新(Adam)')}; n = 0
for i in range(50):
    pp = pos[i * B:(i + 1) * B]
    X.DEV = torch.device('cpu'); t0 = time.time(); inp = X.make_inputs(S, pp, DS[pp], feats, offs8, o1, offs2); y = torch.from_numpy(S[pp]); t1 = time.time()
    inp = {k: v.to(dev) for k, v in inp.items()}; y = y.to(dev); sync(); t2 = time.time()
    loss = -net.logp(inp, y).mean(); sync(); t3 = time.time()
    opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); sync(); t4 = time.time()
    opt_s.step(); sync(); t5 = time.time(); opt_d.step(); sync(); t6 = time.time()
    if i >= 10:
        n += 1
        for k, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)): T[k] += v
tot = sum(T.values()) / n
print('device', dev, ' 1歩あたり 合計 %.1f ms (%.1f 歩/秒)' % (tot * 1000, 1 / tot))
for k, v in T.items(): print('  %-22s %6.1f ms  %4.0f%%' % (k, v / n * 1000, 100 * v / n / tot))
