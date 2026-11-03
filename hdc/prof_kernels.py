"""GPU のどの計算が時間を食っているかを 上位から表示する (速い経路の 1 歩)。
使い方 (Colab): !cd /content/ai-lab && git pull -q && cd hdc && python3 prof_kernels.py
"""
import time, numpy as np, torch
import exp_ctx as X

W = '/tmp/tok'; dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); X.DEV = dev; X.SPARSE = False
print('device', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
Tv = np.fromfile(W + '/val.bin', dtype=np.uint16); Dv = np.load(W + '/val_docs.npy')
S = Tv[:2_500_000].astype(np.int64); N = len(S); DS = np.maximum(Dv[np.maximum(np.searchsorted(Dv, np.arange(N), 'right') - 1, 0)], 0)
c = 8; B = 512; feats = [1, 2, 3]
offs8 = torch.arange(1, c + 1, device=dev); o1 = [(1 + (torch.arange(X.NS, device=dev) * w) // X.NS) for w in X.W1]; offs2 = torch.arange(c + 1, X.W2 + 1, device=dev)
cnt = np.bincount(S, minlength=X.V).astype(np.float64); pu = (cnt + 1) / (cnt.sum() + X.V); rar = -np.log2(pu); rar[0] = 0; rar[1] = 0
net = X.Net(128, c, 4, 256, 4, 4, feats, rar).to(dev)
opt_s = X.LazyAdam(net.sparse_params(), lr=5e-3); opt_d = torch.optim.Adam(net.dense_params(), lr=2e-3)
S_t = torch.from_numpy(S).to(dev); DS_t = torch.from_numpy(DS).to(dev); pos_t = torch.from_numpy(np.random.default_rng(0).integers(5000, N, B * 40)).to(dev)
def step(i):
    pp = pos_t[i * B:(i + 1) * B]; inp = X.make_inputs_t(S_t, pp, DS_t[pp], feats, offs8, o1, offs2); y = S_t[pp]
    loss = -net.logp(inp, y).mean(); opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); opt_s.step(); opt_d.step(); return loss
for i in range(5): step(i)
if dev.type == 'cuda': torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == 'cuda' else [])
with profile(activities=acts) as prof:
    for i in range(5, 15): step(i)
    if dev.type == 'cuda': torch.cuda.synchronize()
key = 'cuda_time_total' if dev.type == 'cuda' else 'cpu_time_total'
print(prof.key_averages().table(sort_by=key, row_limit=14, max_name_column_width=60))
