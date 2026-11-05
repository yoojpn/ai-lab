import sys, copy; sys.path.insert(0, '/home/claude/ai-lab/hdc')
import numpy as np, torch, exp_ctx as X
X.V = 512; X.SPARSE = False; X.MC = 16; X.W2 = 16; X.K2 = 4; X.CV = True
d, c, L, n, kk, k = 32, 8, 2, 16, 4, 4
rar = np.random.rand(X.V)
def build(vs):
    torch.manual_seed(0); net = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar)
    X.Net.VSINK = vs; sinks = {}
    if vs: net.Vsink = [torch.zeros_like(v.weight) for v in net.Val]; sinks = {id(v.weight): net.Vsink[l] for l, v in enumerate(net.Val)}
    rs = {id(v.weight): (lambda l=l: net._rows[l]) for l, v in enumerate(net.Val)}
    return net, X.LazyAdam(net.sparse_params(), lr=1e-2, rowsrc=rs, sinks=sinks), torch.optim.Adam(net.dense_params(), lr=1e-3)
rng = np.random.default_rng(1); T = torch.from_numpy(rng.integers(1, X.V, 4000)); ds = torch.zeros(4000, dtype=torch.long)
offs8 = torch.arange(1, c + 1); o1 = [torch.arange(1, 9), torch.arange(9, 17)]; offs2 = torch.arange(1, 17)
res = []
for vs in (False, True):
    X.W1 = [4, 8]; net, os_, od = build(vs)
    for step in range(5):
        pos = torch.from_numpy(rng.integers(100, 3900, 64)) if False else torch.arange(100 + step * 50, 164 + step * 50)
        inp = X.make_inputs_t(T, pos, ds[pos], [1, 2, 3], offs8, o1, offs2)
        a, g = net.cache(inp); cv = net.ctxv(inp, a); h = net.hidden(inp, cv)
        loss = ((h @ net.U.t()).log_softmax(1)[torch.arange(64), T[pos]]).mean().neg()
        os_.zero_grad(); od.zero_grad(); loss.backward(); os_.step(); od.step()
    res.append((loss.item(), [v.weight.clone() for v in net.Val], [b.clone() for b in net.Vsink] if vs else None))
print('loss', res[0][0], res[1][0])
for l in range(L): print('Val', l, 'max diff', (res[0][1][l] - res[1][1][l]).abs().max().item())
print('sink 残り(0であるべき)', [b.abs().max().item() for b in res[1][2]])

# vfac: 勾配が共有部品に届く・形が合う・R=1,vfull=0 の初期出力の大きさ
X.VFAC = 4; X.VFULL = False; X.Net.VSINK = False
net = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar); assert len(net.Val) == 0
pos = torch.arange(100, 164); inp = X.make_inputs_t(T, pos, ds[pos], [1, 2, 3], offs8, o1, offs2)
a, g = net.cache(inp); h = net.hidden(inp, net.ctxv(inp, a)); loss = ((h @ net.U.t()).log_softmax(1)[torch.arange(64), T[pos]]).mean().neg(); loss.backward()
print('vfac 損失', loss.item(), '| VA 勾配ノルム', [p.grad.norm().item() for p in net.VA], '| 全行表あり 併用も可')
X.VFULL = True; net2 = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar); print('併用の Val 数', len(net2.Val))

# hng: 表が引かれ、勾配が届き、LazyAdam の rowsrc で更新できる
X.HNG = 10; X.VFAC = 0; X.VFULL = True
net3 = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar); net3.H.weight.requires_grad_(True)
rs = {id(v.weight): (lambda l=l: net3._rows[l]) for l, v in enumerate(net3.Val)}; rs[id(net3.H.weight)] = (lambda: net3._hrows)
o3 = X.LazyAdam(net3.sparse_params(), lr=1e-2, rowsrc=rs); w0 = net3.H.weight.detach().clone()
inp = X.make_inputs_t(T, pos, ds[pos], [1, 2, 3], offs8, o1, offs2); a, g = net3.cache(inp); h = net3.hidden(inp, net3.ctxv(inp, a))
l3 = ((h @ net3.U.t()).log_softmax(1)[torch.arange(64), T[pos]]).mean().neg(); o3.zero_grad(); l3.backward(); o3.step()
ch = (net3.H.weight.detach() != w0).any(1).sum().item(); print('hng 損失', l3.item(), '更新された行', ch, '(引いた行 最大', len(set(net3._hrows.tolist())), ')')
