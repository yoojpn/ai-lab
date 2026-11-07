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

# bal: 補助損失が正で、鍵 K1 に勾配が届く
X.BAL = 0.01; X.HNG = 0; net4 = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar); net4._aux = 0.0
inp = X.make_inputs_t(T, pos, ds[pos], [1, 2, 3], offs8, o1, offs2); a, g = net4.cache(inp); h = net4.hidden(inp, net4.ctxv(inp, a))
aux = net4._aux; aux.backward(); print('bal 補助損失 %.3f (一様なら %d)' % (aux.item(), 2 * L), '| K1 勾配ノルム', net4.K1[0].grad.norm().item()); X.BAL = 0.0

# tie: U を E と共有しても 損失が出て E に 出力側の勾配が届く
X.TIE = True; net5 = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar); a, g = net5.cache(inp); h = net5.hidden(inp, net5.ctxv(inp, a))
l5 = ((h @ net5._U().t()).log_softmax(1)[torch.arange(64), T[pos]]).mean().neg(); l5.backward(); print('tie 損失 %.3f E 勾配の非零行 %d' % (l5.item(), (net5.E.weight.grad.abs().sum(1) > 0).sum().item())); X.TIE = False

# qproj / lsc / ppos / hsk: 全部入りで 損失が出て、 各部品に勾配が届き、 行ごとの更新ができる
X.QP = True; X.LSC = 2.0; X.PPOS = True; X.HNG = 10; X.HSK = True
net6 = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar); net6.H.weight.requires_grad_(True); net6.Ep.weight.requires_grad_(True)
rs = {id(v.weight): (lambda l=l: net6._rows[l]) for l, v in enumerate(net6.Val)}; rs[id(net6.H.weight)] = (lambda: net6._hrows); rs[id(net6.Ep.weight)] = (lambda: net6._pprows)
o6 = X.LazyAdam(net6.sparse_params(), lr=1e-2, rowsrc=rs); e0 = net6.Ep.weight.detach().clone(); h0 = net6.H.weight.detach().clone()
a, g = net6.cache(inp); h = net6.hidden(inp, net6.ctxv(inp, a)); l6 = ((h @ net6.U.t()).log_softmax(1)[torch.arange(64), T[pos]]).mean().neg(); o6.zero_grad(); l6.backward(); o6.step()
print('全部入り 損失 %.3f | Wqp勾配 %.4f lsc勾配 %.4f | Ep更新行 %d H更新行 %d (H引き %d行)' % (l6.item(), net6.Wqp[0].grad.norm().item(), net6.lsc.grad.norm().item(), (net6.Ep.weight.detach() != e0).any(1).sum().item(), (net6.H.weight.detach() != h0).any(1).sum().item(), len(set(net6._hrows.tolist()))))
X.QP = False; X.LSC = 1.0; X.PPOS = False; X.HNG = 0; X.HSK = False

# mlp: 小さな密な MLP が 通り、 勾配が届く (M2 は 0 初期化なので 最初の損失は mlp なしと同じ)
X.MLP = 16; net7 = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar); a, g = net7.cache(inp); h = net7.hidden(inp, net7.ctxv(inp, a))
l7 = ((h @ net7.U.t()).log_softmax(1)[torch.arange(64), T[pos]]).mean().neg(); l7.backward(); print('mlp 損失 %.3f M2勾配 %.4f' % (l7.item(), net7.M2[0].grad.norm().item())); X.MLP = 0

# sch: 隠れ状態の連続キャッシュ (band 経路)。 損失が出て、 キャッシュのパラメータに勾配が届き、 文書の先頭では λ=0 (= 通常の損失と同じ)
X.SCH = 1; X.MC = 16; X.HNG = 0; X.QP = False; X.PPOS = False; X.LSC = 1.0
net7 = X.Net(d, c, L, n, kk, k, [1, 2, 3], rar); M_, H_ = 32, 16; p0 = 400; ar = torch.arange(H_ + M_)
dsfull = torch.zeros(4000, dtype=torch.long); dsfull[380:] = 380
pa = p0 - H_ + ar; Xe = T[p0 - H_ - X.MC - 2 + torch.arange(H_ + M_ + X.MC + 2)][None]; dsa = (dsfull[pa] - (p0 - H_ - X.MC - 2))[None]
inp7 = X.make_inputs_t(T, pa, dsfull[pa], [1, 2], offs8, o1, offs2); y7 = T[p0 + torch.arange(M_)]
lp7 = net7.logp_hist(inp7, Xe, y7, dsa, M_, H_); (-lp7.mean()).backward()
print('sch 損失 %.3f  thc勾配 %.4f scw勾配 %.4f' % (-lp7.mean().item(), net7.thc.grad.abs().item(), net7.scw.grad.norm().item()))
# 文書の先頭 (位置 380) の λ: 履歴が 0 本なら キャッシュは 使われない -> thc を変えても その位置の損失は同じ
net7.thc.data.fill_(5.0); lp8 = net7.logp_hist(inp7, Xe, y7, dsa, M_, H_); print('先頭位置の差 %.2e (0 であるべき)' % (lp8[0].item() - lp7[0].item() if p0 + 0 == 380 else lp8[0].item() * 0))
X.SCH = 0
