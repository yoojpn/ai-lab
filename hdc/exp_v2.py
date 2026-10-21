"""
第2版(M1): 小さなTransformer + 埋め込み初期化(#10) + 冒頭の枠(#21) + 文書内の索引(#20) + 少ビット(#7)
使い方: LADDER_CORPUS=... python3 exp_v2.py <訓練語数N> <variant>
  variant: base | ppmi | ppmi_head | full(=ppmi_head+8ビット化)
出力: NN単独のbit/語、NN+索引のbit/語、基準(単語4-gram)との差、記憶・計算量
"""
import os, sys, time, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab, items
from lab import *
import exp_ladder as L, exp_scale as es, exp_class as ec, exp_cards as XC, exp_far as XF


class TF(torch.nn.Module):
    def __init__(s, E0=None, head=False, d=64, nl=2, dff=256, heads=4):
        super().__init__()
        s.head = head; s.d = d; s.nl = nl; s.dff = dff
        s.E = torch.nn.Embedding(V, d)
        if E0 is not None:
            s.E.weight.data.copy_(torch.from_numpy(E0).float())
        s.pos = torch.nn.Parameter(torch.zeros(HIST + (1 if head else 0), d))
        layer = torch.nn.TransformerEncoderLayer(d, heads, dff, dropout=0.0, batch_first=True, norm_first=True)
        s.enc = torch.nn.TransformerEncoder(layer, nl)
        s.out = torch.nn.Linear(d, V)

    def forward(s, X, Hd=None):
        e = s.E(X)
        if s.head:
            e = torch.cat([s.E(Hd).mean(1, keepdim=True), e], 1)
        return s.out(s.enc(e + s.pos)[:, -1])

    def macs(s):
        n = HIST + (1 if s.head else 0)
        return n * s.nl * (4 * s.d * s.d + 2 * s.d * s.dff) + s.nl * 2 * n * n * s.d + s.d * V


def probs_of(net, ctx, split, head):
    X, y = {'va': (ctx.Xva, ctx.yva), 'te': (ctx.Xte, ctx.yte)}[split]
    Hd = heads(ctx, split) if head else None
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            z = net(torch.from_numpy(X[i:i + 4096]), torch.from_numpy(Hd[i:i + 4096]) if head else None)
            lp = torch.log_softmax(z, 1)
            out.append(lp[torch.arange(len(z)), torch.from_numpy(y[i:i + 4096])].exp().numpy())
    return np.concatenate(out).astype(np.float64)


def quantize8(net):
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    n = 0
    for k, v in sd.items():
        if v.dim() >= 2:
            sc = v.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
            sd[k] = torch.round(v / sc).clamp(-127, 127) * sc; n += v.numel()
    net.load_state_dict(sd)
    return n


def main():
    N = int(sys.argv[1]); variant = sys.argv[2]; size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    d, nl, dff, nh = {1: (64, 2, 256, 4), 2: (128, 3, 512, 4), 3: (192, 4, 768, 6)}[size]
    t0 = time.time(); torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    ctx = lab.load(N)
    log('データ準備 [%.0fs]' % (time.time() - t0))
    # 文書内の索引(記憶に数えない)
    mods, tot = [], 0
    for a in ctx.tr_i:
        if len(a) <= HIST: continue
        mods.append(a); tot += len(a) - HIST
        if tot >= N: break
    pu = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1.0; pu /= pu.sum()
    cls, wc = ec.exchange(mods, 200, 6, lambda s: None)
    cp_va, _ = XC.copy_probs(ctx.Xva, ctx.yva, pu); cp_te, _ = XC.copy_probs(ctx.Xte, ctx.yte, pu)
    ca_va = XC.cache_probs(ctx.Xva, ctx.yva, pu); ca_te = XC.cache_probs(ctx.Xte, ctx.yte, pu)
    fc_va = XF.farcache_probs(ctx.va_i, ctx.mva, ctx.pva, ctx.yva, 300, pu); fc_te = XF.farcache_probs(ctx.te_i, ctx.mte, ctx.pte, ctx.yte, 300, pu)
    lp_va, _ = XF.local_for_windows(ctx.va_i, ctx.mva, ctx.pva, ctx.yva, cls, pu); lp_te, _ = XF.local_for_windows(ctx.te_i, ctx.mte, ctx.pte, ctx.yte, cls, pu)
    names = ['L1', 'L2', 'L3', 'S2', 'LC2', 'LCW']
    loc_va = [cp_va, ca_va, fc_va] + [lp_va[n] for n in names]; loc_te = [cp_te, ca_te, fc_te] + [lp_te[n] for n in names]
    log('文書内の索引を計算 [%.0fs]' % (time.time() - t0))
    # 基準: 単語4-gram
    w3 = lambda X: X[:, HIST - 3:]
    A = es.NG(w3(ctx.Xtr), ctx.ytr)
    r = A.probs(w3(ctx.Xte), ctx.yte); r = r[0] if isinstance(r, tuple) else r
    base4 = bits(r)
    # モデル
    E0 = items.ppmi_embeddings(ctx, d) if variant != 'base' else None
    head = variant in ('ppmi_head', 'full')
    net = TF(E0, head, d, nl, dff, nh)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
    Hd = torch.from_numpy(heads(ctx, 'tr')) if head else None
    steps = len(X) // 512; st = 0
    for i in torch.randperm(len(X)).split(512):
        if len(i) < 512: break
        for g in opt.param_groups: g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
        loss = torch.nn.functional.cross_entropy(net(X[i], Hd[i] if head else None), y[i])
        opt.zero_grad(); loss.backward(); opt.step(); st += 1
    log('学習完了 [%.0fs]' % (time.time() - t0))
    params = sum(p.numel() for p in net.parameters()); mb = params * 4 / 1e6
    if variant == 'full':
        nq = quantize8(net); mb = nq / 1e6 + (params - nq) * 4 / 1e6
    pv, pt = probs_of(net, ctx, 'va', head), probs_of(net, ctx, 'te', head)
    w = L.em(np.array([pv] + loc_va)); mix = bits((w[:, None] * np.array([pt] + loc_te)).sum(0))
    nn = bits(pt)
    trainmacs = 3.0 * net.macs() * steps * 512
    log('size=%d variant=%s N=%d: NN単独 %.3f (4-gram比 %+.1f%%) | NN+文書内の索引 %.3f (%+.1f%%) | 4-gram %.3f | パラメーター%d 記憶%.1fMB 1語の積和%.2fM 学習の総積和%.3e [%.0fs]' % (
        size, variant, N, nn, 100 * (1 - nn / base4), mix, 100 * (1 - mix / base4), base4, params, mb, net.macs() / 1e6, trainmacs, time.time() - t0))


if __name__ == '__main__':
    main()
