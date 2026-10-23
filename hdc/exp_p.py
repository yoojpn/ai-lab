"""
P群: 前提を変える案の最初の試験。小さな言語モデル(lab.Net: 直前8語+その前4語の平均)の土俵で、bit/語を測る。
  p1 <K>      : 小さな専門家K個(文脈のクラスタごとに別々に学習)を、検索(最寄りの中心)で選ぶ。1個の同じ大きさのモデルと比べる
  p2 bp|local : 3層MLP。逆伝播 対 層ごとの局所学習(補助の出力層、層間の勾配を切る)
  p3          : コーパスそのものをモデルにする。訓練窓の索引(最寄りk個の次の語)だけの予測、NNとの混合
  p4          : 重みをハッシュで共有(小さな銀行から大きな行列を作る) 対 同じパラメーター数の普通のNN
  p5          : 確信度の門。小さなNNが自信のあるときはそれで済ませ、無いときだけ大きなNNを使う(計算量と品質の曲線)
使い方: LADDER_CORPUS=... python3 exp_p.py <mode> <N> [引数]
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab, items
from lab import V, HIST, NEAR, DE, bits, log
import exp_scale as es

es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


def macs_net(H):
    return (NEAR + 1) * DE * H + H * V


def ctx_embed(E0, X, last=None):
    """窓の語の埋め込み平均(チャンクで)"""
    out = []
    for i in range(0, len(X), 200000):
        xb = X[i:i + 200000]
        xb = xb if last is None else xb[:, -last:]
        out.append(E0[xb].mean(1))
    return np.concatenate(out).astype(np.float32)


def assign_nearest(emb, C):
    out = []
    Ct = torch.from_numpy(C)
    for i in range(0, len(emb), 100000):
        e = torch.from_numpy(emb[i:i + 100000])
        d = (e * e).sum(1, keepdim=True) - 2 * e @ Ct.T + (Ct * Ct).sum(1)[None]
        out.append(d.argsort(1)[:, :2].numpy())
    return np.concatenate(out)


def p1(ctx, K):
    E0 = items.ppmi_embeddings(ctx, DE)
    from scipy.cluster.vq import kmeans2
    emb_tr = ctx_embed(E0, ctx.Xtr); emb_te = ctx_embed(E0, ctx.Xte)
    rs = np.random.default_rng(0)
    C, _ = kmeans2(emb_tr[rs.choice(len(emb_tr), min(200000, len(emb_tr)), replace=False)], K, minit='++', seed=0)
    C = C.astype(np.float32)
    a_tr = assign_nearest(emb_tr, C)[:, 0]; a_te = assign_nearest(emb_te, C)
    sizes = np.bincount(a_tr, minlength=K)
    log('クラスタの大きさ 最小%d 最大%d' % (sizes.min(), sizes.max()))
    t0 = time.time()
    single = lab.train(ctx, 256, 1.0, E0=E0)
    r0 = lab.forward_all(single, ctx, 'te'); b0 = bits(r0['p'])
    log('単体(H=256): %.4f bit/語 [%.0fs]' % (b0, time.time() - t0))
    P = np.zeros((K, len(ctx.yte)))
    params = sum(p.numel() for p in single.parameters())
    for k in range(K):
        idx = np.where(a_tr == k)[0]
        if len(idx) < 2048:
            net = single
        else:
            net = lab.train(ctx, 256, 1.0, E0=E0, idx=idx, seed=k)
        P[k] = lab.forward_all(net, ctx, 'te')['p']
    top1 = P[a_te[:, 0], np.arange(len(a_te))]
    top2 = 0.5 * (top1 + P[a_te[:, 1], np.arange(len(a_te))])
    res = dict(K=K, single=b0, top1=bits(top1), top2=bits(top2), params_single=params, params_total=params * K,
               macs_top1=macs_net(256), macs_top2=2 * macs_net(256), min_cluster=int(sizes.min()), max_cluster=int(sizes.max()))
    return res


class LNet(torch.nn.Module):
    def __init__(s, H=256, nl=3):
        super().__init__()
        s.E = torch.nn.Embedding(V, DE)
        s.ls = torch.nn.ModuleList([torch.nn.Linear((NEAR + 1) * DE if i == 0 else H, H) for i in range(nl)])
        s.heads = torch.nn.ModuleList([torch.nn.Linear(H, V) for _ in range(nl)])

    def feats(s, X):
        e = s.E(X)
        return torch.cat([e[:, -NEAR:].reshape(len(X), -1), e[:, :-NEAR].mean(1)], 1)

    def forward(s, X, Hd=None, local=False):
        h = s.feats(X); outs = []
        for l, hd in zip(s.ls, s.heads):
            h = torch.relu(l(h))
            outs.append(hd(h))
            if local:
                h = h.detach()
        return outs if local else outs[-1]


def p2(ctx, mode):
    torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
    net = LNet(); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
    steps = len(X) // 512; st = 0; t0 = time.time()
    for i in torch.randperm(len(X)).split(512):
        if len(i) < 512:
            break
        for g in opt.param_groups:
            g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
        if mode == 'local':
            outs = net(X[i], local=True)
            loss = sum(torch.nn.functional.cross_entropy(o, y[i]) for o in outs)
        else:
            loss = torch.nn.functional.cross_entropy(net(X[i]), y[i])
        opt.zero_grad(); loss.backward(); opt.step(); st += 1
    pt = []
    with torch.no_grad():
        for i in range(0, len(ctx.Xte), 4096):
            if mode == 'local':
                z = net(torch.from_numpy(ctx.Xte[i:i + 4096]), local=True)[-1]
            else:
                z = net(torch.from_numpy(ctx.Xte[i:i + 4096]))
            lp = torch.log_softmax(z, 1)
            pt.append(lp[torch.arange(len(z)), torch.from_numpy(ctx.yte[i:i + 4096])].exp().numpy())
    return dict(mode=mode, bits=bits(np.concatenate(pt).astype(np.float64)), sec=time.time() - t0)


def p3(ctx):
    E0 = items.ppmi_embeddings(ctx, DE)
    torch.set_num_threads(os.cpu_count() or 2)
    net = lab.train(ctx, 256, 1.0, E0=E0)
    rn_va = lab.forward_all(net, ctx, 'va'); rn_te = lab.forward_all(net, ctx, 'te')
    nq = 20000
    def keys(X):
        k = E0[X[:, -3:]].reshape(len(X), -1).astype(np.float32)
        return k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)
    Kd = torch.cat([torch.from_numpy(keys(ctx.Xtr[i:i + 500000])).half() for i in range(0, len(ctx.Xtr), 500000)]); yd = torch.from_numpy(ctx.ytr)
    uni = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1; uni /= uni.sum()
    def knn(Xq, yq, k=64, sizes=None):
        sizes = sizes or (len(Kd) // 16, len(Kd) // 4, len(Kd))
        Q = torch.from_numpy(keys(Xq)); out = {}
        for M in [s for s in sizes if s <= len(Kd)]:
            sc_all, ix_all = [], []
            for qi in range(0, len(Q), 500):
                qb = Q[qi:qi + 500]; bs = bi = None
                for c in range(0, M, 500000):
                    s = qb @ Kd[c:min(c + 500000, M)].float().T
                    v, ix = s.topk(k, 1); ix = ix + c
                    if bs is None:
                        bs, bi = v, ix
                    else:
                        bs, j = torch.cat([bs, v], 1).topk(k, 1); bi = torch.cat([bi, ix], 1).gather(1, j)
                sc_all.append(bs); ix_all.append(bi)
            out[M] = (torch.cat(sc_all), torch.cat(ix_all))
        return out
    nv = nq
    res = dict(net_bits_val_subset=None)
    kv = knn(ctx.Xva[:nv], ctx.yva[:nv]); kt = knn(ctx.Xte[:nq], ctx.yte[:nq])
    pnv = rn_va['p'][:nv]; pnt = rn_te['p'][:nq]
    res['net_bits'] = bits(pnt)
    def pk(sc_ix, yq, T):
        sc, ix = sc_ix
        w = torch.softmax(sc / T, 1)
        yn = yd[ix]
        return (w * (yn == torch.from_numpy(yq)[:, None])).sum(1).numpy().astype(np.float64)
    for M in kv:
        best = None
        for T in (0.02, 0.05, 0.1):
            pkv = pk(kv[M], ctx.yva[:nv], T)
            for lam in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8):
                pa = lam * pkv + (1 - lam) * uni[ctx.yva[:nv]]
                ba = bits(pa)
                for w in (0.05, 0.1, 0.2, 0.3, 0.5):
                    bm = bits((1 - w) * pnv + w * (lam * pkv + (1 - lam) * uni[ctx.yva[:nv]]))
                    if best is None or bm < best[0]:
                        best = (bm, T, lam, w)
        _, T, lam, w = best
        pkt = pk(kt[M], ctx.yte[:nq], T)
        pa = lam * pkt + (1 - lam) * uni[ctx.yte[:nq]]
        res['ds=%d' % M] = dict(knn_alone=bits(pa), mix=bits((1 - w) * pnt + w * pa), T=T, lam=lam, w=w)
    return res


class HashLinear(torch.nn.Module):
    def __init__(s, fin, fout, bank, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        s.register_buffer('idx', torch.randint(0, bank, (fout, fin), generator=g))
        s.register_buffer('sgn', (torch.randint(0, 2, (fout, fin), generator=g) * 2 - 1).float())
        s.bank = torch.nn.Parameter(torch.randn(bank) / math.sqrt(fin))
        s.b = torch.nn.Parameter(torch.zeros(fout))

    def forward(s, x):
        W = s.bank[s.idx] * s.sgn
        return x @ W.T + s.b


class HNet(torch.nn.Module):
    def __init__(s, H, bank1, bank2, E0):
        super().__init__()
        s.E = torch.nn.Embedding(V, DE); s.E.weight.data.copy_(torch.from_numpy(E0).float())
        s.l1 = HashLinear((NEAR + 1) * DE, H, bank1, 1); s.l2 = HashLinear(H, V, bank2, 2)

    def forward(s, X, Hd=None):
        e = s.E(X)
        return s.l2(torch.relu(s.l1(torch.cat([e[:, -NEAR:].reshape(len(X), -1), e[:, :-NEAR].mean(1)], 1))))


def p4(ctx, E0=None):
    E0 = items.ppmi_embeddings(ctx, DE)
    res = {}
    def evalnet(net, name):
        r = lab.forward_all(net, ctx, 'te'); res[name] = dict(bits=bits(r['p']), params=sum(p.numel() for p in net.parameters()))
        log('%s %s' % (name, res[name]))
    for H in (48, 180, 1024):
        evalnet(lab.train(ctx, H, 1.0, E0=E0), 'real_H%d' % H)
    for bank in (250000, 1000000):
        net = HNet(1024, bank // 4, bank - bank // 4, E0)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr); steps = len(X) // 512; st = 0
        torch.set_num_threads(os.cpu_count() or 2)
        for i in torch.randperm(len(X)).split(512):
            if len(i) < 512:
                break
            for g in opt.param_groups:
                g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
            loss = torch.nn.functional.cross_entropy(net(X[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step(); st += 1
        evalnet(net, 'hash_virtualH1024_bank%d' % bank)
    return res


def p5(ctx):
    E0 = items.ppmi_embeddings(ctx, DE)
    small = lab.train(ctx, 64, 1.0, E0=E0); big = lab.train(ctx, 512, 1.0, E0=E0)
    rs = lab.forward_all(small, ctx, 'te'); rb = lab.forward_all(big, ctx, 'te')
    ms, mb = macs_net(64), macs_net(512)
    res = dict(small=dict(bits=bits(rs['p']), macs=ms), big=dict(bits=bits(rb['p']), macs=mb), curve=[])
    for q in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        thr = np.quantile(rs['maxp'], q)
        usebig = rs['maxp'] < thr
        p = np.where(usebig, rb['p'], rs['p'])
        res['curve'].append(dict(frac_big=float(usebig.mean()), bits=bits(p), macs=float(ms + usebig.mean() * mb)))
    for H in (128, 256):
        r = lab.forward_all(lab.train(ctx, H, 1.0, E0=E0), ctx, 'te'); res['single_H%d' % H] = dict(bits=bits(r['p']), macs=macs_net(H))
    return res


def main():
    mode = sys.argv[1]; N = int(sys.argv[2]); arg = sys.argv[3] if len(sys.argv) > 3 else None
    torch.set_num_threads(os.cpu_count() or 2)
    t0 = time.time()
    ctx = lab.load(N)
    log('データ準備 [%.0fs]' % (time.time() - t0))
    if mode == 'p1':
        res = p1(ctx, int(arg))
    elif mode == 'p2':
        res = p2(ctx, arg)
    elif mode == 'p3':
        res = p3(ctx)
    elif mode == 'p4':
        res = p4(ctx)
    elif mode == 'p5':
        res = p5(ctx)
    res['mode'] = mode; res['N'] = N; res['sec'] = time.time() - t0
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
