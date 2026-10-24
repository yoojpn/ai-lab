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
    nl = 3
    if ':' in mode:
        mode, nl = mode.split(':'); nl = int(nl)
    torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 2)
    net = LNet(256, nl); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
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
    return dict(mode=mode, layers=nl, bits=bits(np.concatenate(pt).astype(np.float64)), sec=time.time() - t0)


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


def p6(ctx, maxM):
    """索引を大きくし続けると、改善は伸び続けるか。NNは ctx の窓(N語)で学習。索引は文書全体の先頭から maxM 語まで(int8の96次元キー)"""
    E0 = items.ppmi_embeddings(ctx, DE)
    E32 = E0[:, :32].astype(np.float32)
    torch.set_num_threads(os.cpu_count() or 2)
    net = lab.train(ctx, 256, 1.0, E0=E0)
    rn_va = lab.forward_all(net, ctx, 'va'); rn_te = lab.forward_all(net, ctx, 'te')
    def keys(w3):
        k = E32[w3].reshape(len(w3), -1)
        k = k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)
        return k
    Kl, yl, tot = [], [], 0
    for a in ctx.tr_i:
        if len(a) < 4:
            continue
        w3 = np.stack([a[0:-3], a[1:-2], a[2:-1]], 1)
        Kl.append(np.round(keys(w3) * 127).astype(np.int8)); yl.append(a[3:].astype(np.int16)); tot += len(a) - 3
        if tot >= maxM:
            break
    Kd = torch.from_numpy(np.concatenate(Kl)); yd = torch.from_numpy(np.concatenate(yl)).long()
    del Kl, yl
    log('索引 %d 語 (%.1fGB)' % (len(Kd), Kd.numel() / 1e9))
    uni = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1; uni /= uni.sum()
    nq = 20000
    sizes = [m for m in (4000000, 16000000, 32000000, 64000000) if m <= len(Kd)]
    if len(Kd) not in sizes:
        sizes.append(len(Kd))
    def knn(Xq, k=64):
        Q = torch.from_numpy(keys(Xq[:, -3:]).astype(np.float32) * 127)
        out = {m: [[], []] for m in sizes}
        for qi in range(0, len(Q), 1000):
            qb = Q[qi:qi + 1000]; bs = bi = None; done = {}
            for c in range(0, sizes[-1], 250000):
                e = min(c + 250000, sizes[-1])
                s_ = qb @ Kd[c:e].float().T
                v, ix = s_.topk(k, 1); ix = ix + c
                if bs is None:
                    bs, bi = v, ix
                else:
                    bs, j = torch.cat([bs, v], 1).topk(k, 1); bi = torch.cat([bi, ix], 1).gather(1, j)
                for m in sizes:
                    if e == m:
                        done[m] = (bs.clone(), bi.clone())
            for m in sizes:
                out[m][0].append(done[m][0] / 127 / 127); out[m][1].append(done[m][1])
        return {m: (torch.cat(out[m][0]), torch.cat(out[m][1])) for m in sizes}
    kv = knn(ctx.Xva[:nq]); kt = knn(ctx.Xte[:nq])
    pnv = rn_va['p'][:nq]; pnt = rn_te['p'][:nq]
    res = dict(net_bits=bits(pnt), index_tokens=int(len(Kd)))
    def pk(sc_ix, yq, T):
        sc, ix = sc_ix
        w = torch.softmax(sc / T, 1)
        return (w * (yd[ix] == torch.from_numpy(yq)[:, None])).sum(1).numpy().astype(np.float64)
    for m in sizes:
        best = None
        for T in (0.02, 0.05):
            pkv = pk(kv[m], ctx.yva[:nq], T)
            for lam in (0.4, 0.6, 0.8, 0.9):
                pa = lam * pkv + (1 - lam) * uni[ctx.yva[:nq]]
                for w in (0.05, 0.1, 0.2, 0.3, 0.5):
                    bm = bits((1 - w) * pnv + w * pa)
                    if best is None or bm < best[0]:
                        best = (bm, T, lam, w)
        _, T, lam, w = best
        pa = lam * pk(kt[m], ctx.yte[:nq], T) + (1 - lam) * uni[ctx.yte[:nq]]
        res['ds=%d' % m] = dict(knn_alone=bits(pa), mix=bits((1 - w) * pnt + w * pa), gain=res['net_bits'] - bits((1 - w) * pnt + w * pa), T=T, lam=lam, w=w)
        log('索引%d語: 混合 %.4f (改善 %.4f)' % (m, res['ds=%d' % m]['mix'], res['ds=%d' % m]['gain']))
    return res

def p7(ctx, maxM, Hs=(32, 64, 128, 256, 512)):
    """索引を大きくし続けると、改善は伸び続けるか。NNは ctx の窓(N語)で学習。索引は文書全体の先頭から maxM 語まで(int8の96次元キー)"""
    E0 = items.ppmi_embeddings(ctx, DE)
    E32 = E0[:, :32].astype(np.float32)
    torch.set_num_threads(os.cpu_count() or 2)
    def keys(w3):
        k = E32[w3].reshape(len(w3), -1)
        k = k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)
        return k
    Kl, yl, tot = [], [], 0
    for a in ctx.tr_i:
        if len(a) < 4:
            continue
        w3 = np.stack([a[0:-3], a[1:-2], a[2:-1]], 1)
        Kl.append(np.round(keys(w3) * 127).astype(np.int8)); yl.append(a[3:].astype(np.int16)); tot += len(a) - 3
        if tot >= maxM:
            break
    Kd = torch.from_numpy(np.concatenate(Kl)); yd = torch.from_numpy(np.concatenate(yl)).long()
    del Kl, yl
    log('索引 %d 語 (%.1fGB)' % (len(Kd), Kd.numel() / 1e9))
    uni = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1; uni /= uni.sum()
    nq = 20000
    sizes = [m for m in (4000000, 16000000, 32000000, 64000000) if m <= len(Kd)]
    if len(Kd) not in sizes:
        sizes.append(len(Kd))
    def knn(Xq, k=64):
        Q = torch.from_numpy(keys(Xq[:, -3:]).astype(np.float32) * 127)
        out = {m: [[], []] for m in sizes}
        for qi in range(0, len(Q), 1000):
            qb = Q[qi:qi + 1000]; bs = bi = None; done = {}
            for c in range(0, sizes[-1], 250000):
                e = min(c + 250000, sizes[-1])
                s_ = qb @ Kd[c:e].float().T
                v, ix = s_.topk(k, 1); ix = ix + c
                if bs is None:
                    bs, bi = v, ix
                else:
                    bs, j = torch.cat([bs, v], 1).topk(k, 1); bi = torch.cat([bi, ix], 1).gather(1, j)
                for m in sizes:
                    if e == m:
                        done[m] = (bs.clone(), bi.clone())
            for m in sizes:
                out[m][0].append(done[m][0] / 127 / 127); out[m][1].append(done[m][1])
        return {m: (torch.cat(out[m][0]), torch.cat(out[m][1])) for m in sizes}
    kv = knn(ctx.Xva[:nq]); kt = knn(ctx.Xte[:nq])
    res = dict(index_tokens=int(len(Kd)), nets={})
    def pk(sc_ix, yq, T):
        sc, ix = sc_ix
        w = torch.softmax(sc / T, 1)
        return (w * (yd[ix] == torch.from_numpy(yq)[:, None])).sum(1).numpy().astype(np.float64)
    for H in Hs:
        net = lab.train(ctx, H, 1.0, E0=E0)
        rn_va = lab.forward_all(net, ctx, 'va'); rn_te = lab.forward_all(net, ctx, 'te')
        pnv = rn_va['p'][:nq]; pnt = rn_te['p'][:nq]
        r = dict(macs=macs_net(H), params=sum(p.numel() for p in net.parameters()), net_bits=bits(pnt))
        for m in sizes:
            best = None
            for T in (0.02, 0.05):
                pkv = pk(kv[m], ctx.yva[:nq], T)
                for lam in (0.4, 0.6, 0.8, 0.9):
                    pa = lam * pkv + (1 - lam) * uni[ctx.yva[:nq]]
                    for w in (0.05, 0.1, 0.2, 0.3, 0.5):
                        bm = bits((1 - w) * pnv + w * pa)
                        if best is None or bm < best[0]:
                            best = (bm, T, lam, w)
            _, T, lam, w = best
            pa = lam * pk(kt[m], ctx.yte[:nq], T) + (1 - lam) * uni[ctx.yte[:nq]]
            r['ds=%d' % m] = bits((1 - w) * pnt + w * pa)
        res['nets']['H=%d' % H] = r
        log('H=%d %s' % (H, r))
    return res


def p8(ctx, maxM):
    """索引の検索を軽くする: 総当たりをやめ、分割(IVF)+圧縮(PQ 16バイト/語)で引く。品質と検索時間を測る"""
    import faiss
    E0 = items.ppmi_embeddings(ctx, DE)
    E32 = E0[:, :32].astype(np.float32)
    torch.set_num_threads(os.cpu_count() or 2); faiss.omp_set_num_threads(os.cpu_count() or 2)
    net = lab.train(ctx, 256, 1.0, E0=E0)
    rn_va = lab.forward_all(net, ctx, 'va'); rn_te = lab.forward_all(net, ctx, 'te')
    def keys(w3):
        k = E32[w3].reshape(len(w3), -1)
        return (k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
    nlist, M = 4096, 16
    quant = faiss.IndexFlatIP(96)
    index = faiss.IndexIVFPQ(quant, 96, nlist, M, 8, faiss.METRIC_INNER_PRODUCT)
    yl, tot, buf, trained = [], 0, [], False
    t0 = time.time()
    for a in ctx.tr_i:
        if len(a) < 4:
            continue
        w3 = np.stack([a[0:-3], a[1:-2], a[2:-1]], 1)
        buf.append(keys(w3)); yl.append(a[3:].astype(np.int16)); tot += len(a) - 3
        if not trained and sum(len(b) for b in buf) >= 1000000:
            index.train(np.concatenate(buf)[:1000000]); trained = True
        if trained and sum(len(b) for b in buf) >= 1000000:
            index.add(np.concatenate(buf)); buf = []
        if tot >= maxM:
            break
    if buf and trained:
        index.add(np.concatenate(buf))
    yd = torch.from_numpy(np.concatenate(yl)).long()
    log('索引 %d 語 構築 %.0fs' % (index.ntotal, time.time() - t0))
    uni = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1; uni /= uni.sum()
    nq = 20000
    pnv = rn_va['p'][:nq]; pnt = rn_te['p'][:nq]
    res = dict(net_bits=bits(pnt), index_tokens=int(index.ntotal), bytes_per_entry=M, runs={})
    def pk(sc, ix, yq, T):
        w = torch.softmax(torch.from_numpy(sc) / T, 1)
        return (w * (yd[torch.from_numpy(ix)] == torch.from_numpy(yq)[:, None])).sum(1).numpy().astype(np.float64)
    for nprobe in (1, 4, 16):
        index.nprobe = nprobe
        t1 = time.time(); sv, iv = index.search(keys(ctx.Xva[:nq, -3:]), 64); tv = time.time() - t1
        t1 = time.time(); st, it = index.search(keys(ctx.Xte[:nq, -3:]), 64); tt = time.time() - t1
        ok = iv >= 0; iv = np.where(ok, iv, 0); it = np.where(it >= 0, it, 0)
        best = None
        for T in (0.02, 0.05):
            pkv = pk(sv, iv, ctx.yva[:nq], T)
            for lam in (0.4, 0.6, 0.8, 0.9):
                pa = lam * pkv + (1 - lam) * uni[ctx.yva[:nq]]
                for w in (0.05, 0.1, 0.2, 0.3, 0.5):
                    bm = bits((1 - w) * pnv + w * pa)
                    if best is None or bm < best[0]:
                        best = (bm, T, lam, w)
        _, T, lam, w = best
        pa = lam * pk(st, it, ctx.yte[:nq], T) + (1 - lam) * uni[ctx.yte[:nq]]
        mix = bits((1 - w) * pnt + w * pa)
        res['runs']['nprobe=%d' % nprobe] = dict(mix=mix, gain=res['net_bits'] - mix, ms_per_query=1000 * tt / nq,
                                                 lookups_per_query=int(nlist * 96 + nprobe * (index.ntotal / nlist) * M))
        log('nprobe=%d: 混合 %.4f (改善 %.4f) %.2fms/語 検索の積和相当 %d' % (nprobe, mix, res['net_bits'] - mix, 1000 * tt / nq, res['runs']['nprobe=%d' % nprobe]['lookups_per_query']))
    return res


class RNet(torch.nn.Module):
    """検索を学習に組み込んだNN: 近い場面の次の語(最大K個)を、特徴と出力の加点として使う。いつ信じるかをNNが学ぶ"""
    def __init__(s, H, E0, T=0.05):
        super().__init__()
        s.E = torch.nn.Embedding(V, DE); s.E.weight.data.copy_(torch.from_numpy(E0).float())
        s.l1 = torch.nn.Linear((NEAR + 1) * DE + DE + 2, H); s.l2 = torch.nn.Linear(H, V); s.gate = torch.nn.Linear(H, 1)
        s.T = T

    def forward(s, X, Ny, Ns):
        e = s.E(X)
        valid = (Ny >= 0)
        w = torch.softmax(torch.where(valid, Ns / s.T, torch.full_like(Ns, -1e4)), 1) * valid
        f = (w[..., None] * s.E(Ny.clamp_min(0))).sum(1)
        ex = torch.stack([Ns.max(1).values, (Ns * valid).sum(1) / valid.sum(1).clamp_min(1)], 1)
        h = torch.relu(s.l1(torch.cat([e[:, -NEAR:].reshape(len(X), -1), e[:, :-NEAR].mean(1), f, ex], 1)))
        S = torch.zeros(len(X), V).scatter_add(1, Ny.clamp_min(0), w)
        return s.l2(h) + torch.nn.functional.softplus(s.gate(h)) * S


def p9(ctx, midx):
    import faiss
    E0 = items.ppmi_embeddings(ctx, DE)
    E32 = E0[:, :32].astype(np.float32)
    torch.set_num_threads(os.cpu_count() or 2); faiss.omp_set_num_threads(os.cpu_count() or 2)
    t0 = time.time()
    def keys(w3):
        k = E32[w3].reshape(len(w3), -1)
        return (k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
    # 索引は、NNの学習に使う文書より後ろの文書から作る(学習窓が自分自身を引かないように)
    first = int(ctx.mtr.max()) + 1
    quant = faiss.IndexFlatIP(96); index = faiss.IndexIVFPQ(quant, 96, 4096, 16, 8, faiss.METRIC_INNER_PRODUCT)
    yl, tot, buf, trained = [], 0, [], False
    for a in ctx.tr_i[first:]:
        if len(a) < 4:
            continue
        w3 = np.stack([a[0:-3], a[1:-2], a[2:-1]], 1)
        buf.append(keys(w3)); yl.append(a[3:].astype(np.int16)); tot += len(a) - 3
        if not trained and sum(len(b) for b in buf) >= 1000000:
            index.train(np.concatenate(buf)[:1000000]); trained = True
        if trained and sum(len(b) for b in buf) >= 1000000:
            index.add(np.concatenate(buf)); buf = []
        if tot >= midx:
            break
    if buf and trained:
        index.add(np.concatenate(buf))
    index.make_direct_map()
    yd = torch.from_numpy(np.concatenate(yl)).long()
    index.nprobe = 4
    log('索引 %d 語 (学習窓の文書は除く) 構築 %.0fs' % (index.ntotal, time.time() - t0))
    K = 16
    def retrieve(X):
        S_, N_ = [], []
        for i in range(0, len(X), 500000):
            sc, ix = index.search(keys(X[i:i + 500000, -3:]), K)
            ny = np.where(ix >= 0, yd[torch.from_numpy(np.where(ix >= 0, ix, 0))].numpy(), -1)
            S_.append(sc.astype(np.float16)); N_.append(ny.astype(np.int16))
        return torch.from_numpy(np.concatenate(S_)).float(), torch.from_numpy(np.concatenate(N_)).long(), 
    Str, Ntr = retrieve(ctx.Xtr)[0:2]; log('学習窓の検索 完了 %.0fs' % (time.time() - t0))
    Sva, Nva = retrieve(ctx.Xva[:100000])[0:2]; Ste, Nte = retrieve(ctx.Xte)[0:2]
    uni = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1; uni /= uni.sum()
    res = {}
    # 基準: 普通のNN + 事後の混合
    base = lab.train(ctx, 256, 1.0, E0=E0)
    pb_va = lab.forward_all(base, ctx, 'va')['p'][:100000]; pb_te = lab.forward_all(base, ctx, 'te')['p']
    def pk(Ss, Ns, y, T):
        valid = Ns >= 0
        w = torch.softmax(torch.where(valid, Ss / T, torch.full_like(Ss, -1e4)), 1) * valid
        return (w * (Ns == torch.from_numpy(y)[:, None])).sum(1).numpy().astype(np.float64)
    def posthoc(pv, pt):
        best = None
        for T in (0.02, 0.05):
            kv = pk(Sva, Nva, ctx.yva[:100000], T)
            for lam in (0.6, 0.8, 0.9):
                pa = lam * kv + (1 - lam) * uni[ctx.yva[:100000]]
                for w in (0.05, 0.1, 0.2, 0.3, 0.5):
                    bm = bits((1 - w) * pv + w * pa)
                    if best is None or bm < best[0]:
                        best = (bm, T, lam, w)
        _, T, lam, w = best
        pa = lam * pk(Ste, Nte, ctx.yte, T) + (1 - lam) * uni[ctx.yte]
        return bits((1 - w) * pt + w * pa), dict(T=T, lam=lam, w=w), bits(pa)
    res['base_nn'] = bits(pb_te)
    res['base_posthoc'], res['base_posthoc_params'], res['knn_alone'] = posthoc(pb_va, pb_te)
    log('基準 NN %.4f / NN+事後混合 %.4f / 検索だけ %.4f' % (res['base_nn'], res['base_posthoc'], res['knn_alone']))
    # 検索を学習に組み込む
    torch.manual_seed(0)
    net = RNet(256, E0); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr); steps = len(X) // 512; st = 0
    for i in torch.randperm(len(X)).split(512):
        if len(i) < 512:
            break
        for g in opt.param_groups:
            g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
        loss = torch.nn.functional.cross_entropy(net(X[i], Ntr[i], Str[i]), y[i])
        opt.zero_grad(); loss.backward(); opt.step(); st += 1
    def probs(Xa, Na, Sa, ya):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xa), 4096):
                z = net(torch.from_numpy(Xa[i:i + 4096]), Na[i:i + 4096], Sa[i:i + 4096]); lp = torch.log_softmax(z, 1)
                out.append(lp[torch.arange(len(z)), torch.from_numpy(ya[i:i + 4096])].exp().numpy())
        return np.concatenate(out).astype(np.float64)
    pr_va = probs(ctx.Xva[:100000], Nva, Sva, ctx.yva[:100000]); pr_te = probs(ctx.Xte, Nte, Ste, ctx.yte)
    res['loop_nn'] = bits(pr_te)
    res['loop_posthoc'], res['loop_posthoc_params'], _ = posthoc(pr_va, pr_te)
    log('検索を学習に組み込んだNN %.4f / さらに事後混合 %.4f (基準の最良 %.4f)' % (res['loop_nn'], res['loop_posthoc'], res['base_posthoc']))
    # カーネルリッジの重み(近傍同士の似かよりを補正)
    def ridge_pk(Xq, Sq, Nq, y, lam):
        out = []
        for i in range(0, len(Xq), 2000):
            ids = np.zeros((min(2000, len(Xq) - i), K), np.int64)
            sc, ix = index.search(keys(Xq[i:i + 2000, -3:]), K)
            B = len(ix); ok = ix >= 0
            vec = np.stack([index.reconstruct_batch(np.where(ok[:, j], ix[:, j], 0)) for j in range(K)], 1)
            q = torch.from_numpy(keys(Xq[i:i + 2000, -3:]))
            V_ = torch.from_numpy(vec)
            G = V_ @ V_.transpose(1, 2); s_ = (V_ @ q[..., None])[..., 0]
            a = torch.linalg.solve(G + lam * torch.eye(K)[None], s_[..., None])[..., 0].clamp_min(0)
            a = a * torch.from_numpy(ok)
            a = a / a.sum(1, keepdim=True).clamp_min(1e-9)
            ny = torch.from_numpy(np.where(ok, yd[torch.from_numpy(np.where(ok, ix, 0))].numpy(), -1))
            out.append((a * (ny == torch.from_numpy(y[i:i + 2000])[:, None])).sum(1).numpy().astype(np.float64))
        return np.concatenate(out)
    nq = 30000
    for lam in (0.1, 0.5, 2.0):
        pa = 0.9 * ridge_pk(ctx.Xte[:nq], None, None, ctx.yte[:nq], lam) + 0.1 * uni[ctx.yte[:nq]]
        pa_soft = 0.9 * pk(Ste[:nq], Nte[:nq], ctx.yte[:nq], 0.05) + 0.1 * uni[ctx.yte[:nq]]
        res['ridge_lam=%s' % lam] = dict(knn_alone_ridge=bits(pa), knn_alone_softmax=bits(pa_soft),
                                          mix_ridge=bits(0.8 * pb_te[:nq] + 0.2 * pa), mix_softmax=bits(0.8 * pb_te[:nq] + 0.2 * pa_soft))
        log('ridge lam=%s %s' % (lam, res['ridge_lam=%s' % lam]))
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
    elif mode == 'p6':
        res = p6(ctx, int(arg))
    elif mode == 'p7':
        res = p7(ctx, int(arg), tuple(int(h) for h in os.environ['HS'].split(',')) if os.environ.get('HS') else (32, 64, 128, 256, 512))
    elif mode == 'p8':
        res = p8(ctx, int(arg))
    elif mode == 'p9':
        res = p9(ctx, int(arg))
    res['mode'] = mode; res['N'] = N; res['sec'] = time.time() - t0
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
