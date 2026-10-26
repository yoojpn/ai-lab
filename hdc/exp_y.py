"""
Y: 構造の案(MoE, 積キーメモリ層, 3-gram検索特徴)を exp_x と同じ土台で測る。
 X(元の説明): 「表の案」を、同じ土台(d128, 4層, 文脈64語, 1600万語, 3エポック, lr4e-3)で1か所ずつ変えて測る。
 opt   : adamw | muon (層の2次元重みだけMuon: Newton-Schulz5回, 更新量をAdamWと同じ大きさに合わせる)
 head  : full | hier (語を頻度順に71クラスに分け p(クラス)*p(語|クラス)) | samp (負例K個の標本softmax。評価は全語)
 hemb  : 0 | B  (直前語+今の語のハッシュ(2^B個)の埋め込みを足す=ハッシュ化n-gram埋め込み)
 soft  : ε  正解を (1-ε)*one-hot + ε*(2-gramの分布) に置き換える
 sched : cos | wsd(後ろ20%で線形に下げる)、ema : 重みの指数移動平均の減衰(0なら無し)
 rho   : 0 | keep  (数える2-gramとの損失差が大きい語だけ keep割 を学習に使う。FLOPは全語を順伝播・逆伝播するので同じ)
 ind   : 1 なら誘導特徴を足す(いま見ている語が文脈内で前に出た位置の「次の語」を埋め込みで足す。勾配なしの配線)
 ffn   : dense | moe(E専門家, top1, 有効FLOPは同じ) | pkm(層1,2のFFNを積キーメモリに置換)
 ret   : 1 なら直前2語の次語の最頻値(3-gram, 自分を除く)を特徴として足す
 init  : ppmi なら埋め込みを共起のSVD(PPMI)で初期化
FLOP: 順+逆=6*積和。headごとに出力層の積和を変える(hier: d*(C+Vc), samp: d*(K+1))。
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import lab, items
from lab import V, bits, log
import exp_scale as es
from exp_t3 import split_windows
from exp_t4 import GT
from exp_r import Prior
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)
C = 71; VC = -(-V // C)


def ind_feature(x):
    """x:(b,n) → 各位置tで、x_t が前(j<t)に最後に出た位置jの次の語 x_{j+1}。なければ0"""
    n = x.shape[1]
    eq = (x[:, :, None] == x[:, None, :]) & torch.tril(torch.ones(n, n, dtype=torch.bool), -1)[None]
    ar = torch.arange(1, n + 1)[None, None, :]
    idx = (eq * ar).max(-1).values                      # j+1 (なければ0)
    f = x.gather(1, idx.clamp(max=n - 1))
    return torch.where(idx > 0, f, torch.zeros_like(f))


class Block(torch.nn.Module):
    def __init__(s, d, heads, ffn):
        super().__init__()
        s.ln1 = torch.nn.LayerNorm(d); s.ln2 = torch.nn.LayerNorm(d)
        s.attn = torch.nn.MultiheadAttention(d, heads, batch_first=True); s.ffn = ffn

    def forward(s, h, src_mask=None, is_causal=False):
        x = s.ln1(h); h = h + s.attn(x, x, x, attn_mask=src_mask, need_weights=False)[0]
        return h + s.ffn(s.ln2(h))


class DenseFFN(torch.nn.Module):
    def __init__(s, d):
        super().__init__(); s.a = torch.nn.Linear(d, 4 * d); s.b = torch.nn.Linear(4 * d, d)

    def forward(s, x): return s.b(torch.relu(s.a(x)))


class MoEFFN(torch.nn.Module):
    def __init__(s, d, E=8):
        super().__init__(); s.E = E; s.r = torch.nn.Linear(d, E); s.ex = torch.nn.ModuleList([DenseFFN(d) for _ in range(E)]); s.aux = 0.0

    def forward(s, x):
        sh = x.shape; x = x.reshape(-1, sh[-1]); pr = torch.softmax(s.r(x), 1); g, a = pr.max(1)
        out = torch.zeros_like(x)
        for e in range(s.E):
            ix = (a == e).nonzero(as_tuple=True)[0]
            if len(ix): out[ix] = g[ix, None] * s.ex[e](x[ix])
        f = torch.bincount(a, minlength=s.E).float() / len(a); s.aux = s.E * (f * pr.mean(0)).sum()
        return out.reshape(sh)


class PKM(torch.nn.Module):
    def __init__(s, d, nk=128, heads=4, dk=32, topk=8):
        super().__init__(); s.nk, s.h, s.dk, s.k = nk, heads, dk, topk
        s.q = torch.nn.Linear(d, heads * dk * 2); s.K1 = torch.nn.Parameter(torch.randn(nk, dk) * 0.1); s.K2 = torch.nn.Parameter(torch.randn(nk, dk) * 0.1)
        s.Vv = torch.nn.EmbeddingBag(nk * nk, d, mode='sum'); torch.nn.init.normal_(s.Vv.weight, std=d ** -0.5)

    def forward(s, x):
        sh = x.shape; x = x.reshape(-1, sh[-1]); n = len(x); q = s.q(x).reshape(n, s.h, 2, s.dk)
        s1 = torch.einsum('nhd,kd->nhk', q[:, :, 0], s.K1); s2 = torch.einsum('nhd,kd->nhk', q[:, :, 1], s.K2)
        v1, i1 = s1.topk(s.k, -1); v2, i2 = s2.topk(s.k, -1)
        sc = (v1[..., :, None] + v2[..., None, :]).reshape(n, s.h, -1); ix = (i1[..., :, None] * s.nk + i2[..., None, :]).reshape(n, s.h, -1)
        v, j = sc.topk(s.k, -1); ids = ix.gather(-1, j); w = torch.softmax(v, -1)
        out = s.Vv(ids.reshape(n, -1), per_sample_weights=w.reshape(n, -1))
        return out.reshape(sh)


def build_layers(d, nl, kind, heads=4):
    L = []
    for i in range(nl):
        f = DenseFFN(d)
        if kind == 'moe': f = MoEFFN(d)
        if kind == 'pkm' and i in (1, 2): f = PKM(d)
        L.append(Block(d, heads, f))
    return torch.nn.ModuleList(L)


def build_ret(stream):
    """直前2語(prev,cur)→最頻の次語top1とその回数c1。 返り値(uc, top1, c1)"""
    st = stream.astype(np.int64); code = st[:-2] * V + st[1:-1]; nxt = st[2:]
    up, pc = np.unique(code * V + nxt, return_counts=True)
    c = up // V; o = np.lexsort((-pc, c)); c = c[o]; up = up[o]; pc = pc[o]
    first = np.r_[True, c[1:] != c[:-1]]
    return c[first], (up[first] % V).astype(np.int64), pc[first].astype(np.int64)


def ret_feat(tab, X, Y=None):
    """X:(m,n)入力語, Y:(m,n)その次の語(学習時のleave-one-out用)。 返り値 (tok, bucket) 各(m,n)"""
    uc, top, c1 = tab
    prev = np.pad(X[:, :-1], ((0, 0), (1, 0))); code = (prev * V + X).reshape(-1)
    i = np.minimum(np.searchsorted(uc, code), len(uc) - 1); ok = uc[i] == code
    t = np.where(ok, top[i], 0); c = np.where(ok, c1[i], 0)
    if Y is not None:
        hit = ok & (t == Y.reshape(-1)); c = c - hit
    t = np.where(c > 0, t, 0); b = np.minimum(c, 3)
    return torch.from_numpy(t.reshape(X.shape)), torch.from_numpy(b.reshape(X.shape))


class XT(GT):
    def __init__(s, d, nl, seq, cfg, freq_rank, q=None):
        super().__init__(d, nl, seq)
        s.head = cfg.get('head', 'full'); s.hb = cfg.get('hemb', 0); s.ind = cfg.get('ind', 0); s.ret = cfg.get('ret', 0)
        if cfg.get('ffn', 'dense') != 'dense': s.layers = build_layers(d, nl, cfg['ffn'])
        if s.ret: s.Er = torch.nn.Embedding(V, d); s.Eb = torch.nn.Embedding(4, d); torch.nn.init.normal_(s.Er.weight, std=0.02); torch.nn.init.normal_(s.Eb.weight, std=0.02)
        if s.hb: s.H = torch.nn.Embedding(1 << s.hb, d); torch.nn.init.normal_(s.H.weight, std=0.02)
        if s.ind: s.Ei = torch.nn.Embedding(V, d); torch.nn.init.normal_(s.Ei.weight, std=0.02)
        if s.head == 'hier':
            s.Wc = torch.nn.Linear(d, C); s.Ww = torch.nn.Parameter(torch.randn(C, VC, d) * 0.02); s.bw = torch.nn.Parameter(torch.zeros(C, VC))
            s.register_buffer('cls', torch.from_numpy(freq_rank // VC)); s.register_buffer('pos_in', torch.from_numpy(freq_rank % VC))
        if s.head == 'samp':
            s.register_buffer('q', torch.from_numpy(q))

    def hidden(s, X, fr=None):
        n = X.shape[1]; mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
        h = s.E(X) + s.pos[:n]
        if s.hb:
            prev = F.pad(X[:, :-1], (1, 0)); idx = ((prev * 1000003 + X) * 2654435761) % (1 << s.hb)
            h = h + s.H(idx)
        if s.ind: h = h + s.Ei(ind_feature(X))
        if s.ret: h = h + s.Er(fr[0]) + s.Eb(fr[1])
        for L in s.layers[:s.active]:
            h = L(h, src_mask=mask, is_causal=True)
        return h

    def logp_true(s, h, y):
        if s.head == 'hier':
            lc = torch.log_softmax(s.Wc(h), 1); c = s.cls[y]; p = s.pos_in[y]
            lw = torch.log_softmax(torch.einsum('nd,nvd->nv', h, s.Ww[c]) + s.bw[c], 1)
            return lc[torch.arange(len(y)), c] + lw[torch.arange(len(y)), p]
        return torch.log_softmax(s.out(h), 1)[torch.arange(len(y)), y]


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float(); X = X / (X.norm() + 1e-7); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


class Muon:
    def __init__(s, params, lr, mom=0.95):
        s.p = params; s.lr = lr; s.mom = mom; s.buf = [torch.zeros_like(p) for p in params]

    def step(s):
        for p, bf in zip(s.p, s.buf):
            if p.grad is None: continue
            bf.mul_(s.mom).add_(p.grad); g = p.grad + s.mom * bf
            u = newton_schulz(g) * 0.2 * math.sqrt(max(p.shape))
            p.data.add_(u.to(p.dtype), alpha=-s.lr)

    def zero_grad(s):
        for p in s.p: p.grad = None


def run(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    ctx = lab.load(cfg['N']); d, nl, seq = cfg.get('d', 128), cfg.get('nl', 4), cfg.get('seq', 64); S = seq + 1
    bs = cfg.get('bs', 128); lr = cfg.get('lr', 4e-3); head = cfg.get('head', 'full'); eps = cfg.get('soft', 0.0)
    rho = cfg.get('rho', 0.0); sched = cfg.get('sched', 'cos'); emad = cfg.get('ema', 0.0); K = cfg.get('K', 512)
    stream = np.concatenate(ctx.tr_i)[:cfg['N']]
    M = len(stream) // S; chunks = torch.from_numpy(stream[:M * S].reshape(M, S).astype(np.int64))
    if cfg.get('dedup'):
        st = stream.astype(np.uint64); P = np.uint64(1000003); h = np.zeros(len(st) - 7, np.uint64)
        for j in range(8): h = h * P + st[j:len(st) - 7 + j] + np.uint64(1)
        u, inv, c = np.unique(h, return_inverse=True, return_counts=True); dup = np.r_[(c[inv] > 1), np.zeros(7, bool)]
        fr = dup[:M * S].reshape(M, S).mean(1); keep = fr <= cfg['dedup']
        log('重複8-gram位置 %.4f | 重複の多い塊(>%.2f) %.4f を除く' % (dup.mean(), cfg['dedup'], 1 - keep.mean()))
        chunks = chunks[torch.from_numpy(keep)]; M = len(chunks)
    cnt = np.bincount(stream, minlength=V).astype(np.float64)
    order = np.argsort(-cnt, kind='stable'); rank = np.empty(V, np.int64); rank[order] = np.arange(V)
    qd = ((cnt + 1) ** 0.75); qd = (qd / qd.sum()).astype(np.float32)
    net = XT(d, nl, S, cfg, rank, qd)
    if cfg.get('resid'): net.a = torch.nn.Parameter(torch.ones(1))
    if cfg.get('init') == 'ppmi':
        E0 = items.ppmi_embeddings(ctx, d); E0 = E0 / (E0.std() + 1e-9)
        net.E.weight.data.copy_(torch.from_numpy(E0.astype(np.float32)))
    pr = Prior(stream) if (eps > 0 or rho > 0 or cfg.get('resid')) else None
    FRtr = None; FRte = None
    if cfg.get('ret'):
        tab = build_ret(stream); xs = chunks[:, :-1].numpy(); ys = chunks[:, 1:].numpy()
        FRtr = ret_feat(tab, xs, ys); FRte = ret_feat(tab, split_windows(ctx.te_i, ctx.mte, ctx.pte, 20000, seq))
        log('ret特徴 作成 (自分を除く3-gramの最頻次語) 有効率 %.3f' % (FRtr[0] > 0).float().mean())
    if cfg.get('opt', 'adamw') == 'muon':
        mp = [p for n_, p in net.layers.named_parameters() if p.ndim == 2]; mid = {id(p) for p in mp}
        rest = [p for p in net.parameters() if id(p) not in mid]
        opts = [Muon(mp, lr), torch.optim.AdamW(rest, lr=lr, weight_decay=0.0)]
    else:
        if cfg.get('mup'):
            hid = [p for p in net.layers.parameters() if p.ndim == 2]; hid_id = {id(p) for p in hid}
            oth = [p for p in net.parameters() if id(p) not in hid_id]
            opts = [torch.optim.AdamW([{'params': hid, 'scale': 128.0 / d}, {'params': oth, 'scale': 1.0}], lr=lr, weight_decay=0.0)]
        else:
            opts = [torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)]
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()} if emad else None
    params = sum(p.numel() for p in net.parameters())
    ffn = cfg.get('ffn', 'dense'); body = nl * (12 * d * d + seq * d)
    if ffn == 'moe': body += nl * d * 8
    if ffn == 'pkm': body += 2 * (32 * 4 * 2 * d + 4 * 2 * 128 * 32 + 8 * 4 * d - 8 * d * d)
    macs = body + {'full': d * V, 'hier': d * (C + VC), 'samp': d * (K + 1)}[head]
    steps = int(cfg.get('epochs', 3) * (M // bs)); nev = 10; every = max(1, steps // nev)
    yte = ctx.yte[:20000]; Xte = torch.from_numpy(split_windows(ctx.te_i, ctx.mte, ctx.pte, len(yte), seq)); yt = torch.from_numpy(yte)
    curve = []; flops = 0.0; st = 0; ttrain = 0.0

    def evaluate():
        net.eval(); backup = None
        if ema is not None:
            backup = {k: v.clone() for k, v in net.state_dict().items()}; net.load_state_dict(ema)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xte), 1024):
                fr = None if FRte is None else (FRte[0][i:i + 1024], FRte[1][i:i + 1024])
                h = net.hidden(Xte[i:i + 1024], fr)[:, -1]
                if cfg.get('resid'):
                    z = net.out(h) + net.a * pr.logp(Xte[i:i + 1024, -1])
                    out.append(torch.log_softmax(z, 1)[torch.arange(len(z)), yt[i:i + 1024]].exp().numpy())
                else:
                    out.append(net.logp_true(h, yt[i:i + 1024]).exp().numpy())
        if backup is not None: net.load_state_dict(backup)
        net.train(); return bits(np.concatenate(out).astype(np.float64))
    log('params %d macs/語 %.3e chunks %d steps %d' % (params, macs, M, steps))
    while st < steps:
        for i in torch.randperm(M).split(bs):
            if len(i) < bs or st >= steps: break
            t = time.time(); fr = st / steps; w = min(1, (st + 1) / 300)
            f = (0.5 * (1 + math.cos(math.pi * fr)) * 0.95 + 0.05) if sched == 'cos' else (1.0 if fr < 0.8 else max(0.05, 1 - (fr - 0.8) / 0.2))
            for o in opts:
                if isinstance(o, Muon): o.lr = lr * w * f
                else:
                    for g in o.param_groups: g['lr'] = lr * w * f * g.get('scale', 1.0)
            b = chunks[i]; x, y = b[:, :-1], b[:, 1:]
            fr = None if FRtr is None else (FRtr[0][i], FRtr[1][i])
            h = net.hidden(x, fr).reshape(-1, d); yf = y.reshape(-1)
            if head == 'samp':
                neg = torch.multinomial(net.q, K, replacement=True)
                zt = (h * net.out.weight[yf]).sum(1) + net.out.bias[yf] - torch.log(net.q[yf])
                zn = h @ net.out.weight[neg].T + net.out.bias[neg] - torch.log(net.q[neg])
                zn = zn.masked_fill(neg[None, :] == yf[:, None], -1e9)
                tok = torch.logsumexp(torch.cat([zt[:, None], zn], 1), 1) - zt
            elif head == 'hier':
                tok = -net.logp_true(h, yf)
            else:
                lg = net.out(h)
                if cfg.get('resid'): lg = lg + net.a * pr.logp(x.reshape(-1), yf)
                if eps > 0:
                    pb = pr.logp(x.reshape(-1), yf).exp()
                    tok = -(((1 - eps) * F.one_hot(yf, V).float() + eps * pb) * torch.log_softmax(lg, 1)).sum(1)
                else:
                    tok = F.cross_entropy(lg, yf, reduction='none')
            if rho > 0:
                with torch.no_grad():
                    ref = -pr.logp(x.reshape(-1), yf)[torch.arange(len(yf)), yf]
                    ex = tok.detach() - ref; k = max(1, int(len(ex) * rho)); thr = torch.topk(ex, k).values[-1]; msk = (ex >= thr).float()
                loss = (tok * msk).sum() / msk.sum()
            else:
                loss = tok.mean()
            if ffn == 'moe': loss = loss + 0.01 * sum(l.ffn.aux for l in net.layers)
            for o in opts: o.zero_grad()
            loss.backward()
            for o in opts: o.step()
            st += 1; ttrain += time.time() - t; flops += 6.0 * macs * bs * seq
            if ema is not None:
                dec = min(emad, (1 + st) / (10 + st))
                with torch.no_grad():
                    for k_, v in net.state_dict().items():
                        if v.dtype.is_floating_point: ema[k_].mul_(dec).add_(v, alpha=1 - dec)
            if st % every == 0 or st == steps:
                bb = evaluate(); curve.append((st, round(ttrain, 1), float('%.3e' % flops), round(bb, 4)))
                log('step %d 学習時間 %.0fs FLOP %.2e bit/語 %.4f' % (st, ttrain, flops, bb))
    res = dict(cfg=cfg, params=params, curve=curve, final_bits=curve[-1][3], flops=flops, train_sec=ttrain)
    log('RESULT ' + json.dumps(res))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
