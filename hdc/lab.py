"""
自動実験の共通部品。全ての施策の試験は、同じデータ・同じ基準(小さなNN)の上で、差分だけを測る。
  load(N)        : データ(語の窓=直前12語)を用意
  train(ctx,...) : 小さなNN(直前8語は位置つき+その前の4語は平均)を学習。施策ごとの差し込み口つき
  evaluate(...)  : 確率・bit/語
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
import numpy as np
import torch
import exp_scale as es
import exp_far as XF

V = es.V; HIST = XF.HIST; DE = 64; NEAR = 8


def log(s):
    print(s, flush=True)


class Ctx:
    pass


def load(N):
    c = Ctx()
    corp = es.build_corpus()
    cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    c.tr_i, c.va_i, c.te_i = conv(corp[2]), conv(corp[1]), conv(corp[0])
    c.Xtr, c.ytr, c.mtr, c.ptr = XF.lwindows_meta(c.tr_i, N)
    c.Xva, c.yva, c.mva, c.pva = XF.lwindows_meta(c.va_i, es.N_VAL)
    c.Xte, c.yte, c.mte, c.pte = XF.lwindows_meta(c.te_i, es.N_TEST)
    c.N = N
    return c


def heads(ctx, split):
    """各窓が属する文書の、先頭12語(文書の設定を残す枠)"""
    docs = {'tr': ctx.tr_i, 'va': ctx.va_i, 'te': ctx.te_i}[split]
    mid = {'tr': ctx.mtr, 'va': ctx.mva, 'te': ctx.mte}[split]
    H = np.zeros((len(mid), HIST), np.int64)
    for m in np.unique(mid):
        h = docs[m][:HIST]
        H[mid == m, :len(h)] = h
    return H


class Net(torch.nn.Module):
    def __init__(s, H=256, head=False, E0=None):
        super().__init__()
        s.head = head
        s.E = torch.nn.Embedding(V, DE)
        if E0 is not None:
            s.E.weight.data.copy_(torch.from_numpy(E0).float())
        s.l1 = torch.nn.Linear((NEAR + 1 + (1 if head else 0)) * DE, H)
        s.l2 = torch.nn.Linear(H, V)
        s.H = H

    def forward(s, X, Hd=None):
        e = s.E(X)
        parts = [e[:, -NEAR:].reshape(len(X), -1), e[:, :-NEAR].mean(1)]
        if s.head:
            parts.append(s.E(Hd).mean(1))
        return s.l2(torch.relu(s.l1(torch.cat(parts, 1))))

    def macs(s):
        return s.l1.in_features * s.H + s.H * V


def train(ctx, H=256, ep=1.0, head=False, E0=None, idx=None, net=None, lr=2e-3, seed=0, bs=512):
    torch.manual_seed(seed)
    torch.set_num_threads(os.cpu_count() or 2)
    net = net or Net(H, head, E0)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
    Hd = torch.from_numpy(heads(ctx, 'tr')) if head else None
    n = len(X) if idx is None else len(idx)
    steps = int(ep * (n // bs)); st = 0
    while st < steps:
        perm = torch.randperm(n)
        if idx is not None:
            perm = torch.from_numpy(idx)[perm]
        for i in perm.split(bs):
            if len(i) < bs or st >= steps:
                break
            for g in opt.param_groups:
                g['lr'] = lr if st < 0.6 * steps else lr / 2
            loss = torch.nn.functional.cross_entropy(net(X[i], Hd[i] if head else None), y[i])
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
    return net


def forward_all(net, ctx, split, head=False, bsz=4096):
    """真の語の確率、最大確率、エントロピー、1位が正解か"""
    X, y = {'tr': (ctx.Xtr, ctx.ytr), 'va': (ctx.Xva, ctx.yva), 'te': (ctx.Xte, ctx.yte)}[split]
    Hd = heads(ctx, split) if head else None
    pt, mx, en, ok = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bsz):
            xb = torch.from_numpy(X[i:i + bsz]); yb = torch.from_numpy(y[i:i + bsz])
            z = net(xb, torch.from_numpy(Hd[i:i + bsz]) if head else None)
            lp = torch.log_softmax(z, 1); p = lp.exp()
            pt.append(p[torch.arange(len(z)), yb].numpy()); mx.append(p.max(1).values.numpy())
            en.append(-(p * lp).sum(1).numpy()); ok.append((p.argmax(1) == yb).numpy())
    return dict(p=np.concatenate(pt).astype(np.float64), maxp=np.concatenate(mx), ent=np.concatenate(en),
                ok=np.concatenate(ok), loss=-np.log2(np.concatenate(pt) + 1e-12))


def bits(p):
    return float(-np.log2(p + 1e-12).mean())


def baseline(ctx, H=256):
    t0 = time.time()
    net = train(ctx, H)
    r = forward_all(net, ctx, 'te')
    return net, r, bits(r['p']), time.time() - t0
