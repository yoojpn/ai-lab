"""
施策ごとの試験。どれも lab.py の同じデータ・同じ基準NNの上で、差分だけを測る。
各関数は dict(結果) を返す。  番号は「1〜41の施策」の番号。
"""
import time, json, math
import numpy as np
import torch
import lab
from lab import *


def auc(score, pos):
    """scoreが高いほど pos(=True) になりやすい、のAUROC"""
    o = np.argsort(score, kind='mergesort'); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    n1 = pos.sum(); n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def i06_cascade(ctx):
    """#6 簡単な語は小さなモデル、難しい語だけ大きなモデル(段階計算)"""
    big, rb, bb, _ = baseline(ctx, 256)
    tiny = train(ctx, 32); rt = forward_all(tiny, ctx, 'te')
    out = dict(big_bits=bb, tiny_bits=bits(rt['p']), macs_big=big.macs(), macs_tiny=tiny.macs(), rows=[])
    for tau in (0.2, 0.3, 0.5, 0.7, 0.9):
        use = rt['maxp'] > tau
        p = np.where(use, rt['p'], rb['p']); f = float(use.mean())
        macs = tiny.macs() + (1 - f) * big.macs()
        out['rows'].append(dict(tau=tau, tiny_share=f, bits=bits(p), macs=macs, saving=big.macs() / macs))
    return out


def i07_quant(ctx):
    """#7 重みを少ないビットで持つ"""
    net, r, b0, _ = baseline(ctx, 256)
    base = {k: v.clone() for k, v in net.state_dict().items()}
    params = sum(p.numel() for p in net.parameters())
    rows = [dict(bits=32, mb=params * 4 / 1e6, bpw=b0)]
    for nb in (8, 6, 4, 3, 2):
        sd = {k: v.clone() for k, v in base.items()}
        for k in ('E.weight', 'l1.weight', 'l2.weight'):
            w = sd[k]; q = 2 ** (nb - 1) - 1
            sc = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / q
            sd[k] = torch.round(w / sc).clamp(-q, q) * sc
        net.load_state_dict(sd)
        rows.append(dict(bits=nb, mb=params * nb / 8 / 1e6, bpw=bits(forward_all(net, ctx, 'te')['p'])))
    return dict(rows=rows)


def ppmi_embeddings(ctx, k=DE):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    C = np.zeros(V * V, np.float64)
    X = ctx.Xtr
    for d in (1, 2):
        for j in range(X.shape[1] - d):
            C += np.bincount(X[:, j] * V + X[:, j + d], minlength=V * V)
    C = C.reshape(V, V); C = C + C.T
    rs, cs, tot = C.sum(1) + 1e-9, C.sum(0) + 1e-9, C.sum()
    with np.errstate(divide='ignore'):
        P = np.maximum(np.log(C * tot / (rs[:, None] * cs[None, :]) + 1e-12), 0)
    U, S, _ = svds(csr_matrix(P), k=k)
    E = U * np.sqrt(S)
    return E / E.std() * 0.1


def i10_embed_init(ctx):
    """#10 近い意味の語を近くに置いた埋め込みで始める(学習が速く・良くなるか)"""
    E0 = ppmi_embeddings(ctx)
    rows = []
    for ep in (0.1, 0.25, 1.0):
        a = bits(forward_all(train(ctx, 256, ep=ep), ctx, 'te')['p'])
        b = bits(forward_all(train(ctx, 256, ep=ep, E0=E0), ctx, 'te')['p'])
        rows.append(dict(epochs=ep, random_init=a, ppmi_init=b))
    return dict(rows=rows)


def i21_head_frame(ctx):
    """#21 文書の冒頭(設定)を、最後まで枠として残す"""
    _, _, b0, _ = baseline(ctx, 256)
    net = train(ctx, 256, head=True)
    return dict(base=b0, head_frame=bits(forward_all(net, ctx, 'te', head=True)['p']))


def i25_calibration(ctx):
    """#25 自信の低さから、間違いを見分けられるか(「不確か」と言えるか)"""
    net, r, b0, _ = baseline(ctx, 256)
    ok = r['ok']
    ece, edges = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r['maxp'] > lo) & (r['maxp'] <= hi)
        if m.any():
            ece += m.mean() * abs(r['maxp'][m].mean() - ok[m].mean())
    o = np.argsort(-r['maxp'])
    cov = []
    for q in (1.0, 0.8, 0.6, 0.4, 0.2):
        s = o[:int(q * len(o))]; cov.append(dict(answered=q, accuracy=float(ok[s].mean())))
    return dict(bits=b0, top1_acc=float(ok.mean()), auc_maxp=auc(r['maxp'], ok), auc_entropy=auc(-r['ent'], ok), ece=ece, coverage=cov)


def i29_surprise_gate(ctx):
    """#29 驚いた語だけで学習すれば、少ない学習量で同じ精度になるか"""
    S = len(ctx.Xtr) // 512
    full = train(ctx, 256, ep=1.0)
    b_full = bits(forward_all(full, ctx, 'te')['p'])
    warm = train(ctx, 256, ep=0.3)
    state = {k: v.clone() for k, v in warm.state_dict().items()}
    # 学習データ全体の損失(順伝播だけ)
    tmp = forward_all(warm, ctx, 'tr')
    loss = tmp['loss']
    q = 0.4; n = int(q * len(loss))
    hard = np.argsort(-loss)[:n]
    rnd = np.random.default_rng(0).permutation(len(loss))[:n]
    res = dict(full_bits=b_full, full_steps=S, warm_bits=bits(forward_all(warm, ctx, 'te')['p']))
    for name, idx in (('hard40', hard), ('random40', rnd)):
        net = Net_copy(state)
        net = train(ctx, 256, ep=1.0, idx=idx.astype(np.int64), net=net, lr=1e-3)
        res[name] = bits(forward_all(net, ctx, 'te')['p'])
    res['gated_steps'] = int(0.3 * S + q * S)
    return res


def Net_copy(state):
    n = lab.Net(256); n.load_state_dict(state); return n


class TinyTF(torch.nn.Module):
    def __init__(s, d=64, L=2, dff=256, heads=4):
        super().__init__()
        s.d, s.L, s.dff = d, L, dff
        s.E = torch.nn.Embedding(V, d); s.pos = torch.nn.Parameter(torch.zeros(HIST, d))
        layer = torch.nn.TransformerEncoderLayer(d, heads, dff, dropout=0.0, batch_first=True, norm_first=True)
        s.enc = torch.nn.TransformerEncoder(layer, L)
        s.out = torch.nn.Linear(d, V)

    def forward(s, X, Hd=None):
        h = s.enc(s.E(X) + s.pos)
        return s.out(h[:, -1])

    def macs(s):
        return HIST * s.L * (4 * s.d * s.d + 2 * s.d * s.dff) + s.L * 2 * HIST * HIST * s.d + s.d * V


def i41_transformer(ctx):
    """#41 同じ計算量で、小さなTransformerと並べる"""
    t0 = time.time(); mlp, r, b0, _ = baseline(ctx, 256); t_mlp = time.time() - t0
    t0 = time.time()
    torch.manual_seed(0)
    tf = TinyTF()
    opt = torch.optim.Adam(tf.parameters(), lr=2e-3)
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
    steps = len(X) // 512; st = 0
    for i in torch.randperm(len(X)).split(512):
        if len(i) < 512:
            break
        for g in opt.param_groups:
            g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
        loss = torch.nn.functional.cross_entropy(tf(X[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step(); st += 1
    pt = []
    with torch.no_grad():
        for i in range(0, len(ctx.Xte), 4096):
            z = tf(torch.from_numpy(ctx.Xte[i:i + 4096])); lp = torch.log_softmax(z, 1)
            pt.append(lp[torch.arange(len(z)), torch.from_numpy(ctx.yte[i:i + 4096])].exp().numpy())
    b1 = bits(np.concatenate(pt).astype(np.float64))
    return dict(mlp_bits=b0, mlp_macs=mlp.macs(), mlp_params=sum(p.numel() for p in mlp.parameters()), mlp_sec=t_mlp,
                tf_bits=b1, tf_macs=tf.macs(), tf_params=sum(p.numel() for p in tf.parameters()), tf_sec=time.time() - t0)


ITEMS = {'06': i06_cascade, '07': i07_quant, '10': i10_embed_init, '21': i21_head_frame,
         '25': i25_calibration, '29': i29_surprise_gate, '41': i41_transformer}
