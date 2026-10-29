"""
HY2: 表(CB2, 勾配なし) + 小さな密Transformer(勾配あり)。 表が記憶を持つので、骨格は小さくて済むか?
  D   : 小さなTransformerだけ (勾配降下, AdamW, 2エポック)
  T   : 表だけ (CB2 の +all_gated)
  TD  : 表 + Transformer の log 確率 (混ぜる重み数十個だけ L-BFGS、ゲートつき)
 比べる量: bit/語、語あたり推論MAC、学習FLOP (6 x MAC x 学習語数)
 参照 (密LLM d128, 4層, 16M語, 2エポック): 5.130 bit, MAC 1.46e6, 学習 2.8e14 FLOP
使い方: LADDER_CORPUS=... python3 exp_hy2.py '{"N":16000000,"d":64,"nl":4,"epochs":2}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
from exp_cb import V, SEQ, LN2, BUCKETS, log


class GT(torch.nn.Module):
    def __init__(s, d, nl, seq, heads=4):
        super().__init__()
        s.E = torch.nn.Embedding(V, d); s.pos = torch.nn.Parameter(torch.zeros(seq, d))
        s.layers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True) for _ in range(nl)])
        s.out = torch.nn.Linear(d, V)

    def forward(s, X, last_only=False):
        n = X.shape[1]; mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
        h = s.E(X) + s.pos[:n]
        for L in s.layers:
            h = L(h, src_mask=mask, is_causal=True)
        return s.out(h[:, -1] if last_only else h)


def dense_macs(d, nl, seq=64):
    return nl * (12 * d * d + seq * d) + d * V


def train_dense(x, d, nl, epochs, bs, lr):
    xs = torch.from_numpy(x.astype(np.int64)); S = SEQ + 1; M = len(xs) // S
    chunks = xs[:M * S].view(M, S)
    net = GT(d, nl, SEQ)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    per = M // bs; steps = epochs * per; warm = 100; step = 0; g = torch.Generator().manual_seed(0)
    t0 = time.time(); run = 0.0
    for ep in range(epochs):
        perm = torch.randperm(M, generator=g)
        for i in range(per):
            b = chunks[perm[i * bs:(i + 1) * bs]]
            lr_t = lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))
            for gr in opt.param_groups:
                gr['lr'] = lr_t
            loss = Fn.cross_entropy(net(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            run = 0.98 * run + 0.02 * loss.item() if step else loss.item(); step += 1
            if step % 200 == 0:
                log('学習 step %d / %d, loss %.3f bit (移動平均), %.0f 秒' % (step, steps, run / LN2, time.time() - t0))
    return net, time.time() - t0, steps * bs * SEQ


def dense_logp(net, X, bs=1000):
    net.eval(); out = []
    with torch.no_grad():
        for a in range(0, len(X), bs):
            out.append(torch.log_softmax(net(X[a:a + bs], last_only=True), 1))
    net.train()
    return torch.cat(out)


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000)
    d, nl, epochs, bs, lr = cfg['d'], cfg['nl'], cfg.get('epochs', 2), cfg.get('bs', 128), cfg.get('lr', 4e-3)
    x, Xva, yva, Xte, yte = B.load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d' % (len(x), len(yva), len(yte)))
    G = B.Model()
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    t0 = time.time()
    C = B.build_counts(x); G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float()
    G.tri = B.build_tri(x)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, cfg.get('beta', 1280.0)) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, Sv, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    En = U[:, :64] * Sv[:64].sqrt()[None, :]; En = En / (En.norm(dim=1, keepdim=True) + 1e-9)
    G.cls = B.kmeans(En, C2.K); del C, ppmi
    cls_np = G.cls.numpy(); G.sp = {}
    for nm, spec in C2.SPECS.items():
        G.sp[nm] = C2.build_sparse(x, cls_np, spec, C2.MINCNT.get(nm, 1))
    t_table = time.time() - t0; log('表の構築 %.0f 秒' % t_table)
    net, t_train, ntok = train_dense(x, d, nl, epochs, bs, lr)
    macs = dense_macs(d, nl); fl = 6.0 * macs * ntok
    log('小さなTransformer d=%d nl=%d: パラメータ %d, MAC/語 %.3e, 学習 %.0f 秒, %.3e FLOP' % (d, nl, sum(p.numel() for p in net.parameters()), macs, t_train, fl))
    del x
    Fv, gv = C2.collect(Xva, G); Fv['dn'] = dense_logp(net, Xva)
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    fitted = {}
    w, D, b = C2.fit(Fv, allx, yva, gate=gv, gated=True); fitted['T'] = (allx, w, D, True); log('T 調整窓 %.4f' % b)
    w, D, b = C2.fit(Fv, ['uni', 'dn'], yva, iters=60); fitted['D'] = (['uni', 'dn'], w, D, False); log('D 調整窓 %.4f' % b)
    w, D, b = C2.fit(Fv, allx + ['dn'], yva, gate=gv, gated=True); fitted['TD'] = (allx + ['dn'], w, D, True); log('TD 調整窓 %.4f' % b)
    tot = {vn: 0.0 for vn in fitted}; n_te = len(yte)
    dn_te = dense_logp(net, Xte)
    for a in range(0, n_te, 250):
        Fc, gc = C2.all_feats(Xte[a:a + 250], G); yc = yte[a:a + 250]; Fc['dn'] = dn_te[a:a + 250]
        with torch.no_grad():
            for vn, (nms, w, D, gated) in fitted.items():
                z = C2.logits(Fc, nms, w, gc, D); tot[vn] += Fn.cross_entropy(z, yc, reduction='sum').item()
    bits = {vn: tot[vn] / n_te / LN2 for vn in tot}
    tab_macs = 72 * V; cores = os.cpu_count() or 1
    res = dict(cfg=cfg, test_bits=bits, d_macs=macs, tab_macs=tab_macs, td_macs=macs + tab_macs, t_table=t_table, t_train=t_train, flops_train=fl,
               cores=cores, dense_weights=fitted['TD'][1][-1].item())
    log('--- 結果 d=%d nl=%d ---' % (d, nl))
    log('T  (表だけ)        %.4f  MAC %.2e  表の構築 %.0f 秒 x %d コア' % (bits['T'], tab_macs, t_table, cores))
    log('D  (Transformerだけ) %.4f  MAC %.2e  学習 %.2e FLOP, %.0f 秒 x %d コア' % (bits['D'], macs, fl, t_train, cores))
    log('TD (表+Transformer) %.4f  MAC %.2e' % (bits['TD'], macs + tab_macs))
    log('参照 密d128(16M語) 5.130  MAC 1.46e6  学習 2.8e14 FLOP')
    log('RESULT_HY2 ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
