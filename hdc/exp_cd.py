"""
CD: (C) 易しい語の割合と、安い経路(数える2-gram)に回したときの品質/計算、
    (D) 各層の出力が「直近k語」だけでどこまで決まるか(群内分散で測る R^2)。
密モデル(d,nl)を普通に学習してから測る。評価は検証でなくテスト窓(学習に使っていない)。
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, log
import exp_y
from exp_r import Prior


def main(cfg):
    net, Xte, yt = exp_y.run(cfg)
    net.eval(); d, nl = cfg.get('d', 128), cfg.get('nl', 4); seq = cfg.get('seq', 64)
    ctx = lab.load(cfg['N']); stream = np.concatenate(ctx.tr_i)[:cfg['N']]
    pr = Prior(stream)
    macs = nl * (12 * d * d + seq * d) + d * V
    pn, pp, pmax, hs = [], [], [], [[] for _ in range(nl + 1)]
    with torch.no_grad():
        for i in range(0, len(Xte), 1024):
            x = Xte[i:i + 1024]; y = yt[i:i + 1024]; n = x.shape[1]
            mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
            h = net.E(x) + net.pos[:n]; hs[0].append(h[:, -1].clone())
            for k, L in enumerate(net.layers[:net.active]):
                h = L(h, src_mask=mask, is_causal=True); hs[k + 1].append(h[:, -1].clone())
            lp = torch.log_softmax(net.out(h[:, -1]), 1)
            pn.append(lp[torch.arange(len(y)), y].exp())
            lq = pr.logp(x[:, -1]).exp()
            pp.append(lq[torch.arange(len(y)), y]); pmax.append(lq.max(1).values)
    pn = torch.cat(pn).double().numpy(); pp = torch.cat(pp).double().numpy(); pmax = torch.cat(pmax).double().numpy()
    bn = float(-np.log2(pn).mean()); res = dict(cfg=cfg, macs=macs, bits_nn=bn, bits_prior=float(-np.log2(pp).mean()))
    res['easy'] = {str(t): float((pn > t).mean()) for t in (0.5, 0.9, 0.99)}
    gate = []
    for tau in (0.3, 0.5, 0.7, 0.9, 0.95):
        r = pmax >= tau; mixed = np.where(r, pp, pn); b = float(-np.log2(mixed).mean())
        gate.append(dict(tau=tau, routed=float(r.mean()), bits=b, dbits=b - bn, rel_macs=float(1 - r.mean())))
    res['gate'] = gate
    for k in (2, 3):
        key = np.zeros(len(Xte), np.int64)
        for j in range(1, k + 1): key = key * 1000003 + Xte[:, -j].numpy()
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        m = cnt[inv] >= 2; cov = float(m.mean()); r2 = []
        for l in range(nl + 1):
            H = torch.cat(hs[l]).double().numpy()[m]; g = inv[m]
            _, gi = np.unique(g, return_inverse=True); ng = gi.max() + 1
            mu = np.zeros((ng, H.shape[1])); np.add.at(mu, gi, H); mu /= np.bincount(gi, minlength=ng)[:, None]
            within = ((H - mu[gi]) ** 2).sum(); total = ((H - H.mean(0)) ** 2).sum()
            r2.append(float(1 - within / total))
        res['local%d' % k] = dict(coverage=cov, r2_by_layer=r2)
    log('RESULT_CD ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
