"""
PCLS: クラスを「座標のk-means」から「予測でまとめるクラス」に変えると、 CB2全部入りは何 bit 変わるか。 wiki のみ。
  km200 / pred200 / km500 / pred500 を同じ表・同じ混合器で比べる。 クラスを使う専門家: c123, wc, cc4 と 回路の 'class' 版。
使い方: LADDER_CORPUS=... python3 exp_pcls.py '{"N":16000000,"ntune":4000,"ntest":20000,"Ks":[200,500]}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
from pcls import predictive_classes
from exp_cb import V, SEQ, LN2, log


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000); Ks = cfg.get('Ks', [200, 500])
    x, Xva, yva, Xte, yte = B.load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d' % (len(x), len(yva), len(yte)))
    G = B.Model()
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = B.build_counts(x)
    G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float(); G.tri = B.build_tri(x)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, cfg.get('beta', 1280.0)) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, S, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :64] * S[:64].sqrt()[None, :]; E = E / (E.norm(dim=1, keepdim=True) + 1e-9); del C, ppmi
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    res = {}
    for K in Ks:
        km = B.kmeans(E, K)
        for mode in ('km', 'pred'):
            t0 = time.time(); C2.K = K
            G.cls = km if mode == 'km' else predictive_classes(G.C1, K, km, iters=cfg.get('iters', 15), log=log)
            G.sp = {nm: C2.build_sparse(x, G.cls.numpy(), spec, C2.MINCNT.get(nm, 1)) for nm, spec in C2.SPECS.items()}
            Fv, gv = C2.collect(Xva, G); w, D, b = C2.fit(Fv, allx, yva, gate=gv, gated=True); del Fv
            tot = 0.0
            for a in range(0, len(yte), 250):
                Fc, gc = C2.all_feats(Xte[a:a + 250], G)
                with torch.no_grad():
                    tot += Fn.cross_entropy(C2.logits(Fc, allx, w, gc, D), yte[a:a + 250], reduction='sum').item()
            bits = tot / len(yte) / LN2
            res['%s%d' % (mode, K)] = dict(tune=b, test=bits, entries={nm: int(len(G.sp[nm][0])) for nm in C2.SPECS})
            log('%s K=%d: 調整窓 %.4f, テスト %.4f bit/語 (%.0f 秒)' % (mode, K, b, bits, time.time() - t0))
    log('--- 結果 ---')
    for k, v in res.items():
        log('%-8s テスト %.4f bit/語' % (k, v['test']))
    log('RESULT_PCLS ' + json.dumps(dict(cfg=cfg, res=res)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
