"""
ATT: 学習なしの「やわらかい induction」 (Attention相当) を、CB2 の全部入りに足したとき、何 bit 変わるか。
  鍵   : 過去の各語 j (0..SEQ-2) の座標 E[x_j]  (E = PPMI の SVD 座標 64 次元, 長さ1に正規化)
  問い : 直前の語 x_{SEQ-1} の座標
  重み : softmax_j( beta * cos(問い, 鍵_j) )           ← beta は固定の数個から混合器が選ぶ (学習するのは混ぜる重みだけ)
  値   : as_b : 過去語 j の「次に来た語」 x_{j+1} の one-hot を、重みで足す。  特徴 = log1p(10 * 合計)
         ab_b : 過去語 j の bigram 行 P(次 | x_j) を、重みで混ぜる。              特徴 = log(合計 + 1e-6)
  比べるもの : 同じ実行内の '+all_gated' (CB2の全部入り) と、'+as', '+ab', '+as+ab'
使い方: LADDER_CORPUS=... python3 exp_att.py '{"N":16000000,"ntune":4000,"ntest":20000}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
from exp_cb import V, SEQ, LN2, BUCKETS, log

BETAS = [4.0, 16.0, 64.0]


def att_feats(X, G, betas=BETAS, sub=64):
    outs = {}
    for a in range(0, len(X), sub):
        Xc = X[a:a + sub]; n = len(Xc)
        q = G.E[Xc[:, SEQ - 1]]                          # n x 64
        Kt = G.E[Xc[:, :SEQ - 1]]                        # n x 63 x 64
        sim = (Kt * q[:, None, :]).sum(-1)               # n x 63
        succ = Xc[:, 1:SEQ]                              # n x 63
        rows = G.Pbi[Xc[:, :SEQ - 1]]                    # n x 63 x V
        for b in betas:
            w = torch.softmax(b * sim, 1)
            o = torch.zeros(n, V); o.scatter_add_(1, succ, w)
            outs.setdefault('as_%g' % b, []).append(torch.log1p(10.0 * o))
            m = torch.bmm(w[:, None, :], rows)[:, 0, :]
            outs.setdefault('ab_%g' % b, []).append(torch.log(m + 1e-6))
    return {k: torch.cat(v) for k, v in outs.items()}


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000)
    x, Xva, yva, Xte, yte = B.load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d' % (len(x), len(yva), len(yte)))
    G = B.Model()
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = B.build_counts(x); log('距離の束ごとの共起を数えた')
    G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float()
    G.tri = B.build_tri(x)
    beta = cfg.get('beta', 1280.0)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, beta) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, S, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :64] * S[:64].sqrt()[None, :]; E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    G.E = E; G.cls = B.kmeans(E, C2.K); del C, ppmi
    G.Pbi = torch.exp(G.pmF[0]) * G.pu[None, :]          # 距離1の平滑化した P(次|前)
    cls_np = G.cls.numpy(); G.sp = {}
    for nm, spec in C2.SPECS.items():
        G.sp[nm] = C2.build_sparse(x, cls_np, spec, C2.MINCNT.get(nm, 1))
    del x
    t = time.time(); Fv, gv = C2.collect(Xva, G); Fv.update(att_feats(Xva, G))
    log('調整窓の特徴 %.0f 秒, 専門家 %d 個' % (time.time() - t, len(Fv)))
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    AS = ['as_%g' % b for b in BETAS]; AB = ['ab_%g' % b for b in BETAS]
    variants = {'+all_gated': allx, '+as': allx + AS, '+ab': allx + AB, '+as+ab': allx + AS + AB}
    fitted = {}
    for vn, nm in variants.items():
        w, D, b = C2.fit(Fv, nm, yva, gate=gv, gated=True); fitted[vn] = (nm, w, D)
        log('%-12s 調整窓 %.4f  att重み %s' % (vn, b, {n: round(float(w[i]), 3) for i, n in enumerate(nm) if n.startswith(('as_', 'ab_'))}))
    del Fv
    tot = {vn: 0.0 for vn in fitted}; n_te = len(yte)
    for a in range(0, n_te, 250):
        Fc, gc = C2.all_feats(Xte[a:a + 250], G); Fc.update(att_feats(Xte[a:a + 250], G)); yc = yte[a:a + 250]
        with torch.no_grad():
            for vn, (nm, w, D) in fitted.items():
                tot[vn] += Fn.cross_entropy(C2.logits(Fc, nm, w, gc, D), yc, reduction='sum').item()
        if (a // 250) % 20 == 0:
            log('テスト %d / %d' % (a, n_te))
    res = dict(cfg=cfg, test_bits={vn: tot[vn] / n_te / LN2 for vn in tot},
               weights={vn: dict(zip(fitted[vn][0], [round(float(v), 3) for v in fitted[vn][1]])) for vn in ('+as+ab',)})
    for vn in tot:
        log('テスト %-12s %.4f bit/語' % (vn, res['test_bits'][vn]))
    log('差 (+as+ab) - (+all_gated) = %+.4f bit' % (res['test_bits']['+as+ab'] - res['test_bits']['+all_gated']))
    log('RESULT_ATT ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
