"""
H6: 索引(データストア)は、学習なしでパラメータの代わりになるか。 kNN-LM (Khandelwal 2019) の最小版。
 1. 密モデル(d)を普通に学習(exp_y.run)。
 2. 学習データの各位置の最終層の出力h と 次の語 を保存 = 索引(T個)。
 3. テスト語ごとに、現在のhに近いT個から上位k個を取り、近い順に重みをつけて「次の語」の確率を作る。
 4. p = lam * p_knn + (1-lam) * p_lm で bit/語 を測る。 lam と 温度は、テスト窓の前半5000で決め、後半15000で報告(漏れ防止)。
計算量: 総当たりの探索は 1語あたり T*d 積和。近似探索の理想値は 約 4*sqrt(T)*d (報告は両方)。
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, log
import exp_y


def hidden_last(net, X, bs=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net.hidden(X[i:i + bs])[:, -1].float())
    return torch.cat(out)


def main(cfg):
    net, Xte, yt = exp_y.run(cfg)
    net.eval(); d = cfg.get('d', 128); nl = cfg.get('nl', 4); seq = cfg.get('seq', 64); S = seq + 1
    ctx = lab.load(cfg['N']); stream = np.concatenate(ctx.tr_i)[:cfg['N']]
    M = len(stream) // S; chunks = torch.from_numpy(stream[:M * S].reshape(M, S).astype(np.int64))
    if cfg.get('sel') == 'rand':
        # 核は D 語だけで学習(exp_y と同じ乱数で選ぶ)。索引は、核が学習しなかった塊(src=out)か、全部(src=all)から作る。
        Dn = int(cfg['Dw'] / S); perm = torch.randperm(M, generator=torch.Generator().manual_seed(0))
        order = perm[Dn:] if cfg.get('src', 'out') == 'out' else perm
        chunks = chunks[order]; M = len(chunks); log('索引の元: 核が学習していない塊 %d (src=%s)' % (M, cfg.get('src', 'out')))
    Tmax = cfg.get('Tmax', 4_000_000); lo = 32; per = S - 1 - lo
    nw = min(M, -(-Tmax // per)); Ks, ys = [], []
    with torch.no_grad():
        for i in range(0, nw, 512):
            b = chunks[i:i + 512]; h = net.hidden(b[:, :-1])[:, lo:]          # (b, 32, d): 文脈が十分長い位置だけ
            Ks.append(h.reshape(-1, d).float()); ys.append(b[:, 1 + lo:].reshape(-1))
    K = torch.cat(Ks)[:Tmax]; tg = torch.cat(ys)[:Tmax]; kn = (K ** 2).sum(1)
    log('index %d x %d' % tuple(K.shape))
    n = min(20000, len(Xte)); Q = hidden_last(net, Xte[:n]); y = yt[:n]
    with torch.no_grad():
        plm = torch.cat([torch.softmax(net.out(Q[i:i + 1024]), 1)[torch.arange(len(Q[i:i + 1024])), y[i:i + 1024]] for i in range(0, n, 1024)]).double().numpy()
    res = dict(cfg=cfg, macs_body=nl * (12 * d * d + seq * d) + d * V, bits_lm_all=float(-np.log2(plm).mean()))
    tune, rep = slice(0, 5000), slice(5000, n); kk = 64
    res['bits_lm_rep'] = float(-np.log2(plm[rep]).mean()); res['store'] = []
    for T in cfg.get('Ts', [250_000, 1_000_000, 4_000_000]):
        T = min(T, len(K)); KT, kT, gT = K[:T], kn[:T], tg[:T]
        D = torch.empty(n, kk); I = torch.empty(n, kk, dtype=torch.long); c = max(32, int(2e9 // (4 * T)))
        with torch.no_grad():
            for a in range(0, n, c):
                q = Q[a:a + c]; dist = (q ** 2).sum(1)[:, None] - 2 * q @ KT.T + kT[None]
                v, ix = torch.topk(dist, kk, dim=1, largest=False); D[a:a + c] = v.clamp_min(0); I[a:a + c] = ix
        hit = (gT[I] == y[:, None]).double()                                   # (n,k) 近傍の「次の語」が正解か
        best = None
        for trel in (0.25, 0.5, 1, 2, 4):
            tau = trel * D.mean(1, keepdim=True).clamp_min(1e-6)
            w = torch.softmax(-D / tau, 1).double(); pk = (w * hit).sum(1).numpy()
            for lam in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
                p = lam * pk + (1 - lam) * plm; b = float(-np.log2(p[tune]).mean())
                if best is None or b < best[0]: best = (b, trel, lam, pk)
        _, trel, lam, pk = best; p = lam * pk + (1 - lam) * plm
        res['store'].append(dict(T=T, trel=trel, lam=lam, bits=float(-np.log2(p[rep]).mean()), d_bits=float(-np.log2(p[rep]).mean() - res['bits_lm_rep']),
                                 knn_macs_brute=T * d, knn_macs_ann=4 * math.sqrt(T) * d, knn_top1_correct=float(hit[:, 0].mean().item())))
        log('T=%d trel=%s lam=%s bits=%.4f (lm %.4f)' % (T, trel, lam, res['store'][-1]['bits'], res['bits_lm_rep']))
    log('RESULT_KNN ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
