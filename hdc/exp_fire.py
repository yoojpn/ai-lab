"""
M: FFNのどのニューロンが発火するかを、安い低ランク予測器で先に当てて、当たったものだけ計算したら品質と計算はどうなるか。
 学習済み密モデルの linear1 (dff x d) を SVD して上位rランクだけ使う(閉形式、追加学習なし): o_hat = U_r S_r (V_r^T x)
 予測: o_hat > -m*std(o_hat) なら「発火する」とみなし、その分だけ正確に計算。他は0にする(=近似)。
 計算量/語/層 = 予測 r*(d+dff) + 本計算 2*d*dff*(予測発火率)  vs 密 2*d*dff。
 報告: bit/語、予測発火率、真の発火の取りこぼし率、FFN積和の比、全体積和の比。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, log
import exp_y


def main(cfg):
    net, Xte, yt = exp_y.run(cfg)
    net.eval(); d = cfg.get('d', 128); nl = cfg.get('nl', 4); seq = cfg.get('seq', 64)
    n = min(20000, len(Xte)); Xq, yq = Xte[:n], yt[:n]
    L0 = net.layers[0]; dff = L0.linear1.weight.shape[0]
    body = nl * (12 * d * d + seq * d) + d * V; ffn_dense = 2 * d * dff
    for L in net.layers:
        L.activation_relu_or_gelu = 0
        L.linear1.register_forward_hook(lambda m, a, o, L=L: setattr(L, '_x', a[0]))
    state = dict(mode=None); stats = dict(pred=[], miss=[])
    def mk(L, r):
        U, S, Vt = torch.linalg.svd(L.linear1.weight.detach(), full_matrices=False)
        return (U[:, :r] * S[:r]), Vt[:r], L.linear1.bias.detach()
    def make_act(L):
        def act(o):
            if state['mode'] is None: return torch.relu(o)
            r, m = state['mode']; Ur, Vr, b = state['pk'][id(L)]
            oh = (L._x @ Vr.T) @ Ur.T + b; mask = oh > -m * oh.std(-1, keepdim=True)
            true = o > 0
            stats['pred'].append(mask.float().mean().item()); stats['miss'].append(((true & ~mask).float().sum() / true.float().sum().clamp_min(1)).item())
            return torch.relu(o) * mask
        return act
    for L in net.layers: L.activation = make_act(L)
    def bits():
        out = []
        with torch.no_grad():
            for i in range(0, n, 500):
                x = Xq[i:i + 500]; m_ = x.shape[1]; mask = torch.nn.Transformer.generate_square_subsequent_mask(m_)
                h = net.E(x) + net.pos[:m_]
                for L in net.layers[:net.active]: h = L(h, src_mask=mask, is_causal=True)
                lp = torch.log_softmax(net.out(h[:, -1]), 1); out.append(lp[torch.arange(len(x)), yq[i:i + 500]].exp().numpy())
        return float(-np.log2(np.concatenate(out).astype(np.float64)).mean())
    base = bits(); log('base %.4f' % base)
    res = dict(cfg=cfg, base=base, rows=[], dff=dff)
    for r in (4, 8, 16, 32):
        state['pk'] = {id(L): mk(L, r) for L in net.layers}
        for m in (0.0, 0.5, 1.0, 2.0):
            state['mode'] = (r, m); stats['pred'].clear(); stats['miss'].clear()
            b = bits(); fire = float(np.mean(stats['pred'])); miss = float(np.mean(stats['miss']))
            ffn = (r * (d + dff) + ffn_dense * fire) / ffn_dense; tot = (body - nl * ffn_dense + nl * ffn_dense * ffn) / body
            row = dict(r=r, m=m, bits=b, dbits=b - base, pred_fire=fire, miss=miss, ffn_ratio=ffn, total_ratio=tot)
            res['rows'].append(row); log('r=%d m=%.1f bits %.4f (%+.4f) 予測発火 %.3f 取りこぼし %.3f FFN積和比 %.3f 全体比 %.3f' % (r, m, b, b - base, fire, miss, ffn, tot))
    state['mode'] = None
    log('RESULT_FIRE ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
