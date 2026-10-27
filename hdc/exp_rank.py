"""
点群(各語の数字の列)は、実際には幅dのうちどれだけの次元を使っているか。 学習したあとのモデルで測る。
 - 各層の出力(最後の語の位置)の「有効次元」: 分散の90%/99%に必要な主成分の数 r90, r99 と participation ratio
 - FFNの中間(幅4d、ReLUのあと)の有効次元と発火割合
 - 重み行列(in_proj, out_proj, linear1, linear2)の特異値: エネルギー(σ^2)の90%/99%に必要なランク
幅dを変えて、有効次元が d に比例するか(=節約できない)、ほぼ一定か(=規模が大きいほど節約できる)を見る。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, log
import exp_y


def spec(M):
    M = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    c = np.cumsum(s) / s.sum()
    return dict(r90=int(np.searchsorted(c, 0.90) + 1), r99=int(np.searchsorted(c, 0.99) + 1), pr=float(s.sum() ** 2 / (s ** 2).sum()), n=int(len(s)))


def wspec(W):
    s = np.linalg.svd(W, compute_uv=False) ** 2
    c = np.cumsum(s) / s.sum()
    return dict(r90=int(np.searchsorted(c, 0.90) + 1), r99=int(np.searchsorted(c, 0.99) + 1), full=int(min(W.shape)))


def main(cfg):
    net, Xte, yt = exp_y.run(cfg)
    net.eval(); d = cfg.get('d', 128); n = min(8000, len(Xte))
    acts, ffn = [], []
    hooks = []
    store = {}
    for i, L in enumerate(net.layers):
        hooks.append(L.linear1.register_forward_hook(lambda m, a, o, i=i: store.setdefault(i, []).append(torch.relu(o[:, -1]).detach().numpy())))
    hs = [[] for _ in range(len(net.layers) + 1)]
    with torch.no_grad():
        for i in range(0, n, 1000):
            x = Xte[i:i + 1000]; m = x.shape[1]
            mask = torch.nn.Transformer.generate_square_subsequent_mask(m)
            h = net.E(x) + net.pos[:m]; hs[0].append(h[:, -1].numpy())
            for k, L in enumerate(net.layers):
                h = L(h, src_mask=mask, is_causal=True); hs[k + 1].append(h[:, -1].numpy())
    res = dict(cfg=cfg, d=d, resid=[spec(np.concatenate(h)) for h in hs], ffn=[], weights=[])
    for i in range(len(net.layers)):
        A = np.concatenate(store[i]); sp = spec(A); sp['fire'] = float((A > 0).mean()); sp['dff'] = A.shape[1]; res['ffn'].append(sp)
        L = net.layers[i]
        res['weights'].append(dict(in_proj=wspec(L.self_attn.in_proj_weight.detach().numpy()), out_proj=wspec(L.self_attn.out_proj.weight.detach().numpy()),
                                   linear1=wspec(L.linear1.weight.detach().numpy()), linear2=wspec(L.linear2.weight.detach().numpy())))
    log('RESULT_RANK ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
