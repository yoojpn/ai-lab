"""
LLM1: LLMの全結合層(FFN)を「記憶層(Product-Key Memory)」に置き換えると、同じ計算量で品質が上がるか。
  FFN = 全ての語に毎回かけ算する知識の置き場。記憶層 = 大きな表から、必要な数行だけを引く。表を大きくしても計算量はほぼ増えない。
  基準 = 小さなTransformer(最後の層は、最後の語の位置だけを計算する。出力に使うのはそこだけ)。
  変えるのは最後の層のFFNだけ(variantで指定)。語の窓は直前12語。
使い方: LADDER_CORPUS=... python3 exp_llm1.py <訓練語数N> <variant> [d=64]
  variant: ffn<dff> (例 ffn256) | pkm<nk>[h<heads>][k<topk>] (例 pkm64, pkm256h2)
出力: NN単独のbit/語、パラメーター数、1語の積和、学習時間
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, HIST, bits, log
import exp_scale as es

es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


class PKM(torch.nn.Module):
    """Product-Key Memory。クエリを2分割し、各半分でnk個の鍵から上位kを選び、その積集合(k*k)から上位kの値を足し合わせる"""
    def __init__(s, d, nk, heads=2, topk=16):
        super().__init__()
        s.d, s.nk, s.h, s.k = d, nk, heads, topk
        s.q = torch.nn.Linear(d, heads * d, bias=False)
        s.k1 = torch.nn.Parameter(torch.randn(heads, nk, d // 2) / math.sqrt(d // 2))
        s.k2 = torch.nn.Parameter(torch.randn(heads, nk, d // 2) / math.sqrt(d // 2))
        s.v = torch.nn.EmbeddingBag(nk * nk, d, mode='sum')
        torch.nn.init.normal_(s.v.weight, std=d ** -0.5)
        s.bn = torch.nn.LayerNorm(d)

    def forward(s, x):                       # x: (B,1,d) → (B,1,d)
        B = x.shape[0]
        q = s.q(x[:, 0]).view(B, s.h, s.d)
        q = s.bn(q)
        q1, q2 = q[..., :s.d // 2], q[..., s.d // 2:]
        s1 = torch.einsum('bhd,hnd->bhn', q1, s.k1); s2 = torch.einsum('bhd,hnd->bhn', q2, s.k2)
        v1, i1 = s1.topk(s.k, -1); v2, i2 = s2.topk(s.k, -1)
        sc = (v1[..., :, None] + v2[..., None, :]).reshape(B, s.h, -1)
        ix = (i1[..., :, None] * s.nk + i2[..., None, :]).reshape(B, s.h, -1)
        top, j = sc.topk(s.k, -1)
        idx = ix.gather(-1, j)
        w = torch.softmax(top, -1)
        out = s.v(idx.reshape(B * s.h, s.k), per_sample_weights=w.reshape(B * s.h, s.k)).view(B, s.h, s.d).sum(1)
        return out[:, None]

    def macs(s):
        return s.h * s.d * s.d + s.h * s.nk * s.d + s.h * s.k * s.k * 2 + s.h * s.k * s.d


class FFN(torch.nn.Module):
    def __init__(s, d, dff):
        super().__init__()
        s.a = torch.nn.Linear(d, dff); s.b = torch.nn.Linear(dff, d); s.dff = dff; s.d = d

    def forward(s, x):
        return s.b(torch.relu(s.a(x)))

    def macs(s):
        return 2 * s.d * s.dff


class Block(torch.nn.Module):
    def __init__(s, d, heads, ffn, last):
        super().__init__()
        s.n1 = torch.nn.LayerNorm(d); s.n2 = torch.nn.LayerNorm(d)
        s.att = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        s.ffn, s.last = ffn, last

    def forward(s, x):
        h = s.n1(x)
        if s.last:
            x = x[:, -1:] + s.att(h[:, -1:], h, h, need_weights=False)[0]
        else:
            x = x + s.att(h, h, h, need_weights=False)[0]
        return x + s.ffn(s.n2(x))


class Model(torch.nn.Module):
    def __init__(s, variant, d=64, nl=2, heads=4):
        super().__init__()
        s.d, s.nl = d, nl
        s.E = torch.nn.Embedding(V, d); s.pos = torch.nn.Parameter(torch.zeros(HIST, d))
        blocks = []
        for l in range(nl):
            last = l == nl - 1
            if last and variant.startswith('pkm'):
                import re
                m = re.match(r'pkm(\d+)(?:h(\d+))?(?:k(\d+))?', variant)
                ffn = PKM(d, int(m.group(1)), int(m.group(2) or 2), int(m.group(3) or 16))
            else:
                dff = int(variant[3:]) if variant.startswith('ffn') else 4 * d
                ffn = FFN(d, dff)
            blocks.append(Block(d, heads, ffn, last))
        s.blocks = torch.nn.ModuleList(blocks)
        s.out = torch.nn.Linear(d, V)

    def forward(s, X, Hd=None):
        x = s.E(X) + s.pos
        for b in s.blocks:
            x = b(x)
        return s.out(x[:, -1])

    def macs(s):
        d, n = s.d, HIST
        t = 0
        for i, b in enumerate(s.blocks):
            if b.last:
                t += d * d + n * 2 * d * d + n * d + d * d + b.ffn.macs()
            else:
                t += n * (4 * d * d + b.ffn.macs()) + 2 * n * n * d
        return t + d * V


def main():
    N = int(sys.argv[1]); variant = sys.argv[2]; d = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    t0 = time.time(); torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    ctx = lab.load(N)
    net = Model(variant, d)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
    steps = len(X) // 512; st = 0
    for i in torch.randperm(len(X)).split(512):
        if len(i) < 512:
            break
        for g in opt.param_groups:
            g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
        loss = torch.nn.functional.cross_entropy(net(X[i]), y[i])
        opt.zero_grad(); loss.backward(); opt.step(); st += 1
    pt = []
    with torch.no_grad():
        for i in range(0, len(ctx.Xte), 4096):
            z = net(torch.from_numpy(ctx.Xte[i:i + 4096])); lp = torch.log_softmax(z, 1)
            pt.append(lp[torch.arange(len(z)), torch.from_numpy(ctx.yte[i:i + 4096])].exp().numpy())
    b = bits(np.concatenate(pt).astype(np.float64))
    params = sum(p.numel() for p in net.parameters())
    import json
    res = dict(variant=variant, d=d, N=N, bits=b, params=params, macs=net.macs(), sec=time.time() - t0)
    log('variant=%s d=%d N=%d: %.4f bit/語 | パラメーター%d | 1語の積和%.3fM | %.0fs' % (variant, d, N, b, params, net.macs() / 1e6, time.time() - t0))
    log('RESULT ' + json.dumps(res))


if __name__ == '__main__':
    main()
