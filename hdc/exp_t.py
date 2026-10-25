"""
T: ちゃんとした小さなTransformerを、同じデータ(窓12語)で学習させ、「実測時間 vs bit/語」の曲線を取る。
  数える方式(軽い構成: 31〜35秒で5.49 bit/語)と、同じ品質に届くまでの実測時間を比べるため。
  学習時間(秒)は、評価の時間を除いて数える。評価は同じ20000窓。
使い方: LADDER_CORPUS=... python3 exp_t.py '{"N":16000000,"d":128,"nl":2,"epochs":1}'
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, HIST, bits, log
import exp_scale as es
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


class TF(torch.nn.Module):
    def __init__(s, d, nl, heads=4):
        super().__init__()
        s.E = torch.nn.Embedding(V, d); s.pos = torch.nn.Parameter(torch.zeros(HIST, d))
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        s.enc = torch.nn.TransformerEncoder(layer, nl, enable_nested_tensor=False)
        s.out = torch.nn.Linear(d, V)
    def forward(s, X):
        return s.out(s.enc(s.E(X) + s.pos)[:, -1])


def evaluate(net, X, y):
    net.eval(); pt = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            z = net(torch.from_numpy(X[i:i + 4096])); lp = torch.log_softmax(z, 1)
            pt.append(lp[torch.arange(len(z)), torch.from_numpy(y[i:i + 4096])].exp().numpy())
    net.train()
    return bits(np.concatenate(pt).astype(np.float64))


def train_tf(ctx, d, nl, epochs=1, bs=512, lr=2e-3, log_every=None):
    """窓12語の次語予測でTransformerを学習して返す(混合実験で再利用)"""
    n = len(ctx.Xtr); net = TF(d, nl)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
    steps = int(epochs * (n // bs)); st = 0; t0 = time.time()
    while st < steps:
        for i in torch.randperm(n).split(bs):
            if len(i) < bs or st >= steps: break
            for g in opt.param_groups:
                g['lr'] = lr * min(1, (st + 1) / 300) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.95 + 0.05)
            loss = torch.nn.functional.cross_entropy(net(X[i]), y[i])
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
            if st % 5000 == 0: log('  TF step %d/%d %.0fs' % (st, steps, time.time() - t0))
    net.eval()
    return net, time.time() - t0


def run(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    ctx = lab.load(cfg['N']); n = len(ctx.Xtr); d, nl = cfg['d'], cfg['nl']
    bs = cfg.get('bs', 512); lr = cfg.get('lr', 2e-3)
    net = TF(d, nl); params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    X = torch.from_numpy(ctx.Xtr); y = torch.from_numpy(ctx.ytr)
    steps = int(cfg.get('epochs', 1) * (n // bs)); nev = 10; every = max(1, steps // nev)
    Xte, yte = ctx.Xte[:20000], ctx.yte[:20000]
    curve = []; ttrain = 0.0; st = 0
    log('params %d steps %d' % (params, steps))
    while st < steps:
        for i in torch.randperm(n).split(bs):
            if len(i) < bs or st >= steps: break
            t = time.time()
            for g in opt.param_groups:
                g['lr'] = lr * min(1, (st + 1) / 300) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.95 + 0.05)
            loss = torch.nn.functional.cross_entropy(net(X[i]), y[i])
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
            ttrain += time.time() - t
            if st % every == 0 or st == steps:
                b = evaluate(net, Xte, yte); curve.append((st, round(ttrain, 1), round(b, 4)))
                log('step %d 学習時間 %.0fs bit/語 %.4f' % (st, ttrain, b))
    res = dict(cfg=cfg, params=params, curve=curve, final_bits=curve[-1][2], train_sec=ttrain)
    log('RESULT ' + json.dumps(res))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
