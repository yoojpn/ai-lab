"""
T2: 普通のLLMの学び方(因果マスク+全位置で次の語を学習)の小さなTransformer。「実測時間 vs bit/語」。
  前のexp_tは窓12語の最後の1語しか学習していなかった(学習信号が12分の1)。ここでは13語の塊の全位置を学習する。
  評価は同じ: 窓12語→次の1語 (テスト先頭20000窓)。
使い方: LADDER_CORPUS=... python3 exp_t2.py '{"N":16000000,"d":128,"nl":2,"epochs":3}'
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, HIST, bits, log
import exp_scale as es
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)
S = HIST + 1


class CTF(torch.nn.Module):
    def __init__(s, d, nl, heads=4, seq=S):
        super().__init__()
        s.E = torch.nn.Embedding(V, d); s.pos = torch.nn.Parameter(torch.zeros(seq, d))
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        s.enc = torch.nn.TransformerEncoder(layer, nl, enable_nested_tensor=False)
        s.out = torch.nn.Linear(d, V)

    def forward(s, X, last_only=False):
        n = X.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
        h = s.enc(s.E(X) + s.pos[:n], mask=mask, is_causal=True)
        return s.out(h[:, -1] if last_only else h)


def evaluate(net, X, y):
    net.eval(); pt = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            z = net(torch.from_numpy(X[i:i + 4096]), last_only=True); lp = torch.log_softmax(z, 1)
            pt.append(lp[torch.arange(len(z)), torch.from_numpy(y[i:i + 4096])].exp().numpy())
    net.train()
    return bits(np.concatenate(pt).astype(np.float64))


def run(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    ctx = lab.load(cfg['N']); d, nl = cfg['d'], cfg['nl']; bs = cfg.get('bs', 128); lr = cfg.get('lr', 2e-3)
    # 学習データ: 文書をつないだ流れを13語ずつの塊にする(各語は1回ずつ「答え」になる)
    stream = np.concatenate([a for a in ctx.tr_i])[:cfg['N']]
    M = len(stream) // S; chunks = torch.from_numpy(stream[:M * S].reshape(M, S).astype(np.int64))
    net = CTF(d, nl); params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    steps = int(cfg.get('epochs', 1) * (M // bs)); nev = 12; every = max(1, steps // nev)
    Xte, yte = ctx.Xte[:20000], ctx.yte[:20000]
    curve = []; ttrain = 0.0; st = 0
    log('params %d chunks %d steps %d' % (params, M, steps))
    while st < steps:
        for i in torch.randperm(M).split(bs):
            if len(i) < bs or st >= steps: break
            t = time.time()
            for g in opt.param_groups:
                g['lr'] = lr * min(1, (st + 1) / 300) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.95 + 0.05)
            b = chunks[i]
            lo = net(b[:, :-1])
            loss = torch.nn.functional.cross_entropy(lo.reshape(-1, V), b[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
            ttrain += time.time() - t
            if st % every == 0 or st == steps:
                bb = evaluate(net, Xte, yte); curve.append((st, round(ttrain, 1), round(bb, 4)))
                log('step %d 学習時間 %.0fs bit/語 %.4f' % (st, ttrain, bb))
    res = dict(cfg=cfg, params=params, curve=curve, final_bits=curve[-1][2], train_sec=ttrain)
    log('RESULT ' + json.dumps(res))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
