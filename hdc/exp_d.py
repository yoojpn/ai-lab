"""
D群: データ効率を徹底的に調べる。「必要なデータ量が、容量とともに増えてしまう」問題への攻め方。
  d0            : 学習曲線(読ませる語数 n を変えたときのbit/語)と、語の頻度帯ごとの損失。どこでデータが要るかを見る
  d1 <lever>    : 一つの工夫を足して、同じ語数 n(1M, 4M)で測る。工夫:
                    base   なし(PPMI初期化つきの基準)
                    mt     2語先も同時に当てる(補助の出力層)
                    ep3    同じデータを3回読む
                    hard   難しい窓を選んで学ぶ(4倍の候補から損失の大きい窓を選ぶ)
                    aug    文脈の語を、近い語に15%の確率で置き換える
                    mtaug  mt + aug
  どの工夫も、基準の学習曲線で「同じ品質になるのに必要だった語数」に換算して、データ倍率を出す(あとで集計)。
使い方: LADDER_CORPUS=... python3 exp_d.py <d0|d1> <最大N> [lever]
"""
import os, sys, time, json, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab, items
from lab import V, HIST, NEAR, DE, bits, log
import exp_scale as es

es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


def sub_ctx(ctx, n):
    c = types.SimpleNamespace(**vars(ctx))
    c.Xtr, c.ytr, c.mtr, c.ptr = ctx.Xtr[:n], ctx.ytr[:n], ctx.mtr[:n], ctx.ptr[:n]
    c.N = n
    return c


class Net2(lab.Net):
    def __init__(s, H=256, E0=None):
        super().__init__(H, False, E0)
        s.l2b = torch.nn.Linear(H, V)

    def both(s, X):
        e = s.E(X)
        h = torch.relu(s.l1(torch.cat([e[:, -NEAR:].reshape(len(X), -1), e[:, :-NEAR].mean(1)], 1)))
        return s.l2(h), s.l2b(h)


def nearest_words(E0, k=8):
    En = E0 / (np.linalg.norm(E0, axis=1, keepdims=True) + 1e-9)
    out = np.zeros((V, k), np.int64)
    for i in range(0, V, 500):
        s = En[i:i + 500] @ En.T
        s[np.arange(len(s)), np.arange(i, i + len(s))] = -9
        out[i:i + 500] = np.argsort(-s, 1)[:, :k]
    return torch.from_numpy(out)


def train_lever(ctx, n, lever, H=256, seed=0, bs=512, lr=2e-3):
    torch.manual_seed(seed); torch.set_num_threads(os.cpu_count() or 2)
    c = sub_ctx(ctx, n)
    E0 = items.ppmi_embeddings(c, DE)
    net = Net2(H, E0)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.from_numpy(c.Xtr); y = torch.from_numpy(c.ytr)
    # 2語先の正解(同じ文書の次の窓の正解)
    y2 = torch.full_like(y, -100)
    same = torch.from_numpy(c.mtr[1:] == c.mtr[:-1])
    y2[:-1][same] = y[1:][same]
    use_mt = lever in ('mt', 'mtaug')
    use_aug = lever in ('aug', 'mtaug')
    nb = nearest_words(E0) if use_aug else None
    epochs = 3 if lever == 'ep3' else 1
    steps = int(epochs * (n // bs)); st = 0
    while st < steps:
        perm = torch.randperm(n)
        stride = bs * (4 if lever == 'hard' else 1)
        for i in perm.split(stride):
            if len(i) < stride or st >= steps:
                break
            for g in opt.param_groups:
                g['lr'] = lr if st < 0.6 * steps else lr / 2
            if lever == 'hard':
                with torch.no_grad():
                    l = torch.nn.functional.cross_entropy(net.both(X[i])[0], y[i], reduction='none')
                i = i[l.topk(bs).indices]
            xb = X[i]
            if use_aug:
                m = torch.rand(xb.shape) < 0.15
                rep = nb[xb, torch.randint(0, nb.shape[1], xb.shape)]
                xb = torch.where(m, rep, xb)
            o1, o2 = net.both(xb)
            loss = torch.nn.functional.cross_entropy(o1, y[i])
            if use_mt:
                loss = loss + 0.5 * torch.nn.functional.cross_entropy(o2, y2[i], ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step(); st += 1
    return net


def evaluate(net, ctx):
    Xte = ctx.Xte; yte = ctx.yte
    pt = []
    with torch.no_grad():
        for i in range(0, len(Xte), 4096):
            z = net.both(torch.from_numpy(Xte[i:i + 4096]))[0]
            lp = torch.log_softmax(z, 1)
            pt.append(lp[torch.arange(len(z)), torch.from_numpy(yte[i:i + 4096])].exp().numpy())
    p = np.concatenate(pt).astype(np.float64)
    loss = -np.log2(p + 1e-12)
    bins = {'unk': yte == 0, 'rank1-100': (yte >= 1) & (yte <= 100), 'rank101-1000': (yte > 100) & (yte <= 1000), 'rank1001+': yte > 1000}
    return bits(p), {k: dict(share_of_tokens=float(m.mean()), bits=float(loss[m].mean()), contribution=float(loss[m].sum() / len(loss))) for k, m in bins.items()}


def main():
    mode = sys.argv[1]; Nmax = int(sys.argv[2]); lever = sys.argv[3] if len(sys.argv) > 3 else 'base'
    t0 = time.time()
    ctx = lab.load(Nmax)
    log('データ準備 [%.0fs]' % (time.time() - t0))
    res = dict(mode=mode, lever=lever, rows=[])
    if mode == 'd0':
        ns = [n for n in (250000, 500000, 1000000, 2000000, 4000000, 8000000, 16000000) if n <= len(ctx.Xtr)]
        lv = 'base'
    else:
        ns = [n for n in (1000000, 4000000) if n <= len(ctx.Xtr)]
        lv = lever
    if os.environ.get('DNS'):
        ns = [int(x) for x in os.environ['DNS'].split(',')]
    for n in ns:
        net = train_lever(ctx, n, lv)
        b, bins = evaluate(net, ctx)
        res['rows'].append(dict(n=n, bits=b, bins=bins, sec=time.time() - t0))
        log('lever=%s n=%d: %.4f bit/語 %s [%.0fs]' % (lv, n, b, {k: round(v['contribution'], 3) for k, v in bins.items()}, time.time() - t0))
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
