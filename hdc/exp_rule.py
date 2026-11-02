"""R1: 規則を1回の例で覚える仕組みの最小実験。
世界: 語は C 個 (0..C-1)。規則 s (ずらし量): y = (x + s) % C。
1エピソード: 例 (x, y) を 1 つ見せ (x は 規則 s のもと y = x+s)、 新しい語 x2 に同じ規則を当てて y2 を当てる。
保留: (x2, s) の組の 30% を 学習から外し、 テストはその組だけで行う (表で覚える方式は 原理的に 偶然になる)。
仕組み (結合): 各語に 位相ベクトル phi(c) = exp(i * theta[c]) (D 次元、 学習する)。
  規則ベクトル r = phi(y) * conj(phi(x))   (例から 1 回で作る)
  予測 z = phi(x2) * r,  語 c の点数 = tau * Re( sum conj(phi(c)) * z )
対照: 規則を 「s ごとに ランダムな並べ替え」 にすると 構造がないので、 保留の組では 偶然 (1/C) になるはず。
使い方: python3 exp_rule.py '{"C":32,"D":16,"rule":"shift","steps":3000,"seed":0}'
"""
import sys, json, math
import numpy as np
import torch
import torch.nn.functional as F


def main(cfg):
    C = cfg.get('C', 32); D = cfg.get('D', 16); steps = cfg.get('steps', 3000); B = cfg.get('B', 1); seed = cfg.get('seed', 0)
    rule = cfg.get('rule', 'shift'); hold = cfg.get('hold', 0.3); lr = cfg.get('lr', 0.05)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    perms = np.stack([rng.permutation(C) for _ in range(C)])      # 対照用: s ごとの並べ替え
    def f(s, x): return (x + s) % C if rule == 'shift' else perms[s][x]
    pairs = [(x2, s) for x2 in range(C) for s in range(C)]
    rng.shuffle(pairs); nh = int(len(pairs) * hold); held = set(pairs[:nh]); train = pairs[nh:]
    tr = np.array(train); te = np.array(sorted(held))
    theta = torch.nn.Parameter(torch.randn(C, D) * 3.0); tau = torch.nn.Parameter(torch.tensor(5.0))
    opt = torch.optim.Adam([theta, tau], lr=lr)
    def logits(x, y, x2):
        ph = torch.polar(torch.ones(C, D), theta)
        r = ph[y] * ph[x].conj(); z = ph[x2] * r
        return tau * (z[:, None, :] * ph[None].conj()).real.sum(-1) / D
    def episodes(pr, n):
        i = rng.integers(0, len(pr), n); x2 = pr[i, 0]; s = pr[i, 1]
        x = (x2 + rng.integers(1, C, n)) % C                      # 例の語は x2 と別
        y = np.array([f(si, xi) for si, xi in zip(s, x)]); y2 = np.array([f(si, xi) for si, xi in zip(s, x2)])
        return [torch.from_numpy(a.astype(np.int64)) for a in (x, y, x2, y2)]
    def acc(pr, n=4000):
        with torch.no_grad():
            x, y, x2, y2 = episodes(pr, n); return (logits(x, y, x2).argmax(1) == y2).float().mean().item()
    out = []; first90 = None
    for st in range(1, steps + 1):
        x, y, x2, y2 = episodes(tr, B)
        loss = F.cross_entropy(logits(x, y, x2), y2); opt.zero_grad(); loss.backward(); opt.step()
        if st in (10, 30, 100, 300, 1000, 2000, 3000, 5000, 10000, 20000) or st == steps:
            a_tr, a_te = acc(tr), acc(te); out.append((st * B, a_tr, a_te))
            print('エピソード %6d: 学習の組 %.3f | 保留の組 %.3f' % (st * B, a_tr, a_te), flush=True)
            if first90 is None and a_te >= 0.9: first90 = st * B
    table = C * C * (C - 1)
    print('RESULT_RULE ' + json.dumps(dict(cfg=cfg, curve=out, first_episodes_to_90=first90, table_size=table, chance=1 / C, n_train_pairs=len(train), n_held_pairs=len(te))))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
