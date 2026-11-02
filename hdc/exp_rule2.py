"""R2: 規則の種類を 教えずに、 例 2 つから 規則を見分けて 当てはめる。
世界: 語は C=32 個 (0..31)。 エピソードごとに 規則の種類 (隠す) と 母数を 引く:
  shift: y = (x + s) % C        (s = 1..31)
  xor:   y = x XOR m            (m = 1..31)   <- 5 ビットの語として
  table: y = perm_s[x]          (s = 0..31, 構造なし。 対照: 保留の組では 偶然になるはず)
例を 2 つ (x1,y1), (x2,y2) 見せ、 新しい語 x3 の y3 を当てる。
仕組み (全ての部分が 同じ形。 規則ごとの専用部品はない):
  各語に K 個の部分 x D 個の位相 phi (学習)。 例 j の 規則ベクトル r_j = phi(y_j) * conj(phi(x_j))。
  部分 k の一致度 a_k = Re(<r_1k, r_2k>) / D   (2 つの例で 規則ベクトルが 揃う部分 = 規則に合った部分)
  重み w = softmax(beta * a)   (gate=0 のときは 一様)
  予測 z_k = phi(x3)_k * (r_1k + r_2k)/2,  語 c の点数 = tau * sum_k w_k Re(<z_k, phi(c)_k>) / D
保留: 規則の種類ごとに (x3, 母数) の組の 30% を学習から外し、 テストはその組だけ。
使い方: python3 exp_rule2.py '{"K":4,"D":8,"gate":1,"steps":8000,"B":8,"seed":0}'
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F

FAM = ['shift', 'xor', 'table']


def main(cfg):
    C = 32; K = cfg.get('K', 4); D = cfg.get('D', 8); steps = cfg.get('steps', 8000); B = cfg.get('B', 8); seed = cfg.get('seed', 0)
    gate = cfg.get('gate', 1); hold = cfg.get('hold', 0.3); lr = cfg.get('lr', 0.05)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    perms = np.stack([rng.permutation(C) for _ in range(C)])
    def f(fam, p, x):
        if fam == 0: return (x + p) % C
        if fam == 1: return x ^ p
        return perms[p][x]
    trn, tst = [], []
    for fam in range(3):
        ps = [(x3, p) for x3 in range(C) for p in (range(C) if fam == 2 else range(1, C))]
        rng.shuffle(ps); nh = int(len(ps) * hold)
        tst.append(np.array(sorted(ps[:nh]))); trn.append(np.array(ps[nh:]))
    theta = torch.nn.Parameter(torch.randn(C, K * D) * 3.0); tau = torch.nn.Parameter(torch.tensor(5.0)); beta = torch.nn.Parameter(torch.tensor(5.0))
    opt = torch.optim.Adam([theta, tau, beta], lr=lr)
    def logits(x1, y1, x2, y2, x3):
        ph = torch.polar(torch.ones(C, K * D), theta); n = len(x1)
        r1 = (ph[y1] * ph[x1].conj()).view(n, K, D); r2 = (ph[y2] * ph[x2].conj()).view(n, K, D)
        a = (r1 * r2.conj()).real.sum(-1) / D
        w = F.softmax(beta * a, dim=1) if gate else torch.full((n, K), 1.0 / K)
        z = ph[x3].view(n, K, D) * (r1 + r2) / 2
        pc = ph.view(1, C, K, D).conj()
        sc = (z[:, None] * pc).real.sum(-1) / D               # (n, C, K)
        return tau * (sc * w[:, None, :]).sum(-1), a, w
    def episodes(pool, fam_ids):
        n = len(fam_ids); x1 = np.zeros(n, int); x2 = np.zeros(n, int); x3 = np.zeros(n, int); y1 = np.zeros(n, int); y2 = np.zeros(n, int); y3 = np.zeros(n, int)
        for i, fam in enumerate(fam_ids):
            pr = pool[fam][rng.integers(0, len(pool[fam]))]; q, p = pr
            others = [v for v in range(C) if v != q]; a, b = rng.choice(others, 2, replace=False)
            x1[i], x2[i], x3[i] = a, b, q; y1[i], y2[i], y3[i] = f(fam, p, a), f(fam, p, b), f(fam, p, q)
        return [torch.from_numpy(v.astype(np.int64)) for v in (x1, y1, x2, y2, x3, y3)]
    def acc(pool, n=1500):
        res = []
        with torch.no_grad():
            for fam in range(3):
                x1, y1, x2, y2, x3, y3 = episodes(pool, [fam] * n); lo, a, w = logits(x1, y1, x2, y2, x3); res.append((lo.argmax(1) == y3).float().mean().item())
        return res
    out = []; first = {}
    for st in range(1, steps + 1):
        x1, y1, x2, y2, x3, y3 = episodes(trn, rng.integers(0, 3, B))
        loss = F.cross_entropy(logits(x1, y1, x2, y2, x3)[0], y3); opt.zero_grad(); loss.backward(); opt.step()
        if st in (100, 300, 1000, 2000, 3000, 5000, 8000, 12000, 20000) or st == steps:
            at, ae = acc(trn), acc(tst); out.append((st * B, at, ae))
            print('エピソード %6d: 学習の組 %s | 保留の組 %s' % (st * B, ' '.join('%s %.3f' % (FAM[i], at[i]) for i in range(3)), ' '.join('%s %.3f' % (FAM[i], ae[i]) for i in range(3))), flush=True)
            for i in range(2):
                if i not in first and ae[i] >= 0.9: first[FAM[i]] = st * B
    # どの部分が どの規則を担当したか (保留の組で gate の重みの平均)
    with torch.no_grad():
        wk = {}
        for fam in range(3):
            x1, y1, x2, y2, x3, y3 = episodes(tst, [fam] * 1000); _, a, w = logits(x1, y1, x2, y2, x3); wk[FAM[fam]] = [round(v, 3) for v in w.mean(0).tolist()]
    print('gate の重み (部分ごと):', wk)
    print('RESULT_RULE2 ' + json.dumps(dict(cfg=cfg, curve=out, first_episodes_to_90=first, gate_w=wk)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
