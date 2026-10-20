"""
第5陣: 問題4(考える時間と検証)を深める。
 #70 実行を見ながら進める探索: 小さな方策NN(いまの途中結果と目標を見て、次の操作を選ぶ)で、
     実行しては枝を選ぶ。方策なしの幅優先と、同じ「展開回数」で比べる。さらに、答えを見ずに(例だけで)自己改善する。
"""
import os, time, math, heapq
import numpy as np
import torch
from items3 import _apply as _a10

NOP = 16
MAXD = 7


def _apply(op, l):
    if op < 10: return _a10(op, l)
    if op == 10: return l[1:] + l[:1]                       # 左に回す
    if op == 11: return l[-1:] + l[:-1]                     # 右に回す
    if op == 12: return (l[-1:] + l[1:-1] + l[:1]) if len(l) > 1 else l   # 両端を入れ替え
    if op == 13: return [(x + i) % 10 for i, x in enumerate(l)]  # 位置を足す
    if op == 14: return [(x * x) % 10 for x in l]           # 二乗
    return [l[i] for i in range(0, len(l), 2)]              # 1つおきに取る


def _run(prog, l):
    for op in prog:
        l = _apply(op, l)
    return l


def _make_tasks(n, rng, Ls=(3, 4, 5, 6, 7), n_ex=4):
    tasks = []
    while len(tasks) < n:
        L = int(rng.choice(Ls)); prog = [int(rng.integers(NOP)) for _ in range(L)]
        ins = [[int(x) for x in rng.integers(0, 10, rng.integers(5, 9))] for _ in range(2 * n_ex)]
        outs = [_run(prog, l) for l in ins]
        if any(len(o) == 0 for o in outs) or len({tuple(o) for o in outs}) < 3:
            continue
        tasks.append((prog, ins, outs))
    return tasks


def enc(cur, tgt):
    v = np.zeros((4, 2, 8, 11), np.float32)
    for i in range(4):
        for j, l in enumerate((cur[i], tgt[i])):
            for p in range(8):
                v[i, j, p, l[p] if p < len(l) else 10] = 1
    return v.reshape(-1)


class Pol:
    def __init__(self):
        self.net = torch.nn.Sequential(torch.nn.Linear(704, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, NOP))
        self.w = None

    def fit(self, samples, epochs=25, lr=2e-3, seed=0):
        torch.manual_seed(seed)
        X = torch.from_numpy(np.stack([s[0] for s in samples])); y = torch.tensor([s[1] for s in samples])
        self.net = torch.nn.Sequential(torch.nn.Linear(704, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, NOP))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for ep in range(epochs):
            for i in torch.randperm(len(X)).split(256):
                loss = torch.nn.functional.cross_entropy(self.net(X[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step()
        self.w = [p.detach().numpy().astype(np.float32) for p in self.net.parameters()]

    def logp(self, x):
        W1, b1, W2, b2, W3, b3 = self.w
        h = np.maximum(x @ W1.T + b1, 0); h = np.maximum(h @ W2.T + b2, 0); z = h @ W3.T + b3
        z = z - z.max(); return z - np.log(np.exp(z).sum())


def samples_from(task_list):
    out = []
    for ins, outs, prog in task_list:
        cur = [list(l) for l in ins]
        for op in prog:
            out.append((enc(cur, outs), op)); cur = [_apply(op, l) for l in cur]
    return out


def search(task, pol, budget):
    """方策つき最良優先探索。展開回数 budget まで。見つけた(プログラム, 隠した例でも正しいか)を返す"""
    _, ins, outs = task
    tgt = [list(o) for o in outs[:4]]; start = [list(l) for l in ins[:4]]
    seen = {tuple(tuple(l) for l in start)}
    heap = [(0.0, 0, [], start)]; cnt = 0; exp = 0
    while heap and exp < budget:
        nl, _, prog, cur = heapq.heappop(heap)
        if len(prog) >= MAXD:
            continue
        exp += 1
        lp = pol.logp(enc(cur, tgt)) if pol is not None else np.full(NOP, -math.log(NOP))
        for op in range(NOP):
            nxt = [_apply(op, l) for l in cur]
            key = tuple(tuple(l) for l in nxt)
            if key in seen:
                continue
            seen.add(key)
            p2 = prog + [op]
            if nxt == tgt:
                return p2, all(_run(p2, ins[i]) == outs[i] for i in range(4, 8))
            cnt += 1
            heapq.heappush(heap, (nl - lp[op] if pol is not None else len(p2) + 0.0, cnt, p2, nxt))
    return None, False


def evaluate(tasks, pol, budgets):
    r = {}
    for b in budgets:
        r['expand=%d' % b] = float(np.mean([search(t, pol, b)[1] for t in tasks]))
    return r


def i70_policy_search(ctx=None):
    torch.set_num_threads(os.cpu_count() or 2)
    small = os.environ.get('SMALL')
    rng = np.random.default_rng(0)
    nT, nL, nU, rounds = (40, 300, 200, 2) if small else (150, 1000, 1500, 4)
    test = _make_tasks(nT, rng); labeled = _make_tasks(nL, rng); unl = _make_tasks(nU, rng)
    big = _make_tasks(20000 if not small else 2000, rng)
    budgets = (50, 200, 1000)
    res = {}
    res['no_policy(幅優先)'] = evaluate(test, None, budgets)
    pol = Pol(); pol.fit(samples_from([(t[1][:4], t[2][:4], t[0]) for t in labeled]))
    res['policy_labeled%d' % nL] = evaluate(test, pol, budgets)
    full = Pol(); full.fit(samples_from([(t[1][:4], t[2][:4], t[0]) for t in big]), epochs=15)
    res['policy_labeled%d(上限)' % len(big)] = evaluate(test, full, budgets)
    pool = samples_from([(t[1][:4], t[2][:4], t[0]) for t in labeled])
    for r in range(1, rounds + 1):
        found = []
        for t in unl:
            p, ok = search(t, pol, 500)
            if p is not None:
                found.append((t[1][:4], t[2][:4], p))
        pool = pool + samples_from(found)
        pol = Pol(); pol.fit(pool)
        res['self_improve_round%d(found %d/%d)' % (r, len(found), len(unl))] = evaluate(test, pol, budgets)
    return res


ITEMS4 = {'70': i70_policy_search}
