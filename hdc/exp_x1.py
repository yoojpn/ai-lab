"""
X1: 小さなNN(照合器)が「次にどのカードを試すか」を選ぶと、同じ実行回数で解ける課題が増えるか。
  棚は v3 のやり方(測定つき選択)で作った最終版に固定。照合器の有無だけを変える。
  照合器 = MLP(いまの途中結果と目標 → 棚の各カードの点数)。答えは見ない:
    棚だけで課題プールを解く → 解けた経路で照合器を学習 → 照合器つき探索で再び解く → また学習(自己改善)。
  探索の予算 = 実行した基本操作の数。照合器の呼び出しは、(a)無料 (b)1回=8操作分 の2通りで数える。
使い方: SEED=0 python3 exp_x1.py [small]
"""
import os, sys, time, json, heapq
import multiprocessing as mp
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v3 as V3
from items4 import _apply, enc, NOP

K_TOP = 8
BUDGETS = (2000, 5000, 20000)
_G = {}


def psearch(task, shelf, budget, pol, fcost, maxdepth=12):
    ins, outs = task['ins'], task['outs']
    tgt = [list(o) for o in outs[:4]]; start = [list(l) for l in ins[:4]]
    seen = {tuple(tuple(l) for l in start)}
    heap = [(0.0, 0, (), start, ())]; cnt = 0; used = 0
    while heap and used < budget:
        pr, _, prog, cur, path = heapq.heappop(heap)
        if len(path) >= maxdepth:
            continue
        if pol is None:
            order = [(1.0, i) for i in range(len(shelf))]
        else:
            W1, b1, W2, b2 = pol
            h = np.maximum(enc(cur, tgt) @ W1 + b1, 0); z = h @ W2 + b2
            z = z - z.max(); lp = z - np.log(np.exp(z).sum())
            used += fcost
            top = np.argsort(-lp)[:K_TOP]
            order = [(float(-lp[i]), int(i)) for i in top]
        for c, i in order:
            card = shelf[i]
            nxt = cur
            for op in card:
                nxt = [_apply(op, l) for l in nxt]
            used += len(card) * 4
            key = tuple(tuple(l) for l in nxt)
            if key in seen:
                continue
            seen.add(key)
            p2 = prog + card; path2 = path + (i,)
            if nxt == tgt:
                return list(p2), all(V3.run(p2, ins[j]) == outs[j] for j in range(4, 8)), used, path2
            cnt += 1
            heapq.heappush(heap, (pr + c, cnt, p2, nxt, path2))
            if used >= budget:
                break
    return None, False, used, None


def _work(a):
    i, budget, fcost = a
    return psearch(_G['tasks'][i], _G['shelf'], budget, _G['pol'], fcost)


def solve_all(tasks, shelf, pol, budget, fcost, procs):
    _G.update(tasks=tasks, shelf=shelf, pol=pol)
    with mp.get_context('fork').Pool(procs) as p:
        return p.map(_work, [(i, budget, fcost) for i in range(len(tasks))], chunksize=4)


def evaluate(tasks, shelf, pol, fcost, procs):
    return {'budget=%d' % b: float(np.mean([r[1] for r in solve_all(tasks, shelf, pol, b, fcost, procs)])) for b in BUDGETS}


def build_data(tasks, shelf, results):
    X, y = [], []
    for t, r in zip(tasks, results):
        if r[3] is None:
            continue
        tgt = [list(o) for o in t['outs'][:4]]; cur = [list(l) for l in t['ins'][:4]]
        for i in r[3]:
            X.append(enc(cur, tgt)); y.append(i)
            for op in shelf[i]:
                cur = [_apply(op, l) for l in cur]
    return np.stack(X), np.array(y)


def train_policy(X, y, ncard, epochs=60, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, ncard))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    for _ in range(epochs):
        for i in torch.randperm(len(Xt)).split(256):
            loss = torch.nn.functional.cross_entropy(net(Xt[i]), yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
    W1 = net[0].weight.detach().numpy().T.astype(np.float64); b1 = net[0].bias.detach().numpy().astype(np.float64)
    W2 = net[2].weight.detach().numpy().T.astype(np.float64); b2 = net[2].bias.detach().numpy().astype(np.float64)
    acc = float((net(Xt).argmax(1) == yt).float().mean())
    return (W1, b1, W2, b2), acc


def main():
    small = len(sys.argv) > 1 and sys.argv[1] == 'small'
    SEED = int(os.environ.get('SEED', '0'))
    procs = os.cpu_count() or 2
    rng = np.random.default_rng(SEED)
    macros, forb = V3.make_world(SEED)
    nT, nU, rounds, prounds = (40, 300, 2, 2) if small else (150, 2400, 6, 3)
    test = V3.make_tasks(nT, rng, macros, forb, test=True)
    pool = V3.make_tasks(nU, rng, macros, forb, test=False)
    t0 = time.time()
    # 1. 棚を作る(v3の測定つき選択)
    shelf = [(i,) for i in range(NOP)]
    for r in range(rounds):
        out = V3.solve_all(pool, shelf, 80000, procs)
        solved = [p for p, ok, _ in out if p is not None]
        shelf, new = V3.sleep_measured(solved, shelf, pool, procs)
    print('棚 %d枚 (%.0fs)' % (len(shelf), time.time() - t0), flush=True)
    res = {'seed': SEED, 'shelf_size': len(shelf), 'cards': [list(c) for c in shelf[NOP:]]}
    res['arm0_cards_only'] = evaluate(test, shelf, None, 0, procs)
    print('arm0', res['arm0_cards_only'], '%.0fs' % (time.time() - t0), flush=True)
    # 2. 照合器の自己改善
    X = np.zeros((0, 704), np.float32); y = np.zeros(0, np.int64); pol = None
    for r in range(prounds):
        out = solve_all(pool, shelf, pol, 20000, 0, procs)
        Xn, yn = build_data(pool, shelf, out)
        X = np.concatenate([X, Xn]); y = np.concatenate([y, yn])
        pol, acc = train_policy(X, y, len(shelf))
        e0 = evaluate(test, shelf, pol, 0, procs); e8 = evaluate(test, shelf, pol, 8, procs)
        res['policy_round%d' % (r + 1)] = dict(train_samples=int(len(y)), train_acc=acc, free=e0, forward_cost8=e8,
                                              solved_in_pool=int(sum(o[3] is not None for o in out)))
        print('policy round', r + 1, res['policy_round%d' % (r + 1)], '%.0fs' % (time.time() - t0), flush=True)
    print('RESULT ' + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
