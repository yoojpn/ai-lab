"""
X4: 少ない例から規則を当てる。
  機械 = v3の棚(解けた課題から自動発見したカード)+検証つき探索。見える例の数 k=1..4。隠した4例すべてで正しければ「解けた」。
  比較 = 直接出力型の小さなTransformer(文脈内学習): k個の例(入力→出力)と新しい入力を読み、出力を直接作る。
         学習側には答えの規則を使わず、同じ世界の課題を大量に与える(機械より多くの課題で学習する、有利な条件)。
  試験の課題は、学習に出なかった「部品の組み合わせ」を含む。
使い方: SEED=0 python3 exp_x4.py [small]
"""
import os, sys, time, json, heapq, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v3 as V3
from items4 import _apply, NOP

BUD = (5000, 20000, 80000)


def vsearch(task, shelf, budget, kvis, maxdepth=12):
    ins, outs = task['ins'], task['outs']
    tgt = [list(o) for o in outs[:kvis]]; start = [list(l) for l in ins[:kvis]]
    seen = {tuple(tuple(l) for l in start)}
    heap = [(0, 0, (), start, 0)]; cnt = 0; used = 0
    while heap and used < budget:
        d, _, prog, cur, nd = heapq.heappop(heap)
        if nd >= maxdepth:
            continue
        for card in shelf:
            nxt = cur
            for op in card:
                nxt = [_apply(op, l) for l in nxt]
            used += len(card) * kvis
            key = tuple(tuple(l) for l in nxt)
            if key in seen:
                continue
            seen.add(key)
            p2 = prog + card
            if nxt == tgt:
                return all(V3.run(p2, ins[j]) == outs[j] for j in range(4, 8))
            cnt += 1
            heapq.heappush(heap, (d + 1, cnt, p2, nxt, nd + 1))
            if used >= budget:
                break
    return False


_G = {}


def _w(a):
    i, b, k = a
    return vsearch(_G['t'][i], _G['s'], b, k)


def machine_eval(tasks, shelf, kvis, budget, procs):
    import multiprocessing as mp
    _G['t'] = tasks; _G['s'] = shelf
    with mp.get_context('fork').Pool(procs) as p:
        r = p.map(_w, [(i, budget, kvis) for i in range(len(tasks))], chunksize=4)
    return float(np.mean(r))


# ---------------- 直接出力型の小さなTransformer ----------------
SEP, EOS, PAD, BOS = 10, 11, 12, 13
NTOK = 14


def seq_pair(i, o):
    return list(i) + [SEP] + list(o) + [EOS]


def make_batch(tasks, idx, k_rng, kfix=None):
    srcs, tgts = [], []
    for ti in idx:
        t = tasks[ti]
        k = kfix if kfix else int(k_rng.integers(1, 5))
        order = k_rng.permutation(8)
        vis, q = order[:k], order[k]
        s = []
        for j in vis:
            s += seq_pair(t['ins'][j], t['outs'][j])
        s += list(t['ins'][q]) + [SEP]
        srcs.append(s); tgts.append(list(t['outs'][q]) + [EOS])
    S = max(map(len, srcs)); T = max(map(len, tgts)) + 1
    src = torch.full((len(idx), S), PAD); tin = torch.full((len(idx), T), PAD); tout = torch.full((len(idx), T), PAD)
    for b, (s, t) in enumerate(zip(srcs, tgts)):
        src[b, :len(s)] = torch.tensor(s)
        tin[b, 0] = BOS; tin[b, 1:len(t) + 1] = torch.tensor(t)
        tout[b, :len(t)] = torch.tensor(t)
    return src, tin, tout


class TF(torch.nn.Module):
    def __init__(self, d=128, L=3):
        super().__init__()
        self.emb = torch.nn.Embedding(NTOK, d); self.pos = torch.nn.Embedding(160, d)
        self.tf = torch.nn.Transformer(d, 4, L, L, 4 * d, 0.0, batch_first=True)
        self.out = torch.nn.Linear(d, NTOK)

    def enc(self, x):
        return self.emb(x) + self.pos(torch.arange(x.shape[1])[None])

    def forward(self, src, tin):
        m = torch.nn.Transformer.generate_square_subsequent_mask(tin.shape[1])
        h = self.tf(self.enc(src), self.enc(tin), tgt_mask=m, src_key_padding_mask=(src == PAD), memory_key_padding_mask=(src == PAD))
        return self.out(h)


def train_tf(tasks, steps, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = TF(); opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    for st in range(steps):
        idx = rng.integers(0, len(tasks), 64)
        src, tin, tout = make_batch(tasks, idx, rng)
        lg = m(src, tin)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, NTOK), tout.reshape(-1), ignore_index=PAD)
        for g in opt.param_groups:
            g['lr'] = 5e-4 * min(1, (st + 1) / 200) * (0.5 * (1 + math.cos(math.pi * st / steps)) * 0.9 + 0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0:
            print('  tf step', st, 'loss %.3f' % loss.item(), flush=True)
    return m


def tf_eval(m, tasks, k):
    """隠した4例(4..7番目)を見える k 個の例だけで予測。4つ全部正しければ解けた"""
    ok = []
    rng = np.random.default_rng(123)
    m.eval()
    with torch.no_grad():
        for t in tasks:
            good = True
            for q in range(4, 8):
                s = []
                for j in range(k):
                    s += seq_pair(t['ins'][j], t['outs'][j])
                s += list(t['ins'][q]) + [SEP]
                src = torch.tensor([s]); dec = [BOS]
                for _ in range(14):
                    lg = m(src, torch.tensor([dec]))[0, -1]
                    nx = int(lg.argmax()); dec.append(nx)
                    if nx == EOS:
                        break
                pred = [x for x in dec[1:] if x < 10] if dec[-1] == EOS else None
                if pred != list(t['outs'][q]):
                    good = False; break
            ok.append(good)
    return float(np.mean(ok))


def main():
    small = len(sys.argv) > 1 and sys.argv[1] == 'small'
    SEED = int(os.environ.get('SEED', '0'))
    procs = os.cpu_count() or 2
    torch.set_num_threads(procs)
    rng = np.random.default_rng(SEED)
    macros, forb = V3.make_world(SEED)
    nT, nU, nTF, rounds, steps = (40, 300, 2000, 2, 100) if small else (150, 2400, 20000, 6, 6000)
    test = V3.make_tasks(nT, rng, macros, forb, test=True)
    pool = V3.make_tasks(nU, rng, macros, forb, test=False)
    t0 = time.time()
    shelf = [(i,) for i in range(NOP)]
    for r in range(rounds):
        out = V3.solve_all(pool, shelf, 80000, procs)
        solved = [p for p, ok, _ in out if p is not None]
        shelf, new = V3.sleep_measured(solved, shelf, pool, procs)
    print('棚 %d枚 (%.0fs)' % (len(shelf), time.time() - t0), flush=True)
    res = {'seed': SEED, 'shelf_size': len(shelf), 'machine': {}, 'machine_primitives_only': {}, 'tf_direct': {}}
    prims = [(i,) for i in range(NOP)]
    for k in (1, 2, 3, 4):
        res['machine']['k=%d' % k] = {'budget=%d' % b: machine_eval(test, shelf, k, b, procs) for b in BUD}
        res['machine_primitives_only']['k=%d' % k] = {'budget=%d' % b: machine_eval(test, prims, k, b, procs) for b in BUD}
        print('machine k=%d' % k, res['machine']['k=%d' % k], '| primitives', res['machine_primitives_only']['k=%d' % k], '%.0fs' % (time.time() - t0), flush=True)
    # TF: 学習側にはたくさんの課題(規則の部品は見せないが、課題は大量に与える)
    trn = V3.make_tasks(nTF, np.random.default_rng(SEED + 100), macros, forb, test=False)
    m = train_tf(trn, steps, SEED)
    res['tf_params'] = int(sum(p.numel() for p in m.parameters()))
    for k in (1, 2, 3, 4):
        res['tf_direct']['k=%d' % k] = tf_eval(m, test, k)
    print('tf', res['tf_direct'], 'params', res['tf_params'], '%.0fs' % (time.time() - t0), flush=True)
    print('RESULT ' + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
