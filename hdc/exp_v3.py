"""
v3: 新しい仕組みの最初の試作。
  状態   : 課題の例(入力→出力)と、いまの途中結果
  棚     : 「技能カード」= 操作の列。最初は基本操作16個だけ。
  実行   : カードを実行して途中結果を作る
  検証   : 目標の出力に一致するかを確かめる(見せた4例で確かめ、隠した4例で本当に正しいかを測る)
  探索   : 状態の重複を省いた最良優先探索(カード1枚=1手)
  学習   : 答え(プログラム)は見ない。解けた課題から、よく出る操作の並びを新しいカードにして棚に足す(眠りの段階)
比較   : LLM型 = 小さな系列モデル(GRU)が答えのプログラムを順に生成 → 実行して検証。こちらは答え付きの教師データで学習。
          どちらも「基本操作を何回実行したか」を計算量とする(同じ計算量で比べる)。
課題   : 隠れた12個の「部品」(操作の2〜3個の並び)を組み合わせて、課題が作られる。学習側は部品の存在を知らない。
          試験の課題には、学習の課題に出なかった「部品の組み合わせ」を必ず含める(未知の組み合わせへの一般化)。
使い方: python3 exp_v3.py [small]
"""
import os, sys, time, math, heapq, json, itertools
import multiprocessing as mp
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from items4 import _apply, NOP

NM = 12           # 隠れた部品の数
BUDGETS = (5000, 20000, 80000)


def run(seq, l):
    for op in seq:
        l = _apply(op, l)
    return l


# ---------------- 課題の作り方 ----------------
def make_world(seed=0):
    rng = np.random.default_rng(seed)
    macros = []
    while len(macros) < NM:
        m = tuple(int(x) for x in rng.integers(NOP, size=int(rng.integers(2, 4))))
        if m not in macros:
            macros.append(m)
    pairs = [(a, b) for a in range(NM) for b in range(NM)]
    forb = set(tuple(p) for p in [pairs[i] for i in rng.permutation(len(pairs))[:len(pairs) // 4]])
    return macros, forb


def make_tasks(n, rng, macros, forb, test=False):
    tasks = []
    while len(tasks) < n:
        k = int(rng.integers(2, 5)); ms = [int(rng.integers(NM)) for _ in range(k)]
        has_forb = any((a, b) in forb for a, b in zip(ms, ms[1:]))
        if has_forb != test:
            continue
        prog = [op for m in ms for op in macros[m]]
        ins = [[int(x) for x in rng.integers(0, 10, rng.integers(5, 10))] for _ in range(8)]
        outs = [run(prog, l) for l in ins]
        if any(len(o) == 0 for o in outs) or len({tuple(o) for o in outs}) < 4:
            continue
        tasks.append(dict(prog=prog, ins=ins, outs=outs, ms=ms))
    return tasks


# ---------------- 棚と探索 ----------------
SHELF = [(i,) for i in range(NOP)]


def search(task, shelf, budget, maxdepth=12):
    ins, outs = task['ins'], task['outs']
    tgt = [list(o) for o in outs[:4]]; start = [list(l) for l in ins[:4]]
    seen = {tuple(tuple(l) for l in start)}
    heap = [(0, 0, (), start)]; cnt = 0; used = 0
    while heap and used < budget:
        d, _, prog, cur = heapq.heappop(heap)
        if d >= maxdepth:
            continue
        for card in shelf:
            nxt = cur
            for op in card:
                nxt = [_apply(op, l) for l in nxt]
            used += len(card) * 4
            key = tuple(tuple(l) for l in nxt)
            if key in seen:
                continue
            seen.add(key)
            p2 = prog + card
            if nxt == tgt:
                return list(p2), all(run(p2, ins[i]) == outs[i] for i in range(4, 8)), used
            cnt += 1
            heapq.heappush(heap, (d + 1, cnt, p2, nxt))
            if used >= budget:
                break
    return None, False, used


_G = {}


def _work(args):
    i, budget = args
    return search(_G['tasks'][i], _G['shelf'], budget)


def solve_all(tasks, shelf, budget, procs):
    _G['tasks'] = tasks; _G['shelf'] = shelf
    with mp.get_context('fork').Pool(procs) as pool:
        return pool.map(_work, [(i, budget) for i in range(len(tasks))], chunksize=4)


def evaluate(tasks, shelf, procs):
    r = {}
    for b in BUDGETS:
        res = solve_all(tasks, shelf, b, procs)
        r['budget=%d' % b] = float(np.mean([ok for _, ok, _ in res]))
    return r


def sleep(solved_progs, shelf, add_n=3, min_count=6):
    """解けた課題の操作列から、よく出る並びを探し、棚に足す(圧縮できる量が大きいものから)"""
    cnt = {}
    for p in solved_progs:
        for n in (2, 3, 4):
            for i in range(len(p) - n + 1):
                g = tuple(p[i:i + n]); cnt[g] = cnt.get(g, 0) + 1
    have = set(shelf)
    cands = sorted(((c * (len(g) - 1), g) for g, c in cnt.items() if c >= min_count and g not in have), reverse=True)
    new = []
    for gain, g in cands:
        if len(new) >= add_n:
            break
        # すでに足した並びの一部(または全体を含む)ばかりを避ける
        if any(g in n or n in g for n in new):
            continue
        new.append(g)
    return shelf + new, new



def sleep_measured(solved_progs, shelf, pool, procs, add_n=3, ncand=24, sub=240, budget=5000):
    """候補の並びを1つずつ試しに棚へ足し、同じ予算で解ける課題が実際に増えたものだけ残す"""
    cnt = {}
    for p in solved_progs:
        for n in (2, 3, 4):
            for i in range(len(p) - n + 1):
                g = tuple(p[i:i + n]); cnt[g] = cnt.get(g, 0) + 1
    have = set(shelf)
    cands = [g for _, g in sorted(((c * (len(g) - 1), g) for g, c in cnt.items() if c >= 4 and g not in have), reverse=True)[:ncand]]
    subset = pool[:sub]
    def score(sh):
        return sum(ok for _, ok, _ in solve_all(subset, sh, budget, procs))
    base = score(shelf); new = []
    for _ in range(add_n):
        best, bg = score(shelf), None
        for g in cands:
            if g in new or g in shelf:
                continue
            sc = score(shelf + [g])
            if sc > best:
                best, bg = sc, g
        if bg is None:
            break
        # 別の課題(選ぶのに使っていない)で確かめて、良くならなければ取り消す
        val = pool[sub:2 * sub]
        def vscore(sh):
            return sum(ok for b in (5000, 20000) for _, ok, _ in solve_all(val, sh, b, procs))
        if vscore(shelf + [bg]) <= vscore(shelf):
            cands = [g for g in cands if g != bg]; continue
        shelf = shelf + [bg]; new.append(bg); base = best
    return shelf, new

# ---------------- LLM型の比較対象: GRUが答えを生成 ----------------
def enc_examples(ins, outs):
    v = np.zeros((4, 2, 10, 11), np.float32)
    for i in range(4):
        for j, l in enumerate((ins[i], outs[i])):
            for p in range(10):
                v[i, j, p, l[p] if p < len(l) else 10] = 1
    return v.reshape(-1)


class Gru(torch.nn.Module):
    def __init__(self, H=256):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(4 * 2 * 10 * 11, 512), torch.nn.ReLU(), torch.nn.Linear(512, H))
        self.emb = torch.nn.Embedding(NOP + 2, 64)    # NOP=開始, NOP+1=終了
        self.gru = torch.nn.GRU(64, H, batch_first=True)
        self.out = torch.nn.Linear(H, NOP + 1)         # 0..NOP-1 操作, NOP=終了
        self.H = H


MAXLEN = 14


def train_gru(tasks, epochs, seed=0):
    torch.manual_seed(seed)
    m = Gru()
    X = torch.from_numpy(np.stack([enc_examples(t['ins'], t['outs']) for t in tasks]))
    Y = torch.full((len(tasks), MAXLEN + 1), NOP, dtype=torch.long)
    for i, t in enumerate(tasks):
        for j, op in enumerate(t['prog'][:MAXLEN]):
            Y[i, j] = op
    inp = torch.cat([torch.full((len(tasks), 1), NOP, dtype=torch.long), Y[:, :-1]], 1)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        for i in torch.randperm(len(X)).split(128):
            h0 = m.enc(X[i])[None]
            o, _ = m.gru(m.emb(inp[i]), h0)
            loss = torch.nn.functional.cross_entropy(m.out(o).reshape(-1, NOP + 1), Y[i].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def gru_solve(m, task, budget, rng, batch=64):
    """サンプリングして実行・検証。基本操作を budget まで実行する"""
    ins, outs = task['ins'], task['outs']
    x = torch.from_numpy(enc_examples(ins, outs))[None]
    used = 0
    with torch.no_grad():
        h = m.enc(x)[None].repeat(1, batch, 1)
        tok = torch.full((batch, 1), NOP, dtype=torch.long); seqs = [[] for _ in range(batch)]; alive = np.ones(batch, bool)
        for t in range(MAXLEN + 1):
            o, h = m.gru(m.emb(tok), h)
            p = torch.softmax(m.out(o[:, -1]), 1).numpy().astype(np.float64); p /= p.sum(1, keepdims=True)
            nxt = np.array([rng.choice(NOP + 1, p=p[b]) for b in range(batch)])
            for b in range(batch):
                if alive[b]:
                    if nxt[b] == NOP:
                        alive[b] = False
                    else:
                        seqs[b].append(int(nxt[b]))
            tok = torch.from_numpy(np.where(nxt == NOP, 0, nxt))[:, None]
    return seqs


def gru_eval(m, tasks, budget, rng):
    ok = 0
    for t in tasks:
        used = 0; hit = False
        while used < budget and not hit:
            for s in gru_solve(m, t, budget, rng):
                used += 4 * max(len(s), 1)
                if s and all(run(s, t['ins'][i]) == t['outs'][i] for i in range(4)):
                    hit = all(run(s, t['ins'][i]) == t['outs'][i] for i in range(4, 8)); break
                if used >= budget:
                    break
            else:
                continue
            break
        ok += hit
    return ok / len(tasks)


def main():
    small = len(sys.argv) > 1 and sys.argv[1] == 'small'
    procs = os.cpu_count() or 2
    SEED = int(os.environ.get('SEED', '0'))
    rng = np.random.default_rng(SEED)
    macros, forb = make_world(SEED)
    nT, nU, nL, rounds, gep = (40, 300, 300, 2, 15) if small else (150, 2400, 3000, 6, 40)
    test = make_tasks(nT, rng, macros, forb, test=True)
    pool = make_tasks(nU, rng, macros, forb, test=False)
    labeled = make_tasks(nL, rng, macros, forb, test=False)
    res = {'info': dict(true_macros=[list(m) for m in macros], test_tasks=nT, unlabeled_pool=nU, labeled_for_baseline=nL)}
    t0 = time.time()
    shelf = [(i,) for i in range(NOP)]
    res['round0_primitives_only'] = evaluate(test, shelf, procs)
    print('round0', res['round0_primitives_only'], '%.0fs' % (time.time() - t0), flush=True)
    for r in range(1, rounds + 1):
        out = solve_all(pool, shelf, BUDGETS[1], procs)
        solved = [p for p, ok, _ in out if p is not None]
        shelf, new = sleep_measured(solved, shelf, pool, procs) if os.environ.get('MEASURE') else sleep(solved, shelf)
        ev = evaluate(test, shelf, procs)
        rec = sum(1 for m in macros if tuple(m) in set(shelf))
        res['round%d' % r] = dict(solved_in_pool='%d/%d' % (len(solved), len(pool)), new_cards=[list(c) for c in new], shelf_size=len(shelf), true_macros_found_exactly=rec, **ev)
        print('round%d' % r, res['round%d' % r], '%.0fs' % (time.time() - t0), flush=True)
    # LLM型の比較
    g = train_gru(labeled, gep)
    grng = np.random.default_rng(1)
    res['baseline_gru_labeled%d' % nL] = {'budget=%d' % b: gru_eval(g, test[:(40 if not small else 20)], b, grng) for b in BUDGETS}
    print('baseline', res['baseline_gru_labeled%d' % nL], '%.0fs' % (time.time() - t0), flush=True)
    print('RESULT ' + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
