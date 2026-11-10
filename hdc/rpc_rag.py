"""検索つきの 1 語ずつ生成 (数え上げのみ): 前の発言に合う返事を 検索 (結びつき PPMI) で上位 K 個 取り、 その語の並びを 小さな n-gram (次数 1-3) にして
生成の確率に混ぜる (kNN-LM 型)。 全体の n-gram (次数 1-6) + 前の発言との結びつき の上に 足す。 使い方: python3 rpc_rag.py '発話1' '発話2' ..."""
import sys, collections
import numpy as np
import rpc_gen as G, rpc_chat as RC, chat


def mini_lm(texts):
    c = [collections.defaultdict(collections.Counter) for _ in range(3)]
    for t in texts:
        ids = [0] + G.enc(t) + [2]
        for i in range(1, len(ids)):
            for k in range(3):
                if i - k - 1 >= 0 or k == 0: c[k][tuple(ids[max(0, i - k - 1):i])[-(k + 1):]][ids[i]] += 1
    return c


def pmini(c, hist):
    for k in (2, 1, 0):
        if len(hist) >= k + 1 or k == 0:
            h = tuple(hist[-(k + 1):]); d = c[k].get(h)
            if d: return d, k
    return None, -1


def generate(hist, K=8, beta=1.0, lam=(0.35, 0.55, 0.75), temp=0.8, topk=30, maxlen=40, seed=0):
    rng = np.random.default_rng(seed); last = hist[-1]; cand = RC.reply(last, K); c = mini_lm(cand); T = [0] * 9
    for u in hist: T += G.enc(u) + [2]
    av = G.assoc(G.enc(last)); gen = []; mh = [0]
    for step in range(maxlen):
        P = G.dist(np.array(T, np.uint16), av, beta); d, k = pmini(c, mh)
        if d is not None:
            tot = sum(d.values()); q = np.zeros(len(P)); [q.__setitem__(w, n / tot) for w, n in d.items()]; l = lam[k]; P = (1 - l) * P + l * q
        P[0] = 0
        if step < 1: P[2] = 0
        for b in chat.blocked(gen, 3): P[b] = 0
        idx = np.argsort(-P)[:topk]; p = P[idx] ** (1 / temp); p /= p.sum(); x = int(rng.choice(idx, p=p))
        if x == 2: break
        T.append(x); gen.append(x); mh.append(x)
    return G.tk.decode(gen), cand[:2]


if __name__ == '__main__':
    for q in sys.argv[1:]:
        print('あなた: %s' % q)
        for s in range(3): a, cand = generate([q], seed=s); print('   返事: %s' % a)
        print('   (検索の上位: %s / %s)' % tuple(cand[:2]))
