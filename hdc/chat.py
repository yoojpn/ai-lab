"""会話の試作に話しかける。 n-gram(1..5, WB + 捨てた質量) + キャッシュ(直近 256 語) の確率から 1 語ずつ標本抽出する。 勾配なし・数え上げのみ。
使い方: python3 chat.py '{"history":["こんにちは","..."],"temp":0.8,"topk":40,"seed":0,"n":3}'   history = これまでの発言 (自分/相手 交互, 最後が自分の発言)
"""
import sys, json
import numpy as np
import engine as E
from tokenizers import Tokenizer

W = '/tmp/tok'; V = 32768


def next_dist(model, T):
    i = len(T); arr = np.r_[T, 0].astype(np.uint16); P = (model.cont_p if model.kn_base else model.uni_p).copy()
    for n in model.orders[1:]:
        h = E.ctx_hash(arr, np.array([i]), n - 1, n)[0]; t = model.t[n]
        j = int(np.searchsorted(t['uc'], h))
        if j >= len(t['uc']) or t['uc'][j] != h: continue
        lo = int(np.searchsorted(t['k'], np.uint64(h) << np.uint64(16))); hi = int(np.searchsorted(t['k'], (np.uint64(h) + np.uint64(1)) << np.uint64(16)))
        add = np.zeros(V); add[(np.asarray(t['k'][lo:hi]) & np.uint64(0xFFFF)).astype(np.int64)] = np.asarray(t['c'][lo:hi]).astype(np.float64)
        N, Tn, X = float(t['N'][j]), float(t['T'][j]), float(t['X'][j])
        lm_ = model.lam[n]; P = (add + lm_ * (Tn + X) * P) / (N - X + lm_ * (Tn + X))
    return P / P.sum()


def dist_with_cache(model, T, g):
    P = next_dist(model, np.array(T, np.uint16)); win = [x for x in T[-256:] if x > 2]
    if len(win) > 8:
        pc = np.bincount(win, minlength=V) / len(win); P = (1 - g) * P + g * pc
    P = P.copy(); P[0] = 0; return P / P.sum()


def blocked(gen, n=3):
    """直前に出した n 語の並びを もう一度出す語 (と 直前と同じ語) を禁止"""
    b = set()
    if gen: b.add(gen[-1])
    if len(gen) >= n:
        pre = tuple(gen[-(n - 1):])
        for i in range(len(gen) - n + 1):
            if tuple(gen[i:i + n - 1]) == pre: b.add(gen[i + n - 1])
    return b


def beam_search(model, base, bw, g, maxlen, alpha=0.7):
    beams = [(0.0, [])]; done = []
    for step in range(maxlen):
        cand = []
        for sc, gen in beams:
            P = dist_with_cache(model, base + gen, g)
            if step < 2: P[1] = 0; P[2] = 0
            for x in blocked(gen): P[x] = 0
            P = P / P.sum()
            for x in np.argsort(-P)[:bw]: cand.append((sc + float(np.log(P[x])), gen + [int(x)], int(x)))
        cand.sort(key=lambda c: -c[0]); beams = []
        for sc, gen, x in cand:
            if x in (1, 2): done.append((sc / (len(gen) ** alpha), gen[:-1]))
            else: beams.append((sc, gen))
            if len(beams) >= bw: break
        if done and beams and max(d[0] for d in done) > max(b[0] / ((len(b[1]) + 1) ** alpha) for b in beams): break
    done += [(sc / (max(1, len(gen)) ** alpha), gen) for sc, gen in beams]
    done.sort(key=lambda d: -d[0]); return [g_ for _, g_ in done]


def main(cfg):
    tk = Tokenizer.from_file(W + '/tok.json'); lam = {int(k): v for k, v in json.load(open(W + '/lam.json')).items()}; model = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    rng = np.random.default_rng(cfg.get('seed', 0)); temp = cfg.get('temp', 0.8); topk = cfg.get('topk', 40); g = cfg.get('cache', 0.15)
    base = [0] * 8 + [0]
    for k, t in enumerate(cfg['history']):
        base += tk.encode(t).ids + [2]
    outs = []
    if cfg.get('beam'):
        for gen in beam_search(model, base, cfg['beam'], g, cfg.get('maxlen', 60))[:cfg.get('n', 3)]: outs.append(tk.decode(gen))
        for o in outs: print('> ' + o)
        return
    for r in range(cfg.get('n', 3)):
        T = list(base); gen = []
        for step in range(cfg.get('maxlen', 60)):
            P = next_dist(model, np.array(T, np.uint16))
            win = [x for x in T[-256:] if x > 2]
            if len(win) > 8:
                pc = np.bincount(win, minlength=V) / len(win); P = (1 - g) * P + g * pc
            P[0] = 0
            if step < 2: P[1] = 0; P[2] = 0
            if cfg.get('norep'):
                for b in blocked(gen): P[b] = 0
            if temp <= 0: x = int(np.argmax(P))
            else:
                idx = np.argsort(-P)[:topk]; p = P[idx] ** (1.0 / temp); p /= p.sum(); x = int(rng.choice(idx, p=p))
            if x in (1, 2): break
            T.append(x); gen.append(x)
        outs.append(tk.decode(gen))
    for o in outs: print('> ' + o)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
