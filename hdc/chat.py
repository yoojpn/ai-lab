"""会話の試作に話しかける。 n-gram(1..5, WB + 捨てた質量) + キャッシュ(直近 256 語) の確率から 1 語ずつ標本抽出する。 勾配なし・数え上げのみ。
使い方: python3 chat.py '{"history":["こんにちは","..."],"temp":0.8,"topk":40,"seed":0,"n":3}'   history = これまでの発言 (自分/相手 交互, 最後が自分の発言)
"""
import sys, json
import numpy as np
import engine as E
from tokenizers import Tokenizer

W = '/tmp/tok'; V = 32768


def next_dist(model, T):
    i = len(T); arr = np.r_[T, 0].astype(np.uint16); P = model.uni_p.copy()
    for n in model.orders[1:]:
        h = E.ctx_hash(arr, np.array([i]), n - 1, n)[0]; t = model.t[n]
        j = int(np.searchsorted(t['uc'], h))
        if j >= len(t['uc']) or t['uc'][j] != h: continue
        lo = int(np.searchsorted(t['k'], np.uint64(h) << np.uint64(16))); hi = int(np.searchsorted(t['k'], (np.uint64(h) + np.uint64(1)) << np.uint64(16)))
        add = np.zeros(V); add[(np.asarray(t['k'][lo:hi]) & np.uint64(0xFFFF)).astype(np.int64)] = np.asarray(t['c'][lo:hi]).astype(np.float64)
        N, Tn, X = float(t['N'][j]), float(t['T'][j]), float(t['X'][j])
        P = (add + (Tn + X) * P) / (N + Tn)
    return P / P.sum()


def main(cfg):
    tk = Tokenizer.from_file(W + '/tok.json'); model = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)})
    rng = np.random.default_rng(cfg.get('seed', 0)); temp = cfg.get('temp', 0.8); topk = cfg.get('topk', 40); g = cfg.get('cache', 0.15)
    base = [0] * 8 + [0]
    for k, t in enumerate(cfg['history']):
        base += tk.encode(t).ids + [2]
    outs = []
    for r in range(cfg.get('n', 3)):
        T = list(base); gen = []
        for step in range(cfg.get('maxlen', 60)):
            P = next_dist(model, np.array(T, np.uint16))
            win = [x for x in T[-256:] if x > 2]
            if len(win) > 8:
                pc = np.bincount(win, minlength=V) / len(win); P = (1 - g) * P + g * pc
            P[0] = 0
            if step < 2: P[1] = 0; P[2] = 0
            idx = np.argsort(-P)[:topk]; p = P[idx] ** (1.0 / temp); p /= p.sum(); x = int(rng.choice(idx, p=p))
            if x in (1, 2): break
            T.append(x); gen.append(x)
        outs.append(tk.decode(gen))
    for o in outs: print('> ' + o)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
