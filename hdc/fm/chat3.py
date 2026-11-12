"""会話エンジン v3: 索引の確率 P に 3 つの 特徴を 対数で足す (数え上げのみ)。
  s_doc: 履歴の特徴 (chat2.Chat2.score)、 s_al: アライメント帰納 log1p(F) (align_copy)、 s_dg: 会話だけの n-gram の 比 log(Pd/P) (±8 で 切る)
  P'(w) = P(w) exp(w1 s_doc + w2 s_al + w3 s_dg) / Z"""
import numpy as np
from chat2 import Chat2, V


class Chat3(Chat2):
    def __init__(self, dist, Em, uni, lam, CL64, tk, al, ddist, w=(0.5, 1.0, 0.75)):
        super().__init__(dist, Em, uni, lam, CL64, tk); self.al, self.dd, self.w = al, ddist, list(w); self.absf = True

    def feats(self, ctx, hist_end, P):
        c = np.asarray(ctx, np.int64); f = np.zeros((3, V), np.float32); f[0] = self.score(ctx)
        if self.absf:
            j, sc = self.al.cand_scores(c, hist_end); Fa = np.zeros(V); np.add.at(Fa, c[j], np.exp(sc)); f[1] = np.log1p(Fa)
        else:
            Fa, _ = self.al.feature(c, hist_end); f[1] = np.log1p(Fa)
        f[2] = np.clip(np.log(np.maximum(self.dd(list(ctx)), 1e-12)) - np.log(np.maximum(P, 1e-12)), -8, 8); return f

    def dist3(self, ctx, hist_end, w=None, late=True):
        P = self.dist(ctx).astype(np.float64); f = self.feats(ctx, hist_end, P); w = self.w if w is None else w
        if len(w) == 6: w = w[3:] if late else w[:3]
        s = sum(w[i] * f[i] for i in range(3)); e = P * np.exp(s - s.max()); return e / e.sum()

    def reply3(self, utts, temp=0.7, topk=40, maxlen=80, seed=0, plain=False):
        rng = np.random.default_rng(seed); ctx = [0]
        for u in utts: ctx += self.tk.encode(u).ids + [2]
        h0 = len(ctx); gen = []
        for step in range(maxlen):
            full = ctx + gen
            P = np.asarray(self.dist(full) if plain else self.dist3(full, h0, late=step >= 8), np.float64).copy(); P[0] = 0
            if step < 2: P[2] = 0
            for i in range(len(gen) - 2):
                if gen[i:i + 2] == gen[-2:]: P[gen[i + 2]] = 0
            for x, n in zip(*np.unique(gen[-40:], return_counts=True)):
                if n >= 2: P[x] *= 0.3 ** (n - 1)
            idx = np.argsort(-P)[:topk]; q = P[idx] ** (1 / temp); q /= q.sum(); x = int(rng.choice(idx, p=q))
            if x == 2: break
            gen.append(x)
        return self.tk.decode(gen)
