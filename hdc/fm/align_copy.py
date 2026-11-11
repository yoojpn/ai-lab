"""アライメント帰納: 履歴の語の 直前の窓 と、 今の生成の 直前の窓 の 語どうしの 対応 (my↔your, 私↔あなた 等) を 数えて 学び、
コピー候補の 点数 にする。 数え上げのみ。  score(j) = sum_d S_d[H[j-d], R[-d]],  特徴 F(w) = log sum_{j: H[j]=w} exp(score_j)
S_d は 『コピーが 起きた (履歴 j と 今 t で 語が 同じ) 組』 と 『起きなかった 組』 の 窓の語対の 回数の 対数比。"""
import numpy as np, json, sys, os
V = 32768; D = 3


def load_convs(paths, tk, maxu=300, maxlen=1500):
    convs = []
    for p in paths:
        for l in open(p, encoding='utf-8'):
            d = json.loads(l); c = d['conversations']; u = [(x.get('value') or x.get('content') or '') for x in c]
            if len(u) >= 2: convs.append(u)
    ids = []
    flat = [x[:3000] for u in convs for x in u]; enc = tk.encode_batch(flat); k = 0
    for u in convs:
        s = [0]; starts = []
        for _ in u:
            starts.append(len(s)); s += enc[k].ids[:maxu] + [2]; k += 1
        if len(s) <= maxlen: ids.append((np.array(s, np.int64), starts))
        else: ids.append((np.array(s[:maxlen], np.int64), [x for x in starts if x < maxlen]))
    return ids


def pairs(S, starts, rng, tstep=2, cap=12):
    """学習/評価用の (t, j) 組: t は 2 発話目以降の 位置、 j は 前の発話の中で 同じ語の位置 (正) か 違う語の位置 (負)"""
    out = []
    for ui in range(1, len(starts)):
        s0 = starts[ui]; e = starts[ui + 1] if ui + 1 < len(starts) else len(S)
        if s0 < 4: continue
        hist = S[:s0]
        for t in range(s0 + 1, e, tstep):
            w = S[t]; pos = np.where(hist == w)[0]; pos = pos[pos >= D]
            neg = rng.integers(D, s0, size=min(cap, 4))
            neg = neg[hist[neg] != w]
            if len(pos) > cap: pos = rng.choice(pos, cap, replace=False)
            out.append((t, s0, pos, neg))
    return out


class Align:
    def __init__(self): self.S = {}; self.diag = np.zeros(D)

    def fit(self, convs, seed=0, a=0.5, chunk=2000):
        rng = np.random.default_rng(seed); P = [(np.zeros(0, np.int64), np.zeros(0, np.int64)) for _ in range(D)]; N = [(np.zeros(0, np.int64), np.zeros(0, np.int64)) for _ in range(D)]
        tot = np.zeros((D, 4))
        def merge(old, keys):
            u, c = np.unique(keys, return_counts=True); k = np.concatenate([old[0], u]); v = np.concatenate([old[1], c]); uu, inv = np.unique(k, return_inverse=True)
            return uu, np.bincount(inv, weights=v).astype(np.int64)
        for s0 in range(0, len(convs), chunk):
            K = [[] for _ in range(D)]; Kn = [[] for _ in range(D)]
            for S, starts in convs[s0:s0 + chunk]:
                for t, st, pos, neg in pairs(S, starts, rng):
                    for d in range(D):
                        r = S[t - d - 1]
                        if len(pos): K[d].append(S[pos - d - 1] * V + r)
                        if len(neg): Kn[d].append(S[neg - d - 1] * V + r)
            for d in range(D):
                kp = np.concatenate(K[d]); kn = np.concatenate(Kn[d]); tot[d] += [len(kp), len(kn), (kp // V == kp % V).sum(), (kn // V == kn % V).sum()]
                P[d] = merge(P[d], kp); N[d] = merge(N[d], kn)
            del K, Kn
        for d in range(D):
            up, cp = P[d]; un, cn = N[d]; Np, Nn = tot[d, 0], tot[d, 1]
            allk = np.union1d(up, un); fp = np.zeros(len(allk)); fn = np.zeros(len(allk)); fp[np.searchsorted(allk, up)] = cp; fn[np.searchsorted(allk, un)] = cn
            sc = np.log((fp + a) / (Np + a * 1e3)) - np.log((fn + a) / (Nn + a * 1e3)); keep = (fp + fn) >= 3
            self.S[d] = (allk[keep], sc[keep].astype(np.float32)); self.diag[d] = np.log((tot[d, 2] / Np + 1e-3) / (tot[d, 3] / Nn + 1e-3))
        return self

    def lookup(self, d, h, r):
        keys, sc = self.S[d]; q = h * V + r; i = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1); hit = keys[i] == q
        return np.where(hit, sc[i], np.where(h == r, self.diag[d], 0.0))

    def cand_scores(self, ctx, hist_end, hist_start=0):
        """今の文脈 ctx の 末尾 と、 履歴 [hist_start,hist_end) の 各位置 j の 窓の 対応点 (j は D 以上)"""
        j = np.arange(max(D, hist_start), hist_end); sc = np.zeros(len(j))
        for d in range(D): sc += self.lookup(d, ctx[j - d - 1], int(ctx[-d - 1]))
        return j, sc

    def feature(self, ctx, hist_end, temp=1.0):
        j, sc = self.cand_scores(ctx, hist_end)
        F = np.zeros(V); np.add.at(F, ctx[j], np.exp((sc - sc.max()) / temp)); return F, (sc.max() if len(sc) else 0.0)


def evaluate(al, convs, seed=1, diag_only=False):
    """正解の語が 履歴の中に ある位置で、 履歴の語だけを 候補に 点数から 作った 分布での 平均対数確率 / 先頭一致率"""
    rng = np.random.default_rng(seed); tot = 0; n = 0; top1 = 0; uni = 0
    for S, starts in convs:
        for ui in range(1, len(starts)):
            s0 = starts[ui]; e = starts[ui + 1] if ui + 1 < len(starts) else len(S)
            if s0 < 4: continue
            for t in range(s0 + 1, e, 3):
                w = S[t]
                if not (S[:s0] == w).any(): continue
                j, sc = al.cand_scores(S[:t], s0)
                if diag_only: sc = sum(((S[j - d - 1] == S[t - d - 1]) * 1.0) for d in range(D))
                if len(j) == 0: continue
                F = np.zeros(V); np.add.at(F, S[j], np.exp(sc - sc.max())); p = F[w] / F.sum()
                tot += np.log(max(p, 1e-9)); n += 1; top1 += F.argmax() == w; uni += np.log(max((S[j] == w).sum() / len(j), 1e-9))
    return tot / n, top1 / n, uni / n, n
