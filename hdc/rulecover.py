"""
規則被覆 (rule cover): 役割を教えずに、 「確実に当たる文脈」 を数え上げで集めていく。 勾配なし。
  1. まだ説明できていない位置から標本を取り、 半分で表を作り、 半分で検証する。
  2. 特徴の部分集合 (ビーム探索で大きさ 1..maxk) ごとに、 純度 >= purity かつ 件数 >= minn の文脈だけを「規則」とし、
     検証で  当たった数 - wrong_pen * 外れた数  が最大の部分集合を選ぶ。
  3. 選んだ部分集合で、 残り全体から確実な文脈を表にし、 説明できた位置を除く。 これを繰り返す。
予測: 規則を順に当て、 最初に文脈が見つかった規則の語を返す。 どれにも当たらなければ 未説明 (-1)。
"""
import numpy as np


def pack(cols, sub):
    k = np.zeros(len(next(iter(cols.values()))), np.int64)
    for f in sub:
        k = k * 256 + cols[f]
    return k


def conf_table(key, y, purity, minn, ny=16384):
    kl = key * ny + y
    u, c = np.unique(kl, return_counts=True); ku = u // ny; lu = u % ny
    starts = np.r_[0, np.flatnonzero(ku[1:] != ku[:-1]) + 1]
    tot = np.add.reduceat(c, starts); mx = np.maximum.reduceat(c, starts)
    o = np.lexsort((-c, ku)); best = lu[o][starts]                    # 文脈ごとに最も多い語 (ku でそろえた先頭)
    ok = (tot >= minn) & (mx >= purity * tot)
    return ku[starts][ok], best[ok]


def lookup(tab, key):
    ku, lu = tab
    if len(ku) == 0:
        return np.full(len(key), -1, np.int64)
    i = np.minimum(np.searchsorted(ku, key), len(ku) - 1)
    return np.where(ku[i] == key, lu[i], -1)


class RuleCover:
    def __init__(self, maxk=5, beam=10, purity=0.995, minn=3, wrong_pen=3.0, rounds=30, sample=100000, min_gain=100, seed=0, ny=16384):
        self.__dict__.update(locals()); self.rules = []

    def fit(self, F, y, names, log=print, allowed=None):
        rng = np.random.default_rng(self.seed); N, nf = F.shape; y = y.astype(np.int64)
        remaining = np.arange(N); allowed = list(range(nf)) if allowed is None else list(allowed)
        for rd in range(self.rounds):
            if len(remaining) < 1000:
                break
            smp = remaining if len(remaining) <= self.sample else rng.choice(remaining, self.sample, replace=False)
            half = rng.random(len(smp)) < 0.5; A = smp[half]; B = smp[~half]
            cA = {f: F[A, f].astype(np.int64) for f in allowed}; cB = {f: F[B, f].astype(np.int64) for f in allowed}
            yA, yB = y[A], y[B]
            def score(kA, kB):
                tab = conf_table(kA, yA, self.purity, self.minn)
                p = lookup(tab, kB); hit = p >= 0; good = int((p[hit] == yB[hit]).sum()); bad = int(hit.sum()) - good
                return good - self.wrong_pen * bad, good, bad
            best = (-1e18, None, 0, 0); frontier = [((), np.zeros(len(A), np.int64), np.zeros(len(B), np.int64))]; seen = set()
            for k in range(1, self.maxk + 1):
                cands = []
                for sub, pa, pb in frontier:
                    for f in allowed:
                        if f in sub:
                            continue
                        s2 = tuple(sorted(sub + (f,)))
                        if s2 in seen:
                            continue
                        seen.add(s2)
                        kA = pa * 256 + cA[f]; kB = pb * 256 + cB[f]
                        sc, g, b = score(kA, kB); cands.append((sc, s2 if False else sub + (f,), kA, kB, g, b))
                if not cands:
                    break
                cands.sort(key=lambda t: -t[0])
                if cands[0][0] > best[0] + 1e-9:
                    best = (cands[0][0], cands[0][1], cands[0][4], cands[0][5])
                frontier = [(c[1], c[2], c[3]) for c in cands[:self.beam]]
            if best[1] is None or best[0] < self.min_gain * len(smp) / 100000.0:
                log('  [被覆] 追加なし (最良 %.0f)' % best[0]); break
            sub = tuple(sorted(best[1]))
            cols = {f: F[remaining, f].astype(np.int64) for f in sub}
            key = pack(cols, sub); tab = conf_table(key, y[remaining], self.purity, self.minn)
            p = lookup(tab, key); hit = p >= 0
            self.rules.append((sub, tab)); ncov = int(hit.sum()); nok = int((p[hit] == y[remaining][hit]).sum())
            log('  [被覆] 規則%d: %s | 検証 +%d/-%d | 全体で説明 %d (正解 %d) / 残り %d' % (len(self.rules), [names[f] for f in sub], best[2], best[3], ncov, nok, len(remaining)))
            remaining = remaining[~hit]
        self.names = names
        return self

    def predict(self, F):
        N = len(F); pred = np.full(N, -1, np.int64); rid = np.full(N, -1, np.int64)
        for r, (sub, tab) in enumerate(self.rules):
            todo = np.flatnonzero(pred < 0)
            if len(todo) == 0:
                break
            cols = {f: F[todo, f].astype(np.int64) for f in sub}
            p = lookup(tab, pack(cols, sub)); h = p >= 0
            pred[todo[h]] = p[h]; rid[todo[h]] = r
        return pred, rid
