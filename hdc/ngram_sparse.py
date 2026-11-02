"""n-gram の 「次の語の分布」 を 疎な形で取り出す。
  P(y) = s * cont_p[y] + delta[y]    (s: 何も見つからない語にかかる倍率, delta: 高次で見えた語の上乗せ)
  engine.Model.logprob_at と 同じ確率 (正規化はしない)。
使い方: sparse_dist(model, T, i) -> (s, ids, vals)   T: uint16 の配列, i: 予測する位置
"""
import numpy as np
import engine as E


def sparse_dist(model, T, i, topk=None):
    arr = T; s = 1.0; ids = np.zeros(0, np.int64); vals = np.zeros(0)
    for n in model.orders[1:]:
        h = E.ctx_hash(arr, np.array([i]), n - 1, n)[0]; t = model.t[n]
        j = int(np.searchsorted(t['uc'], h))
        if j >= len(t['uc']) or t['uc'][j] != h: continue
        lo = int(np.searchsorted(t['k'], np.uint64(h) << np.uint64(16))); hi = int(np.searchsorted(t['k'], (np.uint64(h) + np.uint64(1)) << np.uint64(16)))
        a_ids = (np.asarray(t['k'][lo:hi]) & np.uint64(0xFFFF)).astype(np.int64); a_c = np.asarray(t['c'][lo:hi]).astype(np.float64)
        N, Tn, X = float(t['N'][j]), float(t['T'][j]), float(t['X'][j]); lm_ = model.lam[n]
        den = max(N - X + lm_ * (Tn + X), 1e-9); c1 = lm_ * (Tn + X) / den
        s *= c1; vals = vals * c1
        if len(a_ids):
            allid = np.concatenate([ids, a_ids]); allv = np.concatenate([vals, a_c / den])
            ids, inv = np.unique(allid, return_inverse=True); vals = np.bincount(inv, weights=allv, minlength=len(ids))
    if topk is not None and len(ids) > topk:
        p = s * model.cont_p[ids] + vals; o = np.argsort(-p)[:topk]; ids, vals = ids[o], vals[o]
    return s, ids, vals


def prob_of(model, s, ids, vals, y):
    """分布 (s, ids, vals) での 語 y の確率"""
    p = s * model.cont_p[y]
    j = np.searchsorted(ids, y)
    if len(ids) and j < len(ids) and ids[j] == y: p += vals[j]
    return p
