"""
core.py: 領域に依存しない 特徴の自動生成 + 語クラスの発見 (G2 の一般化)。 入力は 離散記号の列 T (uint16) と 文書の開始位置 docs だけ。
  領域の知識は 引数で渡す:  seps = 区切りの記号の id (会話なら <turn>,<eos>; コードなら NEWLINE)。 渡さなければ区切りの特徴は無し。
  特徴 (全部 数え上げで作る; 値は 0..251 の 1 バイト):
    off1..off6        直前の k 語 (頻出 240 語は そのまま、 他は 250)
    sk8..sk64         離れた位置の語 (8,12,16,24,32,48,64 語前)
    c1..c3            直前 3 語の「語クラス」 (G2: 周りの語の分布で自動的にまとめた 200 クラス)
    since_sep, turn_idx   区切りからの距離、 文書内の区切りの数 (会話なら発言の順番)
    cur_first, prev_first, prev_last   今の発言の最初の語 / 前の発言の最初と最後の語 (c_ は クラス版)
    ind1, ind2        直前1語/2語の 直近の前回出現の「次の語」 (窓 128 語内)  = コピー・参照
    topic             直近 128 語に 2 回以上出た 珍しい語 (ハッシュ) = 話題
  features(T, pos, st, fm, cls) -> F (n x nf, int16), names, cp (n x 2: ind1, ind2 の生の id, なければ -1)
  位置 pos は 任意の位置の配列 (全部の位置を作らない。 標本で規則を探し、 評価も標本で行う)。
"""
import numpy as np
import time

OTHER = 250; NONE = 251; BOS_F = 243; EOS_F = 241; TURN_F = 242
WIN = 128


def count_tokens(T, V, chunk=50_000_000):
    c = np.zeros(V, np.int64)
    for a in range(0, len(T), chunk):
        c += np.bincount(np.asarray(T[a:a + chunk]), minlength=V)
    return c


def make_fmap(counts, topk=240):
    V = len(counts); fm = np.full(V, OTHER, np.uint8)
    order = np.argsort(-counts, kind='stable'); order = order[order > 2]; top = order[:topk]
    fm[top] = np.arange(1, len(top) + 1, dtype=np.uint8)
    fm[0] = BOS_F; fm[1] = EOS_F; fm[2] = TURN_F
    return fm


class Struct:
    """文書の開始位置と 区切りの位置 (全体を 1 回走査)"""
    def __init__(self, T, docs, seps, chunk=50_000_000):
        self.docs = np.asarray(docs, np.int64); self.seps = list(seps); sp = []
        if seps:
            for a in range(0, len(T), chunk):
                x = np.asarray(T[a:a + chunk]); sp.append(np.flatnonzero(np.isin(x, seps)) + a)
        self.seppos = np.concatenate(sp) if sp else np.zeros(0, np.int64)


def learn_classes(T, V, k=200, top=16000, dim=48, ntok=20_000_000, blocks=40, seed=0, log=print):
    """語クラスの発見 (G2 の一般化): 周りの語の分布 (PPMI) を 低次元にして まとめる。 戻り値: cls (V,) uint8"""
    from scipy import sparse
    from scipy.sparse.linalg import svds
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time(); N = len(T); bl = ntok // blocks
    starts = np.linspace(8, N - bl - 8, blocks).astype(np.int64)
    smp = np.concatenate([np.asarray(T[a:a + bl]) for a in starts]).astype(np.int64)
    freq = np.bincount(smp, minlength=V); freq[:3] = 0
    ids = np.argsort(-freq, kind='stable')[:top]; ids = ids[freq[ids] > 0]; K = len(ids)
    rank = np.full(V, -1, np.int64); rank[ids] = np.arange(K)
    r = rank[smp]; M = sparse.csr_matrix((K, 2 * K), dtype=np.float64)
    for d in (1, 2):
        a = r[:-d]; b = r[d:]; ok = (a >= 0) & (b >= 0)
        a = a[ok]; b = b[ok]
        rows = np.concatenate([a, b]); cols = np.concatenate([b, K + a])
        M = M + sparse.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(K, 2 * K)).tocsr()
    M = M.tocsr(); tot = M.sum(); rs = np.asarray(M.sum(1)).ravel() + 1e-9; cs = np.asarray(M.sum(0)).ravel() ** 0.75; cs = cs / cs.sum() * tot + 1e-9
    C = M.tocoo(); v = np.log(C.data * tot / (rs[C.row] * cs[C.col])); keep = v > 0
    P = sparse.csr_matrix((v[keep], (C.row[keep], C.col[keep])), shape=M.shape)
    U, S, _ = svds(P, k=dim, random_state=seed); E = U * np.sqrt(S); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=seed).fit(E)
    cls = np.full(V, k, np.uint8); cls[ids] = km.labels_.astype(np.uint8); cls[0] = BOS_F; cls[1] = EOS_F; cls[2] = TURN_F
    log('語クラス: %d 語 -> %d クラス (%.0f 秒)' % (K, k, time.time() - t0))
    return cls


def features(T, pos, st, fm, cls, win=WIN):
    pos = np.asarray(pos, np.int64); n = len(pos)
    d = np.searchsorted(st.docs, pos, side='right') - 1; S = st.docs[np.maximum(d, 0)]
    F = {}
    def tokat(idx):
        ok = (idx >= S) & (idx < pos)
        return np.where(ok, np.asarray(T[np.clip(idx, 0, len(T) - 1)]).astype(np.int32), -1)
    def f_of(raw, m):
        return np.where(raw >= 0, m[np.clip(raw, 0, len(m) - 1)], NONE).astype(np.int16)
    for o in range(1, 7):
        F['off%d' % o] = f_of(tokat(pos - o), fm)
    for o in (8, 12, 16, 24, 32, 48, 64):
        F['sk%d' % o] = f_of(tokat(pos - o), fm)
    for o in range(1, 4):
        F['c%d' % o] = f_of(tokat(pos - o), cls)
    if len(st.seppos):
        sp = st.seppos; k = np.searchsorted(sp, pos, side='left')                 # pos より前の区切りの数
        p1 = np.where(k >= 1, sp[np.maximum(k - 1, 0)], -1); v1 = p1 >= S
        p2 = np.where(k >= 2, sp[np.maximum(k - 2, 0)], -1); v2 = (k >= 2) & (p2 >= S) & v1
        F['since_sep'] = np.minimum(np.where(v1, pos - p1 - 1, pos - S), 63).astype(np.int16)
        F['turn_idx'] = np.minimum(k - np.searchsorted(sp, S, side='left'), 15).astype(np.int16)
        start = np.where(v1, p1 + 1, S + 1)
        cf = np.where(start < pos, np.asarray(T[np.clip(start, 0, len(T) - 1)]).astype(np.int32), -1)
        pf = np.where(v2, np.asarray(T[np.clip(p2 + 1, 0, len(T) - 1)]).astype(np.int32), -1)
        pl = np.where(v1 & (p1 - 1 > S), np.asarray(T[np.clip(p1 - 1, 0, len(T) - 1)]).astype(np.int32), -1)
        for nm, raw in (('cur_first', cf), ('prev_first', pf), ('prev_last', pl)):
            F[nm] = f_of(raw, fm); F['c_' + nm] = f_of(raw, cls)
    ind1 = np.full(n, -1, np.int32); ind2 = np.full(n, -1, np.int32); topic = np.zeros(n, np.int16)
    B = 100_000
    ar = np.arange(1, win + 1)
    for a in range(0, n, B):
        sl = slice(a, min(n, a + B)); p = pos[sl]; Ss = S[sl]
        idx = p[:, None] - ar[None, :]; ok = idx >= Ss[:, None]
        W = np.where(ok, np.asarray(T[np.clip(idx, 0, len(T) - 1)]).astype(np.int32), -1)          # W[:, j] = 位置 pos-(j+1)
        tl = W[:, 0]
        m1 = (W[:, 1:] == tl[:, None]) & (tl[:, None] >= 0)
        has = m1.any(1); j = m1.argmax(1); r = np.arange(len(p))
        ind1[sl] = np.where(has, W[r, j], -1)                                                       # W 添字 j+1 が一致 -> その「次」= W 添字 j
        m2 = (W[:, 1:-1] == tl[:, None]) & (W[:, 2:] == W[:, 1][:, None]) & (W[:, 1][:, None] >= 0) & (tl[:, None] >= 0)
        has2 = m2.any(1); i2 = m2.argmax(1)
        ind2[sl] = np.where(has2, W[r, i2], -1)
        Wc = np.clip(W, 0, len(fm) - 1); rare = np.where((W > 2) & (fm[Wc] == OTHER), W, 0)
        Rs = np.sort(rare, axis=1); eq = (Rs[:, 1:] == Rs[:, :-1]) & (Rs[:, 1:] > 0); ht = eq.any(1); it = eq.argmax(1)
        tp = Rs[r, it + 1].astype(np.int64)
        topic[sl] = np.where(ht, ((tp * 2654435761) % (2 ** 32) >> 12) % 240 + 1, 0)
    F['ind1'] = f_of(ind1, fm); F['ind2'] = f_of(ind2, fm); F['topic'] = topic
    names = list(F.keys()); M = np.stack([F[nm] for nm in names], axis=1).astype(np.int16)
    return M, names, np.stack([ind1, ind2], axis=1)


def cache_stats(T, pos, st, y, win=256):
    """キャッシュ: 直近 win 語 (同じ文書内) に 実際の語 y が何回出たか と 窓の長さ。 -> (cnt, wl)"""
    pos = np.asarray(pos, np.int64); d = np.searchsorted(st.docs, pos, side='right') - 1; S = st.docs[np.maximum(d, 0)]
    cnt = np.zeros(len(pos), np.int32); wl = np.minimum(win, pos - S - 1).astype(np.int32); ar = np.arange(1, win + 1); B = 20000
    for a in range(0, len(pos), B):
        sl = slice(a, min(len(pos), a + B)); p = pos[sl]; idx = p[:, None] - ar[None, :]; ok = idx > S[sl][:, None]
        W = np.where(ok, np.asarray(T[np.clip(idx, 0, len(T) - 1)]).astype(np.int32), -1)
        cnt[sl] = (W == np.asarray(y)[sl][:, None]).sum(1)
    return cnt, wl


class CacheMix:
    """LM と キャッシュの補間。 補間の割合は (LM の自信度, 窓の長さ) の区分ごとに、 検証データで格子探索 (勾配なし)"""
    GRID = np.r_[0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65]

    @staticmethod
    def bins(plm, wl):
        b1 = np.clip((-np.log2(np.maximum(plm, 1e-12)) // 2).astype(int), 0, 7); b2 = np.digitize(wl, [8, 32, 96, 192]); return b1 * 5 + b2

    def fit(self, plm, cnt, wl):
        b = self.bins(plm, wl); pc = np.where(wl > 0, cnt / np.maximum(wl, 1), 0.0); self.lam = np.zeros(8 * 5)
        for k in range(8 * 5):
            m = b == k
            if m.sum() < 200: continue
            ll = [np.log2((1 - g) * plm[m] + g * pc[m]).mean() if True else 0 for g in self.GRID]
            self.lam[k] = self.GRID[int(np.argmax(ll))]
        return self

    def apply(self, plm, cnt, wl):
        b = self.bins(plm, wl); g = np.where(wl > 0, self.lam[b], 0.0); pc = np.where(wl > 0, cnt / np.maximum(wl, 1), 0.0)
        return (1 - g) * plm + g * pc
