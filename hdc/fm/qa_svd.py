"""質問→答え の 内容語の 対応: 隣り合う 発話 (k, k+1) の 語の 同時出現を 数え、 PMI を 特異値分解 (rank 64) して 質問側/答え側の 埋め込みを 作る。
特徴: 答えの 語 w への 加点 = 答え側 Ea[w] · (質問の 内容語の 質問側 Eq の idf 重み平均)。 勾配なし。"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds
V = 32768


class QA:
    def fit(self, convs, groups=None, K=10000, rank=64, maxtok=40, stop=40, seed=0):
        cnt = np.zeros(V)
        for S, st in convs: cnt += np.bincount(S, minlength=V)
        order = np.argsort(-cnt); keep = [t for t in order[:K + stop + 4] if t not in (0, 2, 3)][stop:stop + K]; self.vocab = np.array(keep); self.map = -np.ones(V, np.int64); self.map[self.vocab] = np.arange(len(keep)); n = len(keep)
        C = sp.csr_matrix((n, n), dtype=np.float32); rows = []; cols = []; npair = 0
        def flush():
            nonlocal C, rows, cols
            if rows: r = np.concatenate(rows); c = np.concatenate(cols); C = C + sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n)).tocsr(); rows = []; cols = []
        for S, st in convs:
            for ui in range(len(st) - 1):
                e = st[ui + 2] if ui + 2 < len(st) else len(S)
                q = np.unique(self.map[S[st[ui]:st[ui + 1]]]); a = np.unique(self.map[S[st[ui + 1]:e]]); q = q[q >= 0][:maxtok]; a = a[a >= 0][:maxtok]
                if len(q) == 0 or len(a) == 0: continue
                rows.append(np.repeat(q, len(a))); cols.append(np.tile(a, len(q))); npair += 1
            if len(rows) > 3000: flush()
        flush(); C = C.tocoo(); N = C.data.sum(); rq = np.asarray(C.sum(1)).ravel() + 1; ra = np.asarray(C.sum(0)).ravel() + 1
        pmi = np.log(C.data * N / (rq[C.row] * ra[C.col])); m = (pmi > 0) & (C.data >= 3)
        M = sp.coo_matrix((pmi[m].astype(np.float32), (C.row[m], C.col[m])), shape=(n, n)).tocsr()
        U, s, Vt = svds(M, k=rank); o = np.argsort(-s); U, s, Vt = U[:, o], s[o], Vt[o]
        self.Eq = (U * np.sqrt(s)).astype(np.float32); self.Ea = (Vt.T * np.sqrt(s)).astype(np.float32)
        idf = np.log(len(convs) * 5.0 / (rq + 1)); self.idf = (idf / idf.max()).astype(np.float32); self.npair = npair; return self

    def feature(self, ctx, q0, q1):
        """質問 = ctx[q0:q1] (直前の 発話)。 返り値 [V] (語彙の外は 0)"""
        f = np.zeros(V, np.float32); ids = self.map[np.asarray(ctx[q0:q1])]; ids = ids[ids >= 0]
        if len(ids) == 0: return f
        qv = (self.Eq[ids] * self.idf[ids][:, None]).sum(0); nv = np.linalg.norm(qv)
        if nv == 0: return f
        f[self.vocab] = self.Ea @ (qv / nv); return f
