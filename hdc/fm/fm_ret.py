"""検索型の応答 (数え上げだけ: 文字 1-3 グラムの TF-IDF)。 対話データから (発話, 返事) の対を作り、 いちばん近い発話の返事を返す。
作り方: Retriever.build(pairs) (pairs = [(出所, 発話, 返事), ...])  / 検索: r.search('こんにちは', k=5) -> [(類似度, (出所, 発話, 返事)), ...]
対の集め方は ret_build.py (Colab 上)。 6,052,602 対 (izumi 5.66M, RPC 0.37M, oasst1-ja 22k) で 文字 n-gram の 非ゼロ 8.2 億"""
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
HV = HashingVectorizer(analyzer='char', ngram_range=(1, 3), n_features=2 ** 20, alternate_sign=False, norm=None, dtype=np.float32)


class Retriever:
    def __init__(self, PR, XT, idf, df):
        self.PR = PR; self.XT = XT; self.idf = idf; self.df = df; self.N = len(PR)

    @classmethod
    def build(cls, PR, chunk=200000):
        N = len(PR); parts = []
        for i in range(0, N, chunk):
            M = HV.transform([p[1] for p in PR[i:i + chunk]]); M.data = np.log1p(M.data); parts.append(M)
        X = sp.vstack(parts, format='csr'); del parts
        df = np.bincount(X.indices, minlength=2 ** 20).astype(np.float64); idf = (np.log((N + 1) / (df + 1)) + 1).astype(np.float32)
        X.data *= idf[X.indices]
        nrm = np.sqrt(np.add.reduceat(X.data ** 2, X.indptr[:-1])); nrm[nrm == 0] = 1; X.data /= np.repeat(nrm, np.diff(X.indptr)).astype(np.float32)
        XT = X.T.tocsr(); del X
        return cls(PR, XT, idf, df)

    def search(self, q, k=5, maxdf=0.15):
        v = HV.transform([q]); v.data = np.log1p(v.data); cols = v.indices; w = v.data * self.idf[cols]
        keep = self.df[cols] <= maxdf * self.N; cols = cols[keep]; w = w[keep]
        if len(cols) == 0: return []
        w = w / np.sqrt((w ** 2).sum()); sc = np.zeros(self.N, np.float32); XT = self.XT
        for c, x in zip(cols, w):
            s, e = XT.indptr[c], XT.indptr[c + 1]; sc[XT.indices[s:e]] += x * XT.data[s:e]
        top = np.argpartition(-sc, k)[:k]; top = top[np.argsort(-sc[top])]
        return [(float(sc[i]), self.PR[i]) for i in top]
