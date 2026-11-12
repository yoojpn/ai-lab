"""局所窓 (±2) の PPMI を 特異値分解した 単語埋め込み (形の 類推: 単数→複数, 小文字→大文字 を 見る)。 数え上げのみ。 出力: /tmp/bo24b/E_loc.npy [V, R]"""
import numpy as np, scipy.sparse as sp, sys, time
from scipy.sparse.linalg import svds
V = 32768; K = 24000; R = 128; WIN = 2
T = np.fromfile('/tmp/bo24b/sub.bin', dtype=np.uint16)[8:].astype(np.int64); n = len(T)
cnt = np.bincount(T, minlength=V).astype(np.float64); top = np.argsort(-cnt)[:K]; mp = -np.ones(V, np.int64); mp[top] = np.arange(K)
M = sp.csr_matrix((K, K), dtype=np.float32); t0 = time.time(); CH = 3_000_000
for s in range(0, n, CH):
    seg = T[s:s + CH + WIN]; m = mp[seg]; rows = []; cols = []
    for d in range(1, WIN + 1):
        a = m[:-d]; b = m[d:]; ok = (a >= 0) & (b >= 0); rows += [a[ok], b[ok]]; cols += [b[ok], a[ok]]
    r = np.concatenate(rows); c = np.concatenate(cols); M = M + sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(K, K)).tocsr()
    print(s, time.time() - t0, flush=True)
M = M.tocoo(); N = M.data.sum(); rs = np.asarray(M.sum(1)).ravel() + 1; cs = rs ** 0.75; cs = cs / cs.sum() * rs.sum()
pmi = np.log(M.data * rs.sum() / (rs[M.row] * cs[M.col])); keep = (pmi > 0) & (M.data >= 2)
P = sp.coo_matrix((pmi[keep].astype(np.float32), (M.row[keep], M.col[keep])), shape=(K, K)).tocsr()
U, s, Vt = svds(P, k=R); o = np.argsort(-s); E = np.zeros((V, R), np.float32); E[top] = (U[:, o] * np.sqrt(s[o])).astype(np.float32)
E[top] /= np.maximum(np.linalg.norm(E[top], axis=1, keepdims=True), 1e-6); np.save('/tmp/bo24b/E_loc.npy', E); print('EMB_DONE', time.time() - t0, flush=True)
