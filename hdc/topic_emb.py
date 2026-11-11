"""話題用の語の埋め込み: 「同じ文書の近くに出る」関係 (窓 3〜40 語) の PMI を SVD。 隣り合う語の PMI (構文寄り) と違い、 話題・意味寄り。 数え上げ + 線形代数のみ。
使い方: python3 topic_emb.py <WORK (sub.bin がある所)> [ランク=64]  -> <WORK>/Etopic.npy"""
import sys, os, time, numpy as np, scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
WORK = sys.argv[1]; RK = int(sys.argv[2]) if len(sys.argv) > 2 else 64; V = 32768; t0 = time.time(); log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
tok = np.fromfile(WORK + '/sub.bin', dtype=np.uint16)[8:].astype(np.int64); n = len(tok); OFFS = [3, 5, 8, 12, 18, 26, 40]; acc = None; CH = 2_000_000
for a in range(0, n, CH):
    b = min(a + CH, n); keys = []
    for d in OFFS:
        e = min(b, n - d)
        if e > a: keys.append(tok[a:e] * V + tok[a + d:e + d])
    k = np.concatenate(keys); u, c = np.unique(k, return_counts=True); M = sp.coo_matrix((c.astype(np.float32), (u // V, u % V)), shape=(V, V)).tocsr(); acc = M if acc is None else acc + M
    log('チャンク %d/%d 非零 %d' % (a // CH + 1, (n + CH - 1) // CH, acc.nnz))
acc = (acc + acc.T).tocsr(); tot = acc.sum(); rm = np.asarray(acc.sum(1)).ravel() + 1e-9; cm = rm ** 0.75; cm = cm / cm.sum() * tot
acc = acc.tocoo(); pm = np.log(acc.data * tot / (rm[acc.row] * cm[acc.col])); keep = pm > 0
P = sp.coo_matrix((pm[keep].astype(np.float32), (acc.row[keep], acc.col[keep])), shape=(V, V)).tocsr(); log('PMI 非零 %d' % P.nnz)
U, s, Vt = randomized_svd(P, RK, n_iter=3, random_state=0); Em = U * np.sqrt(s)[None]; Em /= np.linalg.norm(Em, axis=1, keepdims=True) + 1e-9; np.save(WORK + '/Etopic.npy', Em.astype(np.float32)); log('完了')
