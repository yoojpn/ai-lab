"""再利用予測による 行の削減の実測: 単発の行を 「引かれやすさ」 の予測の上位 q だけ残し、 val の bit/token が どれだけ悪化するかと 行数の減りを測る。
使い方: python3 prune_eval.py WORK_DIR (index_analysis.py が作った表 o1..o8 と sub.bin を使う) VAL_BIN"""
import sys, os, numpy as np
import engine as E
D = sys.argv[1]; Tv = np.fromfile(sys.argv[2], dtype=np.uint16); T = np.fromfile(D + '/sub.bin', dtype=np.uint16); nt = len(T) - 8; idx = np.arange(8, len(T))
fq = np.bincount(T.astype(np.int64), minlength=32768) + 1; lf = np.log2(fq)
tab = {n: (np.fromfile('%s/o%d_k.bin' % (D, n), dtype=np.uint64), np.fromfile('%s/o%d_c.bin' % (D, n), dtype=np.uint32)) for n in (6, 8)}


def prune(n, keep):
    """次数 n の表から 単発の行 (文脈 1 回) を 引かれやすさの予測の上位 keep だけ残す。 書き出した接頭辞を返す"""
    k, c = tab[n]; h = E.ctx_hash(T, idx, n - 1, n); key = (h << np.uint64(16)) | T[idx].astype(np.uint64)
    u, inv, cnt = np.unique(h, return_inverse=True, return_counts=True); sing = cnt[inv] == 1; pos = idx[sing]
    ul, il, cl = np.unique(E.ctx_hash(T, idx, n - 2, n - 1), return_inverse=True, return_counts=True); lowcnt = np.log2(cl[il][sing])
    toks = np.stack([T[pos - j].astype(np.int64) for j in range(1, n)], 1); sc = lowcnt + 0.5 * lf[toks].mean(1) + 0.5 * lf[toks].min(1)
    thr = np.quantile(sc, 1 - keep) if keep < 1 else -1e9; drop = np.sort(key[sing][sc < thr]); j = np.minimum(np.searchsorted(drop, k), max(len(drop) - 1, 0)); isdrop = (drop[j] == k) if len(drop) else np.zeros(len(k), bool)
    pf = '%s/p%d_%d' % (D, n, int(keep * 1000))
    if not os.path.exists(pf + '_k.bin'): k[~isdrop].tofile(pf + '_k.bin'); c[~isdrop].tofile(pf + '_c.bin')
    return pf, int((~isdrop).sum())


rng = np.random.default_rng(777); p = rng.integers(16, len(Tv), 20400); p = p[Tv[p] != 0][:20000]; LAM = {2: 2.0, 3: 6.0, 4: 9.0, 5: 6.0, 6: 8.0, 8: 8.0}


def bits(extra):
    pre = {n: '%s/o%d' % (D, n) for n in (1, 2, 3, 4, 5)}; pre.update(extra); M = E.Model(pre, smooth='wb', lam={n: LAM[n] for n in pre if n >= 2}, kn_base=True)
    lp = np.zeros(len(p))
    for a in range(0, len(p), 5000): lp[a:a + 5000] = M.logprob_at(Tv, p[a:a + 5000])
    return -lp.mean()


rows = {n: len(tab[n][0]) for n in (6, 8)}; print('位置 %d 語; 全行: 次数6 %d, 次数8 %d' % (nt, rows[6], rows[8]), flush=True)
print('PRUNE 次数 1-5 だけ: %.4f bit' % bits({}), flush=True)
for keep in (1.0, 0.4, 0.2, 0.1, 0.05, 0.0):
    pf, nr = prune(6, keep); print('PRUNE +次数6 (単発を上位 %3.0f%% だけ残す): %.4f bit | 次数6の行 %d (全行の %.0f%%)' % (keep * 100, bits({6: pf}), nr, 100 * nr / rows[6]), flush=True)
for keep in (1.0, 0.2, 0.05):
    pf6, n6 = prune(6, keep); pf8, n8 = prune(8, keep); print('PRUNE +次数6,8 (単発を上位 %3.0f%% だけ残す): %.4f bit | 行 %d+%d' % (keep * 100, bits({6: pf6, 8: pf8}), n6, n8), flush=True)
