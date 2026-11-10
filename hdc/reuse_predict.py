"""行の再利用の予測: 学習 A の単発の行 (文脈が 1 回しか出ない) のうち、 別の文書 B の位置から 実際に引かれる (文脈が一致する) のはどれか。
行の文脈の特徴 (文脈の語の頻度・下の次数の文脈の回数) で 引かれやすさを 予測できれば、 引かれにくい行を 先に捨てて 容量を減らせる。
捕捉曲線: 予測の上位 p% の行を残すと、 引かれる行のうち 何割を残せるか。 使い方: python3 reuse_predict.py TOK_DIR N_TOKENS"""
import sys, numpy as np
import engine as E
TOK = sys.argv[1]; NTOK = int(sys.argv[2]); Ttr = np.memmap(TOK + '/train.bin', dtype=np.uint16, mode='r'); NT = len(Ttr); nb = 40; bl = NTOK // nb; st = np.linspace(8, NT - bl - 8, nb).astype(np.int64)
blocks = [np.asarray(Ttr[s:s + bl]) for s in st]; mk = lambda bs: np.concatenate([np.zeros(8, np.uint16)] + bs)
A = mk(blocks[0::2]); B = mk(blocks[1::2]); iA = np.arange(8, len(A)); iB = np.arange(8, len(B)); print('A %d 語, B %d 語' % (len(A), len(B)), flush=True)
fq = np.bincount(A.astype(np.int64), minlength=32768) + 1; lf = np.log2(fq)
def ctxh(T, idx, n): return E.ctx_hash(T, idx, n - 1, n)
for n in (6, 8):
    hA = ctxh(A, iA, n); u, inv, cnt = np.unique(hA, return_inverse=True, return_counts=True); single = cnt[inv] == 1
    hB = np.unique(ctxh(B, iB, n)); hit = np.isin(hA[single], hB); idx = iA[single]
    low = ctxh(A, iA, n - 1); ul, il, cl = np.unique(low, return_inverse=True, return_counts=True); lowcnt = np.log2(cl[il][single])          # 1 つ下の次数の文脈の回数 (対数)
    toks = np.stack([A[idx - k].astype(np.int64) for k in range(1, n)], 1); f_mean = lf[toks].mean(1); f_min = lf[toks].min(1); f_next = lf[A[idx].astype(np.int64)]
    print('次数 %d: 単発の行 %d (全行の %.2f), そのうち B から 文脈が引かれる行 %.4f' % (n, single.sum(), single.mean(), hit.mean()), flush=True)
    base = hit.mean()
    for name, sc in (('下の次数の文脈の回数', lowcnt), ('文脈の語の平均頻度', f_mean), ('文脈の語の最小頻度', f_min), ('3 つの合成', lowcnt + 0.5 * f_mean + 0.5 * f_min), ('ランダム', np.random.default_rng(0).random(len(idx)))):
        o = np.argsort(-sc); h = hit[o].cumsum(); tot = h[-1]; row = []
        for p in (0.05, 0.1, 0.2, 0.4, 0.6):
            k = int(p * len(o)); row.append('上位 %2d%% で 引かれる行の %.0f%% を残す' % (p * 100, 100 * h[k - 1] / max(tot, 1)))
        print('   %-18s: %s' % (name, ' | '.join(row)), flush=True)
