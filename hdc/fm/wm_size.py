import numpy as np, sys, math
from math import lgamma
b = np.fromfile(sys.argv[1], dtype=np.uint16).astype(np.int64); n = len(b); B = 31
lc = np.array([(lgamma(B+1)-lgamma(k+1)-lgamma(B-k+1))/math.log(2) for k in range(B+1)]); off = np.ceil(lc)
cur = b.copy(); tot = 0.0; per = []
for lv in range(14, -1, -1):
    bit = (cur >> lv) & 1; m = (-len(bit)) % B; bb = np.concatenate([bit, np.zeros(m, np.int64)]).reshape(-1, B).sum(1)
    s = (off[bb].sum() + 5 * len(bb)) / 8 + n / 512 * 4 / 1   # 区分けの合計ビット + 4バイト/512ビットの標本
    per.append(s); tot += s
    z = cur[bit == 0]; o = cur[bit == 1]; cur = np.concatenate([z, o])       # wavelet matrix: 0 を前に安定に並べ替え
print('n=%d  ウェーブレット行列 (RRR31 見積り): %.1f MB = %.3f B/語 (レベルごと %s)' % (n, tot / 1e6, tot / n, ' '.join('%.0f' % (x / 1e6) for x in per)))
print('参考: 平坦な 15bit = %.3f B/語' % (15 / 8))
