"""ほころびマッチの 被覆率の測定 (数え上げだけ)。 直前 L 語の文脈が 完全には見つからない位置で、 1 語だけ違う (置き換え) 文脈なら 見つかるか、
さらに その文脈の続きに 正解の語があるか を、 語数 S ごとに 数える。 対照: 正解の代わりに ランダムな語 (出現頻度に比例) でも 同じ確認をする。
使い方: python3 fuzzy_cov.py <W: train.bin,val.bin> <S> <L,L,...>"""
import sys, time, numpy as np
W, S = sys.argv[1], int(sys.argv[2]); LS = [int(x) for x in sys.argv[3].split(',')]; t0 = time.time()
log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
tok = np.concatenate([np.asarray(Ttr[s:s + bl]) for s in starts]); n = len(tok)
val = np.fromfile(W + '/val.bin', dtype=np.uint16); rng = np.random.default_rng(777); qi = rng.integers(64, len(val) - 1, 20000)
C1 = np.uint64(0x9E3779B97F4A7C15); SH = np.uint64(29)
def mix(h, t): h = (h ^ (t.astype(np.uint64) + np.uint64(1))) * C1; return h ^ (h >> SH)
def ctx_keys(a, idx, L, wild, base=np.uint64(12345)):
    h = np.full(len(idx), base, np.uint64)
    for k in range(L, 0, -1):
        t = a[idx - k]; h = mix(h, np.full(len(idx), 60000, np.uint16) if k == wild else t)
    return h
uni = np.bincount(tok, minlength=32768).astype(np.float64); uni /= uni.sum(); rnd = rng.choice(32768, size=len(qi), p=uni).astype(np.uint16)
for L in LS:
    ti = np.arange(L + 1, n - 1); q_true = val[qi]; res = {}
    ex_c = None
    for wild in [0] + list(range(1, L + 1)):
        hc = ctx_keys(tok, ti, L, wild); hq = ctx_keys(val, qi, L, wild)
        hp = mix(hc, tok[ti]); pq_t = mix(hq, q_true); pq_r = mix(hq, rnd)
        u = np.unique(hc); c_in = np.isin(hq, u, assume_unique=False)
        pp = np.sort(hp); ht = np.isin(pq_t, pp); hr = np.isin(pq_r, pp)
        if wild == 0: ex_c, ex_ht, ex_hr = c_in, ht & c_in, hr & c_in; fz_c = np.zeros(len(qi), bool); fz_ht = np.zeros(len(qi), bool); fz_hr = np.zeros(len(qi), bool)
        else: fz_c |= c_in; fz_ht |= ht & c_in; fz_hr |= hr & c_in
        del hc, hp, u, pp
    nov = ~ex_c
    print('RESULT S=%d L=%d | 完全一致の文脈あり %.1f%% (続きに正解 %.1f%% / 対照 %.1f%%) | 完全一致なしの位置 %.1f%% の中で: 1語違いの文脈あり %.1f%%, その続きに正解 %.1f%% / 対照(ランダム語) %.1f%%' % (
        n, L, 100 * ex_c.mean(), 100 * ex_ht.sum() / max(1, ex_c.sum()), 100 * ex_hr.sum() / max(1, ex_c.sum()), 100 * nov.mean(), 100 * (fz_c & nov).sum() / nov.sum(),
        100 * (fz_ht & nov).sum() / max(1, (fz_c & nov).sum()), 100 * (fz_hr & nov).sum() / max(1, (fz_c & nov).sum())), flush=True)
