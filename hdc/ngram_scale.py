"""数え上げ (n-gram) だけの傾きを測る: 表を作る語数 S を変えて、 val の bit/語 を出す。
使い方: python3 ngram_scale.py <W: train.bin と val.bin の場所> <作業場所> <S(語数, 例 24000000)> [full=<3.8億語の最終表の場所>]
train.bin から 40 ブロック等間隔に S 語を集める (exp_ctx と同じ集め方)。"""
import sys, os, time, numpy as np
import engine as E
W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); t0 = time.time(); log = lambda s: print('[%4.0fs] %s' % (time.time() - t0, s), flush=True)
MINC = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}; os.makedirs(WORK, exist_ok=True)
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
sub = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); sub.tofile(WORK + '/sub.bin'); n_tok = len(sub) - 8; log('集めた語数 %d' % n_tok)
runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [1, 2, 3, 4, 5], 8_000_000, log)
for n in (1, 2, 3, 4, 5): E.merge(runs[n], WORK + '/final_o%d' % n, MINC[n], 20_000_000, log)
import glob
for f in glob.glob(WORK + '/runs/*'): os.remove(f)
M = E.Model({o: WORK + '/final_o%d' % o for o in (1, 2, 3, 4, 5)}, smooth='wb', lam={2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}, kn_base=True)
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(8, np.uint16), val])
rng = np.random.default_rng(777); idx = rng.integers(8, len(val) - 1, 20000) + 8
lp, hi = M.logprob_at(pad, idx, return_hi=True); b = -lp
print('RESULT S=%d n-gram bit/語 %.4f (最高次数 5: %.1f%%, 4: %.1f%%)' % (n_tok, b.mean(), 100 * (hi == 5).mean(), 100 * (hi == 4).mean()), flush=True)
