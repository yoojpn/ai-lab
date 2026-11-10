"""数え上げ attention: 過去 512 語のなかから 「次の語を写せる位置」を 数え表の得点で選ぶ。 勾配なし・網なし。 CPU。
候補位置 j: 直前の語が 今の直前の語と同じ位置 (T[j-1] == T[t-1])。 候補は T[j] を予測する。
候補の特徴の鍵 = (後ろ向きの一致の長さ L (1..8), 距離の区分, 予測する語の頻度の区分) -> 「T[j] が 本当の次の語だった」 割合 (数え表)。
予測: P(w) = (1-g) * P_ngram(w) + g * q(w),  q(w) = (w を予測する候補の得点の和) / (全候補の得点の和),  g は 得点の合計の区分ごとに val の前半で選び 後半で測る。
比較: 直近 256 語の頻度キャッシュ (今までの方式)。
使い方: python3 count_attn.py N_TRAIN_POS N_VAL"""
import sys, time
import numpy as np
sys.path.insert(0, '/home/claude/ai-lab/hdc')
import engine as E

S = '/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad'
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 150000; NV = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
t0 = time.time(); lg = lambda s: print('[%4ds] %s' % (time.time() - t0, s), flush=True)
Ttr = np.memmap(S + '/dl/tok/train.bin', dtype=np.uint16, mode='r'); Tv = np.fromfile('/tmp/tokhf/tok/val.bin', dtype=np.uint16); W = 512
fq = np.bincount(np.asarray(Ttr[:60_000_000]).astype(np.int64), minlength=32768) + 1; fbk = np.minimum(np.log2(fq).astype(int) // 2, 11)
DB = [1, 2, 3, 4, 6, 9, 17, 33, 65, 129, 257]


def cands(x):
    """x = T[t-W-1 : t] (最後が 直前の語)。 候補 = 同じ文書の過去の全位置 j (T[j] を予測する)。 戻り値: (候補が予測する語, 鍵)。 一致の長さ L = 直前の語から 後ろ向きに 何語 一致するか (0..8)"""
    n = len(x); a = x[-1]; z = np.flatnonzero(x[:-1] == 0); lo = (z[-1] + 1) if len(z) else 0
    ks = np.arange(lo, n - 1)
    if len(ks) == 0: return None
    w = x[ks + 1]; ok = w != 0; ks = ks[ok]; w = w[ok]
    if len(ks) == 0: return None
    L = np.zeros(len(ks), np.int64); alive = np.ones(len(ks), bool)
    for i in range(0, 8):
        idx = ks - i; okk = alive & (idx >= lo); same = np.zeros(len(ks), bool); same[okk] = x[idx[okk]] == x[n - 1 - i]; alive = same; L += alive
    d = np.digitize(n - 1 - ks, DB); u, inv = np.unique(w, return_inverse=True); m = len(u)
    nocc = np.bincount(inv, minlength=m); dmin = np.full(m, 99); np.minimum.at(dmin, inv, d); Lmax = np.zeros(m, np.int64); np.maximum.at(Lmax, inv, L)
    key = (Lmax * 6 + np.digitize(nocc, [2, 3, 4, 6, 10])) * 144 + dmin * 12 + fbk[u]
    return u, key


def collect(T, pos, with_label):
    out = []
    for t in pos:
        x = np.asarray(T[t - W - 1:t]).astype(np.int64); r = cands(x); out.append((r, int(T[t]) if with_label else -1))
    return out


rng = np.random.default_rng(3); ptr = np.sort(rng.integers(W + 2, len(Ttr) - 1, NTR)); ptr = ptr[np.asarray(Ttr[ptr]) != 0]
NK = 9 * 6 * 144; npos = np.zeros(NK); ntot = np.zeros(NK); tr = collect(Ttr, ptr, True)
for r, y in tr:
    if r is None: continue
    w, key = r; np.add.at(ntot, key, 1); np.add.at(npos, key, (w == y))
p0 = npos.sum() / ntot.sum(); ptab = (npos + 20 * p0) / (ntot + 20); lg('学習位置 %d: 候補の平均 %.1f 個, 全体の的中率 %.4f' % (len(tr), ntot.sum() / len(tr), p0))
rv = np.random.default_rng(777); pv = rv.integers(W + 2, len(Tv), int(NV * 1.02)); pv = pv[Tv[pv] != 0][:NV]
lam = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model({n: '%s/ng/tok/final_o%d' % (S, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
lp = np.zeros(len(pv))
for a in range(0, len(pv), 5000): lp[a:a + 5000] = M.logprob_at(Tv, pv[a:a + 5000])
Png = 2.0 ** lp; lg('n-gram %.4f bit' % -np.log2(Png).mean())
Q = np.zeros(len(pv)); Ssum = np.zeros(len(pv)); Qc = np.zeros(len(pv)); Cmax = np.zeros(len(pv))
for i, t in enumerate(pv):
    x = np.asarray(Tv[t - W - 1:t]).astype(np.int64); y = int(Tv[t]); r = cands(x)
    if r is not None:
        w, key = r; p = ptab[key]; Ssum[i] = p.sum(); Q[i] = p[w == y].sum() / max(p.sum(), 1e-12)
    win = x[-256:]; win = win[win > 2]; Qc[i] = (win == y).sum() / len(win) if len(win) > 8 else 0; Cmax[i] = np.bincount(win).max() / len(win) if len(win) > 8 else 0
lg('候補を採点した')
G = np.r_[0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]; bits = lambda P: float(-np.log2(np.maximum(P, 1e-12)).mean()); h = len(pv) // 2
sb = np.digitize(Ssum, [1e-9, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]); cb = np.digitize(np.array([0.0] * len(pv)), [1])


def fit_eval(Qx, bins, nb):
    out = np.zeros(len(pv))
    for A, Bx in ((slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))):
        g = np.zeros(nb)
        for b in range(nb):
            m = np.flatnonzero(bins[A] == b)
            if len(m) < 30: continue
            g[b] = G[int(np.argmin([bits((1 - x) * Png[A][m] + x * Qx[A][m]) for x in G]))]
        out[Bx] = (1 - g[bins[Bx]]) * Png[Bx] + g[bins[Bx]] * Qx[Bx]
    return out


hi = np.zeros(len(pv), int)
lg('ATTN_RESULT n-gram 単独 %.4f' % bits(Png))
cbin = np.digitize(Cmax, [1e-9, 0.02, 0.04, 0.06, 0.1, 0.2])      # 重みの区分は 正解の語を使わず (正解の語の確率で区切ると 答えを見て重みを決める漏れになる), 窓の中の最頻語の割合で決める
lg('ATTN_RESULT n-gram + 直近256語の頻度キャッシュ %.4f' % bits(fit_eval(Qc, cbin, 7)))
lg('ATTN_RESULT n-gram + 数え上げ attention %.4f' % bits(fit_eval(Q, sb, 10)))
comb = np.maximum(Q, 0); both = fit_eval(0.5 * Q + 0.5 * Qc, sb, 10); lg('ATTN_RESULT n-gram + 数え上げ attention + キャッシュ(半々) %.4f' % bits(both))
