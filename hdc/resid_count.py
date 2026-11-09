"""案A: 残差の数え上げ。 数え上げ (n-gram) が 外すときの外れ方を 別の表で数えて 直せるか。 CPU。
学習語の一部 (40 ブロック x 75 万語) を 取り分けて 数え上げの表から除き、 その位置で
   鍵 = (見つかった最高次数 hi, n-gram の第 1 候補 w*, 直前の語 ...) -> 実際の次の語 の回数 を数える。
予測 p(y) = (表の回数(y) + lam * p_ngram(y)) / (表の合計 + lam)  (鍵を粗い順に重ねる)。 評価は val。 比較の基準は 同じ表での n-gram だけ。
使い方: python3 resid_count.py TOK_DIR WORK_DIR [N_HELD] [N_VAL]"""
import os, sys, time, shutil
import numpy as np
import engine as E

TOK = sys.argv[1]; D = sys.argv[2]; NH = int(sys.argv[3]) if len(sys.argv) > 3 else 1_500_000; NV = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); os.makedirs(D, exist_ok=True)
NT = os.path.getsize(TOK + '/train.bin') // 2; Ttr = np.memmap(TOK + '/train.bin', dtype=np.uint16, mode='r')
nb = 40; hb = max(NT // 600, 1000); hb = min(hb, 750_000); starts = np.linspace(2000, NT - hb - 8, nb).astype(np.int64)
segs = []; prev = 0
for s in starts: segs.append((prev, int(s))); prev = int(s) + hb
segs.append((prev, NT)); pad = D + '/cnt_pad.bin'
if not os.path.exists(pad + '.ok'):
    with open(pad, 'wb') as f:
        np.zeros(8, np.uint16).tofile(f)
        for a, b in segs: np.asarray(Ttr[a:b]).tofile(f)
    open(pad + '.ok', 'w').write('1')
NC = os.path.getsize(pad) // 2 - 8; lg('数える語 %d, 取り分け %d 語' % (NC, nb * hb))
MINC = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}; pre = {n: '%s/f_o%d' % (D, n) for n in (1, 2, 3, 4, 5)}
for n in (1, 2, 3, 4, 5):
    if os.path.exists(pre[n] + '_k.bin'): continue
    runs = E.spill(pad, NC, D + '/runs', [n], 8_000_000, lg); E.merge(runs[n], pre[n], MINC[n], 20_000_000, lg); shutil.rmtree(D + '/runs', ignore_errors=True)
lam0 = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}; M = E.Model(pre, smooth='wb', lam=lam0, kn_base=True); top_uni = int(np.argmax(M.uni)); lg('n-gram 準備できた')
Tall = np.fromfile(TOK + '/train.bin', dtype=np.uint16); Tv = np.fromfile(TOK + '/val.bin', dtype=np.uint16)


def topcand(T, p, hi):
    """n-gram が 見つけた最高次数の文脈で 一番多かった次の語"""
    w = np.full(len(p), top_uni, np.int64)
    for n in (5, 4, 3, 2):
        m = np.flatnonzero(hi == n)
        if not len(m): continue
        h = E.ctx_hash(T, p[m], n - 1, n); k = M.t[n]['k']; lo = np.searchsorted(k, h << np.uint64(16)); he = np.searchsorted(k, (h + np.uint64(1)) << np.uint64(16)); ln = he - lo
        ok = ln > 0; m = m[ok]; lo = lo[ok]; ln = ln[ok]
        if not len(m): continue
        off = np.cumsum(ln) - ln; tl = int(ln.sum()); idx = np.arange(tl) - np.repeat(off, ln) + np.repeat(lo, ln); seg = np.repeat(np.arange(len(m)), ln)
        c = np.asarray(M.t[n]['c'][idx]).astype(np.int64); kk = np.asarray(k[idx]); mx = np.maximum.reduceat(c, off); eq = np.flatnonzero(c == mx[seg]); u, first = np.unique(seg[eq], return_index=True)
        w[m[u]] = (kk[eq[first]] & np.uint64(0xFFFF)).astype(np.int64)
    return w


def feats(T, p, CH=100000):
    lp = np.zeros(len(p)); hi = np.zeros(len(p), np.int64); ws = np.zeros(len(p), np.int64)
    for a in range(0, len(p), CH):
        q = p[a:a + CH]; l_, h_ = M.logprob_at(T, q, return_hi=True); lp[a:a + CH] = l_; hi[a:a + CH] = h_; ws[a:a + CH] = topcand(T, q, h_)
    return 2.0 ** lp, hi, ws


def keyof(cols):
    h = np.zeros(len(cols[0]), dtype=np.uint64)
    for c_ in cols: h = (h * np.uint64(0x9E3779B97F4A7C15) + c_.astype(np.uint64) + np.uint64(1)) * np.uint64(0xBF58476D1CE4E5B9)
    return h


class Counts:
    def __init__(s, addr, y):
        s.ctx, ci = np.unique(addr, return_inverse=True); s.tot = np.bincount(ci).astype(np.float64); pk = keyof([addr, y]); s.pk, s.cnt = np.unique(pk, return_counts=True)

    def lookup(s, addr, y):
        i = np.minimum(np.searchsorted(s.ctx, addr), len(s.ctx) - 1); seen = s.ctx[i] == addr; tot = np.where(seen, s.tot[i], 0.0)
        k = keyof([addr, y]); j = np.minimum(np.searchsorted(s.pk, k), len(s.pk) - 1); cy = np.where(s.pk[j] == k, s.cnt[j], 0).astype(np.float64); return tot, cy, seen


# 取り分けの位置 (n-gram の表に入っていない所)
allp = (starts[:, None] + np.arange(hb)[None]).reshape(-1); rng = np.random.default_rng(1); sel = np.sort(rng.choice(len(allp), min(NH, len(allp)), replace=False)); ph = allp[sel]; ph = ph[Tall[ph] != 0]
lg('取り分けの位置 %d' % len(ph)); Ph, Hh, Wh = feats(Tall, ph); lg('取り分け側の特徴 済み: n-gram %.4f bit' % -np.log2(Ph).mean())
rv = np.random.default_rng(777); pv = rv.integers(16, len(Tv), int(NV * 1.02)); pv = pv[Tv[pv] != 0][:NV]; Pv, Hv, Wv = feats(Tv, pv); yh = Tall[ph].astype(np.int64); yv = Tv[pv].astype(np.int64); h2 = len(pv) // 2
lg('val 側の特徴 済み')
bits = lambda P: float(-np.log2(np.maximum(P, 1e-12)).mean())
t1h = Tall[ph - 1].astype(np.int64); t2h = Tall[ph - 2].astype(np.int64); t1v = Tv[pv - 1].astype(np.int64); t2v = Tv[pv - 2].astype(np.int64)
LV = [('hi だけ', [Hh], [Hv]), ('hi+第1候補', [Hh, Wh], [Hv, Wv]), ('hi+第1候補+前1語', [Hh, Wh, t1h], [Hv, Wv, t1v]), ('hi+第1候補+前2語', [Hh, Wh, t1h, t2h], [Hv, Wv, t1v, t2v]), ('第1候補+前1語 (hi なし)', [Wh, t1h], [Wv, t1v])]
lookups = []
for name, ch, cv in LV:
    C = Counts(keyof([x.astype(np.int64) for x in ch]), yh); lookups.append(C.lookup(keyof([x.astype(np.int64) for x in cv]), yv))
GR = (0.5, 2, 8, 32, 128, 512, 2048)


def chain(P0, ks, lams):
    p = P0.copy()
    for k, lm in zip(ks, lams):
        tot, cy, seen = lookups[k]; p = np.where(seen, (cy + lm * p) / (tot + lm), p)
    return p


lg('RESID_RESULT 基準 n-gram だけ (val 後半 %d 位置): %.4f bit' % (len(pv) - h2, bits(Pv[h2:])))
for name_, ks in (('hi だけ(較正)', [0]), ('hi + 第1候補', [0, 1]), ('hi + 第1候補 + 前1語', [0, 1, 2]), ('hi + 第1候補 + 前2語', [0, 1, 3]), ('第1候補+前1語 (hi なし)', [4])):
    lams = [8.0] * len(ks)
    for _ in range(2):
        for i in range(len(ks)):
            best = None
            for v in GR:
                l2 = list(lams); l2[i] = v; sub = [lookups[k] for k in ks]
                pp = Pv[:h2].copy()
                for (tot, cy, seen), lm in zip(sub, l2): pp = np.where(seen[:h2], (cy[:h2] + lm * pp) / (tot[:h2] + lm), pp)
                b = bits(pp)
                if best is None or b < best[0]: best = (b, v)
            lams[i] = best[1]
    pp = chain(Pv, ks, lams); lg('RESID_RESULT %s: val 前半(調整) %.4f -> 後半(評価) %.4f bit  lam=%s' % (name_, bits(pp[:h2]), bits(pp[h2:]), lams))
lg('終わり')
