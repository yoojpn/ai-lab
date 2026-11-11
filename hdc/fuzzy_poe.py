"""ほころび (1語を穴にした文脈) で見つかった「続きの語」を、 n-gram の確率に 対数で足す。 数え上げ + 重み 1 本 (調整位置で数えた最良値)。
P'(w) = P(w) exp(l * f(w)) / Z,  f(w) = log(1 + 票)  (票 = 穴の位置 j ごとの 続きの出現回数の合計),  Z = 1 + sum_{w in 候補} P(w)(exp(l f(w)) - 1)。
l は n-gram の最高次数ごと (1〜5) に 調整位置で選ぶ。 候補の P(w) は 同じ n-gram から引く (合成した窓で)。
使い方: python3 fuzzy_poe.py <W> <WORK> <S> <L>"""
import sys, os, time, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S, L = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(WORK, exist_ok=True); t0 = time.time(); log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
sub = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); sub.tofile(WORK + '/sub.bin'); n_tok = len(sub) - 8; tok = sub
for n in (1, 2, 3, 4, 5):
    if os.path.exists(WORK + '/final_o%d_k.bin' % n): continue
    runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], WORK + '/final_o%d' % n, 1 if n <= 2 else 2, 20_000_000, lambda s: None)
    for f in glob.glob(WORK + '/runs/*'): os.remove(f)
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
idx_te, idx_tu = pos(777, 20000), pos(123, 6000)
M = E.Model({n: WORK + '/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}, kn_base=False)
C1 = np.uint64(0x9E3779B97F4A7C15); SH = np.uint64(29); M48 = np.uint64((1 << 48) - 1)
def mix(h, t): h = (h ^ (t.astype(np.uint64) + np.uint64(1))) * C1; return h ^ (h >> SH)
def ctx_keys(a, idx, wild):
    h = np.full(len(idx), np.uint64(12345), np.uint64)
    for k in range(L, 0, -1): h = mix(h, np.full(len(idx), 60000, np.uint16) if k == wild else a[idx - k])
    return h & M48
ti = np.arange(L + 1, len(tok) - 1)
CAND = {'te': [dict() for _ in idx_te], 'tu': [dict() for _ in idx_tu]}
for wild in range(1, L + 1):
    keys = (ctx_keys(tok, ti, wild) << np.uint64(16)) | tok[ti].astype(np.uint64); keys.sort()
    for nm, idx in (('te', idx_te), ('tu', idx_tu)):
        h = ctx_keys(pad, idx, wild); lo = np.searchsorted(keys, h << np.uint64(16)); hi = np.searchsorted(keys, (h + np.uint64(1)) << np.uint64(16))
        for p in np.flatnonzero(hi > lo):
            ws, cs = np.unique((keys[lo[p]:hi[p]] & np.uint64(0xFFFF)).astype(np.int64), return_counts=True); d = CAND[nm][p]
            for w, c in zip(ws, cs): d[int(w)] = d.get(int(w), 0) + int(c)
    del keys; log('穴の位置 %d 完了' % wild)
TOPK = 20
def prep(nm, idx):
    lp, hi = M.logprob_at(pad, idx, return_hi=True); P = np.exp2(lp); wins = []; owner = []; ws = []; votes = []
    for p, d in enumerate(CAND[nm]):
        if not d: continue
        for w, c in sorted(d.items(), key=lambda z: -z[1])[:TOPK]: owner.append(p); ws.append(w); votes.append(c)
    owner = np.array(owner, int); ws = np.array(ws, np.uint16); votes = np.array(votes, float)
    # 候補の P(w): 合成した窓 (64 語の文脈 + 候補) を 並べた 配列で 同じ n-gram に 引く
    ctx = np.stack([pad[idx[o] - 64:idx[o]] for o in owner]) if len(owner) else np.zeros((0, 64), np.uint16)
    syn = np.concatenate([ctx, ws[:, None]], 1).ravel(); syn = np.concatenate([np.zeros(8, np.uint16), syn]); si = 8 + np.arange(len(owner)) * 65 + 64
    Pw = np.exp2(M.logprob_at(syn, si)) if len(owner) else np.zeros(0)
    return P, hi, owner, ws, votes, Pw
def apply(P, owner, ws, votes, Pw, true, lam):
    """lam: 位置ごとの l (配列)。 戻り値: 新しい P(正解)"""
    e = np.exp(lam[owner] * np.log1p(votes)); Z = np.ones(len(P)); np.add.at(Z, owner, Pw * (e - 1)); out = P.copy()
    bonus = np.ones(len(P)); m = ws == true[owner]; bonus[owner[m]] = e[m]
    return P * bonus / Z
R = {}
for nm, idx in (('te', idx_te), ('tu', idx_tu)):
    R[nm] = prep(nm, idx); log('%s: 候補のある位置 %.1f%%, 平均候補数 %.1f' % (nm, 100 * len(np.unique(R[nm][2])) / len(idx), len(R[nm][2]) / max(1, len(np.unique(R[nm][2])))))
truth = {'te': pad[idx_te], 'tu': pad[idx_tu]}; grid = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]; lam_b = {}
P, hi, owner, ws, votes, Pw = R['tu']
for b in range(1, 6):
    m = hi == b; best = None
    for l in grid:
        lam = np.where(m, l, 0.0); bits = -np.log2(apply(P, owner, ws, votes, Pw, truth['tu'], lam))[m].mean() if m.any() else 0
        if best is None or bits < best[0]: best = (bits, l)
    lam_b[b] = best[1]
P, hi, owner, ws, votes, Pw = R['te']; lam = np.array([lam_b.get(int(h), 0.0) for h in hi]); P2 = apply(P, owner, ws, votes, Pw, truth['te'], lam)
print('RESULT S=%d L=%d  n-gram %.4f -> ほころびを 対数で足す %.4f bit/語 (得 %.4f)  l=%s' % (n_tok, L, -np.log2(P).mean(), -np.log2(P2).mean(), -np.log2(P).mean() + np.log2(P2).mean(), lam_b), flush=True)
