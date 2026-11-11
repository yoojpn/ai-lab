"""対数で足す特徴の枠組み (勾配なし)。 n-gram (次数 1〜5) の確率 P(w) に、 いろいろな「続きの語の手がかり」を 対数で足す:
  P'(w) = P(w) exp(sum_f l_f * log(1 + 票_f(w))) / Z。   票_f(w) = 手がかり f で見つかった 続きの語 w の 出現回数
手がかり (特徴) の種類 = 文脈の 一致のさせ方:
  sub L : 直前 L 語のうち 1 語を 穴にした一致 (置き換え)         del L : 文脈に 余計な 1 語がある (欠け)         ins L : 学習側に 余計な 1 語がある (挿入)
  ex L  : 直前 L 語の 完全一致 (L = 6, 8, 12, 16。 n-gram 本体は 5 まで)
重み l_f は n-gram の最高次数 (1〜5) ごとに 調整位置で 順に 座標を動かして決める (数えた損失だけ)。 貪欲に 1 つずつ 足した順番と 得も 出す。
使い方: python3 boost.py <W: train.bin,val.bin> <WORK> <S>"""
import sys, os, time, glob, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(WORK, exist_ok=True); t0 = time.time(); log = lambda s: print('[%5.0fs] %s' % (time.time() - t0, s), flush=True)
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
if not os.path.exists(WORK + '/sub.bin'):
    sub = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); sub.tofile(WORK + '/sub.bin')
tok = np.fromfile(WORK + '/sub.bin', dtype=np.uint16); n_tok = len(tok) - 8
for n in (1, 2, 3, 4, 5):
    if os.path.exists(WORK + '/final_o%d_k.bin' % n): continue
    runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], WORK + '/final_o%d' % n, 1 if n <= 2 else 2, 20_000_000, lambda s: None)
    for f in glob.glob(WORK + '/runs/*'): os.remove(f)
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); pos = lambda seed, n: np.random.default_rng(seed).integers(64, len(val) - 1, n) + 64
idx_te, idx_tu = pos(777, 20000), pos(123, 20000)
M = E.Model({n: WORK + '/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}, kn_base=False)
C1 = np.uint64(0x9E3779B97F4A7C15); SH = np.uint64(29); M48 = np.uint64((1 << 48) - 1)
def mix(h, t): h = (h ^ (t.astype(np.uint64) + np.uint64(1))) * C1; return h ^ (h >> SH)
def keys_train(D, lo, hi):
    """学習側: 位置 [lo,hi) を 次の語とする 文脈 (距離の集合 D) の ハッシュ。 スライスで読む"""
    h = np.full(hi - lo, np.uint64(12345), np.uint64)
    for k in sorted(D, reverse=True): h = mix(h, tok[lo - k:hi - k])
    return h & M48
def keys_test(D, idx):
    h = np.full(len(idx), np.uint64(12345), np.uint64)
    for k in sorted(D, reverse=True): h = mix(h, pad[idx - k])
    return h & M48
FAM = {}                                                        # 名前 -> [(D_学習, D_テスト), ...]
for L in (6, 8):
    FAM['sub%d' % L] = [(set(range(1, L + 1)) - {j}, set(range(1, L + 1)) - {j}) for j in range(1, L + 1)]
FAM['del8'] = [(set(range(1, 9)), set(range(1, 10)) - {j}) for j in range(1, 10)]
FAM['ins8'] = [(set(range(1, 10)) - {j}, set(range(1, 9))) for j in range(1, 10)]
for L in (6, 8, 12, 16): FAM['ex%d' % L] = [(set(range(1, L + 1)), set(range(1, L + 1)))]
SPECIAL = {'lshA': ([1.0, 0.7, 0.5], 8, 14), 'lshB': ([1.0, 0.8, 0.6, 0.5, 0.4, 0.3], 8, 14)}   # 重み (近い語ほど大), 表の数, ビット数
NAMES = list(FAM) + list(SPECIAL); TOPK = 20
def get_emb(dim=48):
    """隣り合う語の PMI の SVD で 語の埋め込み (数え上げ + 線形代数のみ)"""
    ep = WORK + '/E.npy'
    if os.path.exists(ep): return np.load(ep)
    import scipy.sparse as sp, scipy.sparse.linalg as sl
    x = tok[8:].astype(np.int64); V = E.VOC; key = x[:-1] * V + x[1:]; u, cnt = np.unique(key, return_counts=True); del key
    Mx = sp.coo_matrix((cnt.astype(np.float64), (u // V, u % V)), shape=(V, V)).tocsr(); tot = Mx.sum(); rm = np.asarray(Mx.sum(1)).ravel() + 1e-9; cm = np.asarray(Mx.sum(0)).ravel() ** 0.75 + 1e-9; cm = cm / cm.sum() * tot
    Mx = Mx.tocoo(); pm = np.log(Mx.data * tot / (rm[Mx.row] * cm[Mx.col])); keep = pm > 0
    Pm = sp.coo_matrix((pm[keep], (Mx.row[keep], Mx.col[keep])), shape=(V, V)).tocsr(); U, sg, _ = sl.svds(Pm, k=dim); Em = U * np.sqrt(sg)[None]; Em /= np.linalg.norm(Em, axis=1, keepdims=True) + 1e-9
    Em = Em.astype(np.float32); np.save(ep, Em); return Em
def lsh_keys_train(Em, wts, Rm, lo, hi):
    v = np.zeros((hi - lo, Em.shape[1]), np.float32)
    for k, wk in enumerate(wts, 1): v += wk * Em[tok[lo - k:hi - k]]
    b = (v @ Rm > 0); h = np.zeros(hi - lo, np.uint64)
    for j in range(b.shape[1]): h |= b[:, j].astype(np.uint64) << np.uint64(j)
    return h
def lsh_keys_test(Em, wts, Rm, idx):
    v = np.zeros((len(idx), Em.shape[1]), np.float32)
    for k, wk in enumerate(wts, 1): v += wk * Em[pad[idx - k]]
    b = (v @ Rm > 0); h = np.zeros(len(idx), np.uint64)
    for j in range(b.shape[1]): h |= b[:, j].astype(np.uint64) << np.uint64(j)
    return h
def chunked(func, lo, hi, CH=4_000_000):
    """func(a, b) -> 位置 [a,b) の uint64 キー。 全体を 1 本の配列に (一時の配列は ブロックの分だけ)"""
    out = np.empty(hi - lo, np.uint64)
    for a in range(lo, hi, CH): b = min(a + CH, hi); out[a - lo:b - lo] = func(a, b)
    return out
def build_family(nm):
    out = {'te': [dict() for _ in idx_te], 'tu': [dict() for _ in idx_tu]}; hi = len(tok) - 1
    def add(keys, tests):
        keys.sort()
        for part, idx, h in tests:
            a_ = np.searchsorted(keys, h << np.uint64(16)); b_ = np.searchsorted(keys, (h + np.uint64(1)) << np.uint64(16))
            for p in np.flatnonzero(b_ > a_):
                ws, cs = np.unique((keys[a_[p]:b_[p]] & np.uint64(0xFFFF)).astype(np.int64), return_counts=True); d = out[part][p]
                for w, c in zip(ws, cs): d[int(w)] = d.get(int(w), 0) + int(c)
    if nm in SPECIAL:
        wts, T, nbits = SPECIAL[nm]; Em = get_emb(); st = len(wts) + 8
        for t in range(T):
            Rm = np.random.default_rng(1000 + t).standard_normal((Em.shape[1], nbits)).astype(np.float32)
            keys = chunked(lambda a, b: (lsh_keys_train(Em, wts, Rm, a, b) << np.uint64(16)) | tok[a:b].astype(np.uint64), st, hi)
            add(keys, [('te', idx_te, lsh_keys_test(Em, wts, Rm, idx_te)), ('tu', idx_tu, lsh_keys_test(Em, wts, Rm, idx_tu))]); del keys
    else:
        for Dtr, Dte in FAM[nm]:
            st = max(max(Dtr), 1) + 8; keys = chunked(lambda a, b: (keys_train(Dtr, a, b) << np.uint64(16)) | tok[a:b].astype(np.uint64), st, hi)
            add(keys, [('te', idx_te, keys_test(Dte, idx_te)), ('tu', idx_tu, keys_test(Dte, idx_tu))]); del keys
    return out
CAND = {}; old_cp = WORK + '/cands.pkl'; OLD = pickle.load(open(old_cp, 'rb')) if os.path.exists(old_cp) else {}
for nm in NAMES:
    fp = WORK + '/cand_%s.pkl' % nm
    if nm in OLD: CAND[nm] = OLD[nm]
    elif os.path.exists(fp): CAND[nm] = pickle.load(open(fp, 'rb'))
    else: CAND[nm] = build_family(nm); pickle.dump(CAND[nm], open(fp, 'wb'))
    log('特徴 %s 完了 (候補のある位置: te %.1f%%)' % (nm, 100 * sum(1 for d in CAND[nm]['te'] if d) / len(idx_te)))
del OLD
def prep(part, idx):
    lp, hi = M.logprob_at(pad, idx, return_hi=True); P = np.exp2(lp); owner = []; ws = []; F = []; seen = {}
    for p in range(len(idx)):
        row = {}
        for fi, nm in enumerate(NAMES):
            d = CAND[nm][part][p]
            if not d: continue
            for w, c in sorted(d.items(), key=lambda z: -z[1])[:TOPK]:
                if w not in row: row[w] = np.zeros(len(NAMES))
                row[w][fi] = np.log1p(c)
        for w, v in row.items(): owner.append(p); ws.append(w); F.append(v)
    owner = np.array(owner, int); ws = np.array(ws, np.uint16); F = np.array(F)
    ctx = np.stack([pad[idx[o] - 64:idx[o]] for o in owner]); syn = np.concatenate([np.zeros(8, np.uint16), np.concatenate([ctx, ws[:, None]], 1).ravel()]); si = 8 + np.arange(len(owner)) * 65 + 64
    Pw = np.exp2(M.logprob_at(syn, si)); return P, hi, owner, ws, F, Pw
def sub_of(part, idx, R, b):
    P, hi, owner, ws, F, Pw = R; m = hi == b; loc = -np.ones(len(P), int); loc[m] = np.arange(m.sum()); cm = m[owner]
    return dict(P=P[m], owner=loc[owner[cm]], ws=ws[cm], F=F[cm], Pw=Pw[cm], truth=pad[idx][m], n=int(m.sum()))
def bits_b(sb, lam):
    """バケット内の位置ごとの bit。 lam: 特徴ごとの重み (長さ = 特徴数)"""
    e = np.exp(sb['F'] @ lam); Z = np.ones(sb['n']); np.add.at(Z, sb['owner'], sb['Pw'] * (e - 1)); bonus = np.ones(sb['n']); m = sb['ws'] == sb['truth'][sb['owner']]; bonus[sb['owner'][m]] = e[m]
    return -np.log2(sb['P'] * bonus / Z)
R = {'te': prep('te', idx_te), 'tu': prep('tu', idx_tu)}; log('準備完了: 候補 te %d 件 tu %d 件' % (len(R['te'][2]), len(R['tu'][2])))
SB = {(p, b): sub_of(p, idx, R[p], b) for p, idx in (('te', idx_te), ('tu', idx_tu)) for b in range(1, 6)}
grid = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]; NF = len(NAMES)
def total(part, Lam): return sum(bits_b(SB[(part, b)], Lam[b]).sum() for b in range(1, 6) if SB[(part, b)]['n']) / sum(SB[(part, b)]['n'] for b in range(1, 6))
def tune_feature(Lam, fi):
    """特徴 fi の重みだけを バケットごとに 動かす (他は固定)"""
    for b in range(1, 6):
        sb = SB[('tu', b)]
        if not sb['n']: continue
        best = None
        for g in grid:
            Lam[b, fi] = g; v = bits_b(sb, Lam[b]).mean()
            if best is None or v < best[0] - 1e-9: best = (v, g)
        Lam[b, fi] = best[1]
base = total('te', np.zeros((6, NF))); print('RESULT S=%d  n-gram のみ %.4f bit/語' % (n_tok, base), flush=True)
chosen = []; Lam = np.zeros((6, NF)); cur_tu = total('tu', Lam)
for step in range(NF):                                           # 貪欲: 調整位置で いちばん下がる 特徴を 1 つ足す (足した後に 既存の重みを 1 巡 磨く)
    bestf = None
    for fi in range(NF):
        if fi in chosen: continue
        L2 = Lam.copy(); tune_feature(L2, fi); tl = total('tu', L2)
        if bestf is None or tl < bestf[0]: bestf = (tl, fi, L2)
    if cur_tu - bestf[0] < 0.0003: break
    Lam = bestf[2]; chosen.append(bestf[1])
    for fj in chosen: tune_feature(Lam, fj)
    cur_tu = total('tu', Lam); tb = total('te', Lam); print('RESULT S=%d  貪欲 %d: + %-6s -> test %.4f bit/語 (得 %.4f)' % (n_tok, step + 1, NAMES[bestf[1]], tb, base - tb), flush=True)
tb = total('te', Lam); print('RESULT S=%d  最終 %.4f (得 %.4f = %.2f%%)  重み(b=2..5) %s' % (n_tok, tb, base - tb, 100 * (base - tb) / base, {NAMES[f]: [float(Lam[b, f]) for b in (2, 3, 4, 5)] for f in chosen}), flush=True)
