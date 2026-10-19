"""
実験C: 「遠くまで見る」部品は、数え上げのはしごの上に得を足すか。(GPU不要・数えるだけ)

はしごは直前3語、カードは直前4語しか見ていなかった。ここに、直前 W 語(既定300)を見る部品を足す。
  遠いキャッシュ : 直前W語の中に出た語は、また出やすい(確率 = 出た回数/W を混ぜる)
  話題の連想     : 直前W語(ただし直前12語より前)に語Aがあったとき、次の語がBになりやすい、を訓練で数える。
                   予測は、履歴にある各語Aの「Bになりやすさ」を平均する(索引で引くだけ。全部の組み合わせは見ない)。
                   助詞・冠詞など頻出語(上位100語)は話題を持たないので、Aには使わない。
基準: はしご(単語・クラス・チャンク・チャンククラス) + 繰り返し規則 + 近いキャッシュ。 + 遠い部品 で何%縮むか。
使い方: LADDER_CORPUS=... python3 exp_far.py <訓練語数N> [W]
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time
sys.path.insert(0, '/home/claude/hdc')
import numpy as np
from collections import Counter
import exp_ladder as L
import exp_scale as es
import exp_class as ec
import exp_cards as XC

V = es.V; HIST = L.HIST
log = L.log
SW = 100          # 頻出語(話題を持たない)
NEAR = HIST       # 直前12語は他の部品が見ている


def lwindows_meta(mods, limit):
    X, Y, MID, POS, tot = [], [], [], [], 0
    for mi, a in enumerate(mods):
        n = len(a)
        if n <= HIST:
            continue
        X.append(np.stack([a[j:n - HIST + j] for j in range(HIST)], axis=1)); Y.append(a[HIST:])
        MID.append(np.full(n - HIST, mi, np.int32)); POS.append(np.arange(HIST, n, dtype=np.int32))
        tot += n - HIST
        if tot >= limit:
            break
    return (np.concatenate(X)[:limit], np.concatenate(Y)[:limit],
            np.concatenate(MID)[:limit], np.concatenate(POS)[:limit])


def far_words(a, p, W):
    """位置pの語を予測するときの遠い履歴(直前NEAR語は除く)の、話題を持つ語の集合"""
    lo = max(0, p - W)
    hi = p - NEAR
    if hi <= lo:
        return np.empty(0, np.int64)
    u = np.unique(a[lo:hi])
    return u[u > SW]


def build_assoc(mods, mid, pos, y, W, step, beta=20.0):
    """訓練: 履歴に語Aがあるとき次の語がyだった回数の表 C[A,y]"""
    C = np.zeros(V * V, np.float64)
    idx = np.arange(0, len(y), step)
    bufA, bufY, nb = [], [], 0
    def flush():
        nonlocal bufA, bufY, nb
        if bufA:
            A = np.concatenate(bufA); Y = np.concatenate(bufY)
            C[:] += np.bincount(A * V + Y, minlength=V * V)
        bufA, bufY, nb = [], [], 0
    for k in idx:
        u = far_words(mods[mid[k]], int(pos[k]), W)
        if len(u):
            bufA.append(u); bufY.append(np.full(len(u), y[k], np.int64)); nb += len(u)
            if nb > 4_000_000:
                flush()
    flush()
    C = C.reshape(V, V)
    return C


def assoc_probs(Cn, mods, mid, pos, y, W):
    """各位置の、実際の語yの予測確率(履歴の各語Aの Cn[A,y] の平均)と、履歴にある語数"""
    n = len(y)
    pidx, As = [], []
    for k in range(n):
        u = far_words(mods[mid[k]], int(pos[k]), W)
        if len(u):
            pidx.append(np.full(len(u), k, np.int64)); As.append(u)
    pidx = np.concatenate(pidx); As = np.concatenate(As)
    vals = Cn[As, y[pidx]]
    cnt = np.bincount(pidx, minlength=n).astype(np.float64)
    s = np.bincount(pidx, weights=vals, minlength=n)
    return s, cnt


def topic_nb(R, lpu, mods, mid, pos, y, W, alphas, chunk=4000):
    """証拠を積み上げる話題予測: p(y) ∝ pu(y)·exp(α·Σ_A R[A,y])。αごとの、実際の語の確率を返す"""
    import scipy.sparse as sp
    n = len(y)
    out = {a: np.empty(n) for a in alphas}
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        rows, cols = [], []
        for k in range(lo, hi):
            u = far_words(mods[mid[k]], int(pos[k]), W)
            rows.append(np.full(len(u), k - lo, np.int64)); cols.append(u)
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        H = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(hi - lo, V))
        S = np.asarray(H @ R, dtype=np.float32)            # (チャンク, V)
        yy = y[lo:hi]
        for a in alphas:
            lg = lpu[None, :] + a * S
            lg -= lg.max(1, keepdims=True)
            e = np.exp(lg)
            out[a][lo:hi] = e[np.arange(hi - lo), yy] / e.sum(1)
    return out


def doc_local(a, cls, pu, gam=0.3):
    """文書 a の各位置で、同じ文脈キーが過去(文書の先頭から)にあったときの、実際の語の確率(索引で引くだけ)"""
    import pandas as pd
    n = len(a)
    sh = lambda k: np.concatenate([np.full(k, -1, np.int64), a[:n - k]]) if n > k else np.full(n, -1, np.int64)
    a1, a2, a3 = sh(1), sh(2), sh(3)
    clx = lambda x: np.where(x >= 0, cls[np.maximum(x, 0)], -1)
    c1, c2 = clx(a1), clx(a2)
    keys = {
        'L1': (a1, a1 >= 0), 'L2': (a2 * V + a1, a2 >= 0), 'L3': ((a3 * V + a2) * V + a1, a3 >= 0),
        'S2': (a2, a2 >= 0), 'LC2': (c2 * 1000 + c1, a2 >= 0), 'LCW': (c2 * V + a1, a2 >= 0),
    }
    out, nctx = {}, {}
    ar = np.arange(n, dtype=np.int64)
    for name, (k, ok) in keys.items():
        k = np.where(ok, k, -1 - ar)
        n_ctx = pd.Series(k).groupby(k).cumcount().to_numpy()
        n_ctxy = pd.Series(k * V + a).groupby(k * V + a).cumcount().to_numpy()
        out[name] = (n_ctxy + gam * pu[a]) / (n_ctx + gam)
        nctx[name] = n_ctx
    return out, nctx


def local_for_windows(docs, mid, pos, y, cls, pu):
    cache, P, N = {}, {}, {}
    n = len(y)
    for k in range(n):
        m = int(mid[k])
        if m not in cache:
            cache[m] = doc_local(docs[m], cls, pu)
        o, nc = cache[m]
        for name in o:
            P.setdefault(name, np.empty(n))[k] = o[name][pos[k]]
            N.setdefault(name, np.empty(n, np.int64))[k] = nc[name][pos[k]]
    return P, N


def build_embeddings(Xtr, ytr, dim=64):
    """共起(前後2語以内)→ PPMI → SVD で、単語の座標を作る(GPU不要)"""
    from scipy.sparse.linalg import svds
    C = np.zeros(V * V, np.float64)
    for a, b in ((Xtr[:, HIST - 1], ytr), (ytr, Xtr[:, HIST - 1]), (Xtr[:, HIST - 2], ytr), (ytr, Xtr[:, HIST - 2])):
        C += np.bincount(a * V + b, minlength=V * V)
    C = C.reshape(V, V)
    rs = C.sum(1) + 1e-9; cs = C.sum(0) ** 0.75 + 1e-9
    tot = C.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(C * cs.sum() / (rs[:, None] * cs[None, :]))
    pmi = np.where(np.isfinite(pmi), np.maximum(pmi, 0), 0).astype(np.float32)
    rng = np.random.default_rng(0)
    Y = pmi @ rng.standard_normal((V, dim + 20)).astype(np.float32)
    for _ in range(4):
        Q, _r = np.linalg.qr(Y)
        Q, _r = np.linalg.qr(pmi @ (pmi.T @ Q))
        Y = Q
    Q, _r = np.linalg.qr(Y)
    Ub, S, _vt = np.linalg.svd(Q.T @ pmi, full_matrices=False)
    U, S = (Q @ Ub)[:, :dim], S[:dim]
    E = U * np.sqrt(S)[None, :]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E.astype(np.float32), rs


def kmeans(E, K, w, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(E), K, replace=False)
    Cn = E[idx].copy()
    for _ in range(iters):
        lab = (E @ Cn.T).argmax(1)
        for k in range(K):
            m = lab == k
            if m.any():
                v = (E[m] * w[m, None]).sum(0)
                Cn[k] = v / (np.linalg.norm(v) + 1e-9)
    return (E @ Cn.T).argmax(1).astype(np.int64)


def farcache_probs(mods, mid, pos, y, W, pu):
    n = len(y)
    out = np.empty(n)
    for k in range(n):
        a = mods[mid[k]]; p = int(pos[k]); lo = max(0, p - W)
        seg = a[lo:p]
        out[k] = (seg == y[k]).sum() / max(len(seg), 1)
    return 0.5 * out + 0.5 * pu[y]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    K, M, K2 = 200, 1500, 200
    t0 = time.time()
    corp = es.build_corpus()
    cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr_i, va_i, te_i = conv(corp[2]), conv(corp[1]), conv(corp[0])
    mods, tot = [], 0
    for a in tr_i:
        if len(a) <= HIST:
            continue
        mods.append(a); tot += len(a) - HIST
        if tot >= N:
            break
    Xtr, ytr, mtr, ptr = lwindows_meta(tr_i, N)
    Xva, yva, mva, pva = lwindows_meta(va_i, es.N_VAL)
    Xte, yte, mte, pte = lwindows_meta(te_i, es.N_TEST)
    log('訓練%d 検証%d テスト%d 窓W=%d [%.0fs]' % (len(Xtr), len(Xva), len(Xte), W, time.time() - t0))
    w3 = lambda X: X[:, HIST - 3:]
    pu = np.bincount(ytr, minlength=V).astype(np.float64) + 1.0
    pu /= pu.sum()
    # ---- 基準のはしご(exp_cards と同じ)
    A = es.NG(w3(Xtr), ytr)
    pA_va, mk_va = A.probs(w3(Xva), yva); pA_te, mk_te = A.probs(w3(Xte), yte)
    cls, wc = ec.exchange(mods, K, 6, lambda s: None)
    clsc = np.bincount(cls, weights=wc, minlength=K); wic = np.where(wc > 0, wc / np.maximum(clsc[cls], 1), 1e-9)
    B_ = ec.NGv(cls[w3(Xtr)], cls[ytr], K)
    pB = lambda X, y: B_.probs(cls[w3(X)], cls[y]) * wic[y]
    E_ = L.NG2(cls[w3(Xtr)], ytr, K, V)
    Xf = lambda X: np.stack([cls[X[:, HIST - 3]], cls[X[:, HIST - 2]], X[:, HIST - 1]], axis=1)
    F_ = L.NG2(Xf(Xtr), ytr, V, V)
    lex = L.learn_bpe(mods, M)
    parse, Lmax = L.make_parser(lex)
    Vn = V + M + 1; PAD = V + M
    Utr = L.unit_ctx(Xtr, parse, PAD); Uva = L.unit_ctx(Xva, parse, PAD); Ute = L.unit_ctx(Xte, parse, PAD)
    C_ = L.NG2(Utr, ytr, Vn, V)
    unit_mods = [np.array(parse(tuple(a.tolist())), np.int64) for a in mods]
    ec.V = Vn
    cu, _ = ec.exchange(unit_mods, K2, 6, lambda s: None)
    ec.V = V
    cu = np.append(cu, K2)
    D_ = L.NG2(cu[Utr], ytr, K2 + 1, V)
    log('基準のはしご準備完了 [%.0fs]' % (time.time() - t0))
    base_va = [pA_va, pB(Xva, yva), E_.probs(cls[w3(Xva)], yva), F_.probs(Xf(Xva), yva), C_.probs(Uva, yva), D_.probs(cu[Uva], yva)]
    base_te = [pA_te, pB(Xte, yte), E_.probs(cls[w3(Xte)], yte), F_.probs(Xf(Xte), yte), C_.probs(Ute, yte), D_.probs(cu[Ute], yte)]
    base_bits = -np.log2(pA_te).mean()
    cp_va, st_va = XC.copy_probs(Xva, yva, pu); cp_te, st_te = XC.copy_probs(Xte, yte, pu)
    ca_va = XC.cache_probs(Xva, yva, pu); ca_te = XC.cache_probs(Xte, yte, pu)

    # ---- 遠い部品
    step = max(1, len(ytr) // 1_000_000)
    C = build_assoc(tr_i, mtr, ptr, ytr, W, step)
    rs = C.sum(1, keepdims=True)
    beta = 20.0
    Cn = (C + beta * pu[None, :]) / (rs + beta)
    log('話題の連想の表を作成(訓練窓の%d分の1を使用)[%.0fs]' % (step, time.time() - t0))
    sv, cv = assoc_probs(Cn, va_i, mva, pva, yva, W); st_, ct_ = assoc_probs(Cn, te_i, mte, pte, yte, W)
    pas_va = np.where(cv > 0, sv / np.maximum(cv, 1), pu[yva]); pas_te = np.where(ct_ > 0, st_ / np.maximum(ct_, 1), pu[yte])
    fc_va = farcache_probs(va_i, mva, pva, yva, W, pu); fc_te = farcache_probs(te_i, mte, pte, yte, W, pu)
    R = np.clip(np.log(Cn / pu[None, :]), -3, 3).astype(np.float32)
    alphas = [0.003, 0.006, 0.012, 0.024, 0.05, 0.1]
    nbv = topic_nb(R, np.log(pu).astype(np.float32), va_i, mva, pva, yva, W, alphas)
    ll = {a: -np.log2(nbv[a]).mean() for a in alphas}
    best_a = min(ll, key=ll.get)
    log('話題予測(証拠の積み上げ) 検証 bit/語: ' + ' '.join('α=%g:%.3f' % (a, ll[a]) for a in alphas) + ' → α=%g を採用' % best_a)
    nb_va = nbv[best_a]
    nb_te = topic_nb(R, np.log(pu).astype(np.float32), te_i, mte, pte, yte, W, [best_a])[best_a]
    log('遠い部品の確率を計算 [%.0fs]' % (time.time() - t0))

    # ---- 状況の分け方: (近い繰り返し候補) × (4-gramの最長文脈) × (遠い履歴の語数の多寡)
    mkb = lambda mk: np.where(mk <= 1, 0, mk - 1)
    def bk(st, mk, c):
        return (st * 3 + mkb(mk)) * 2 + (c >= 20)
    bk_va = bk(st_va, mk_va, cv); bk_te = bk(st_te, mk_te, ct_)
    NB = 18

    def evaluate(pv, pt, label):
        Pv, Pt = np.array(pv), np.array(pt)
        Wt = XC.bucket_em(Pv, bk_va, NB)
        b = XC.mix_bits(Pt, bk_te, Wt).mean()
        log('  %-58s %.3f  4-gram比の改善 %5.2f%%' % (label, b, 100 * (1 - b / base_bits)))
        return b

    log('\n単語4-gram テスト %.3f bit/語 / 窓W=%d / 遠い履歴に話題語が20語以上ある位置: %.0f%%'
        % (base_bits, W, 100 * np.mean(ct_ >= 20)))
    log('各部品単独の bit/語: 話題の連想(平均) %.3f / 話題予測(積み上げ) %.3f / 遠いキャッシュ %.3f / 一様(参考) %.3f'
        % (-np.log2(pas_te).mean(), -np.log2(nb_te).mean(), -np.log2(fc_te).mean(), -np.log2(pu[yte]).mean()))
    log('\n=== 同じ「状況別の重み」で比べる ===')
    b0 = evaluate(base_va, base_te, 'はしご')
    b1 = evaluate(base_va + [cp_va, ca_va], base_te + [cp_te, ca_te], 'はしご + 近い繰り返し・キャッシュ(=前回の基準)')
    b2 = evaluate(base_va + [cp_va, ca_va, fc_va], base_te + [cp_te, ca_te, fc_te], '  + 遠いキャッシュ')
    b3 = evaluate(base_va + [cp_va, ca_va, pas_va], base_te + [cp_te, ca_te, pas_te], '  + 話題の連想')
    b4 = evaluate(base_va + [cp_va, ca_va, fc_va, pas_va], base_te + [cp_te, ca_te, fc_te, pas_te], '  + 遠いキャッシュ + 話題の連想')
    # ---- 文書内の索引引き(その場で作る・全文書が対象)
    t1 = time.time()
    lp_va, ln_va = local_for_windows(va_i, mva, pva, yva, cls, pu)
    lp_te, ln_te = local_for_windows(te_i, mte, pte, yte, cls, pu)
    log('文書内の索引引きを計算 [%.0fs]' % (time.time() - t1))
    names = ['L1', 'L2', 'L3', 'S2', 'LC2', 'LCW']
    log('各キー単独の bit/語(文書内で過去に同じ文脈があった位置だけ・割合): ' + ' / '.join(
        '%s %.3f(%.0f%%)' % (nm, -np.log2(lp_te[nm][ln_te[nm] > 0]).mean(), 100 * np.mean(ln_te[nm] > 0)) for nm in names))
    depth = lambda ln: (ln['L1'] > 0).astype(int) + (ln['L2'] > 0) + (ln['L3'] > 0)
    dd = lambda ln: np.where(depth(ln) <= 1, 0, depth(ln) - 1)
    bk2_va = (st_va * 3 + mkb(mk_va)) * 3 + dd(ln_va); bk2_te = (st_te * 3 + mkb(mk_te)) * 3 + dd(ln_te)

    def evaluate2(pv, pt, label):
        Pv, Pt = np.array(pv), np.array(pt)
        Wt = XC.bucket_em(Pv, bk2_va, 27)
        b = XC.mix_bits(Pt, bk2_te, Wt).mean()
        log('  %-58s %.3f  4-gram比の改善 %5.2f%%' % (label, b, 100 * (1 - b / base_bits)))
        return b
    log('\n=== 文書内の索引引きを足す(新しい状況分け。基準も同じ分けで測り直す) ===')
    bb = base_va + [cp_va, ca_va, fc_va]; bt = base_te + [cp_te, ca_te, fc_te]
    r0 = evaluate2(bb, bt, '基準(はしご+繰り返し+キャッシュ+遠いキャッシュ)')
    for nm in names:
        evaluate2(bb + [lp_va[nm]], bt + [lp_te[nm]], '  + %s のみ' % nm)
    r1 = evaluate2(bb + [lp_va[nm] for nm in names], bt + [lp_te[nm] for nm in names], '  + 6種すべて')
    log('  → 索引引きによる追加の改善 %.2f ポイント' % (100 * (r0 - r1) / base_bits))

    # ---- 機械が作った座標でグループ分け(3段階の粗さ)し、学びを似た単語に広げる
    t1 = time.time()
    Emb, rowsum = build_embeddings(Xtr, ytr)
    wts = np.sqrt(rowsum / rowsum.sum() * V)
    log('単語の座標(64次元)を共起のSVDで作成 [%.0fs]' % (time.time() - t1))
    ev_names, ev_va, ev_te = [], [], []
    for Kc in (60, 300, 1000):
        cl = kmeans(Emb, Kc, wts)
        Ea = L.NG2(cl[w3(Xtr)], ytr, Kc, V)
        Xfc = lambda X: np.stack([cl[X[:, HIST - 3]], cl[X[:, HIST - 2]], X[:, HIST - 1]], axis=1)
        Fa = L.NG2(Xfc(Xtr), ytr, max(Kc, V), V)
        ev_names += ['座標群K=%d(群→語)' % Kc, '座標群K=%d(群,群,語)' % Kc]
        ev_va += [Ea.probs(cl[w3(Xva)], yva), Fa.probs(Xfc(Xva), yva)]
        ev_te += [Ea.probs(cl[w3(Xte)], yte), Fa.probs(Xfc(Xte), yte)]
    ex = []
    for wd in ('king', 'france', 'run', 'he', 'blue'):
        if wd in vocab:
            sim = Emb @ Emb[vocab[wd]]
            inv = {i: w for w, i in vocab.items()}
            nn = [inv[i] for i in np.argsort(-sim)[1:7] if i in inv]
            ex.append('%s→%s' % (wd, ','.join(nn)))
    log('座標が近い単語の例: ' + ' / '.join(ex))
    log('座標群の部品を計算 [%.0fs]' % (time.time() - t1))
    for nm, v_, t_ in zip(ev_names, ev_va, ev_te):
        log('  単独 %-26s %.3f bit/語' % (nm, -np.log2(t_).mean()))
    bbl = bb + [lp_va[nm] for nm in names]; btl = bt + [lp_te[nm] for nm in names]
    log('\n=== 座標群の部品を足す(これまでの全部品の上に) ===')
    r2 = evaluate2(bbl, btl, '基準(はしご+繰り返し+キャッシュ+遠い+索引引き6種)')
    for Kc_i, Kc in enumerate((60, 300, 1000)):
        evaluate2(bbl + ev_va[2 * Kc_i:2 * Kc_i + 2], btl + ev_te[2 * Kc_i:2 * Kc_i + 2], '  + 座標群 K=%d' % Kc)
    r3 = evaluate2(bbl + ev_va, btl + ev_te, '  + 座標群 3段階すべて')
    log('  → 座標群による追加の改善 %.2f ポイント / 全部品の合計 改善 %.2f%%' % (100 * (r2 - r3) / base_bits, 100 * (1 - r3 / base_bits)))

    b5 = evaluate(base_va + [cp_va, ca_va, nb_va], base_te + [cp_te, ca_te, nb_te], '  + 話題予測(積み上げ)')
    b6 = evaluate(base_va + [cp_va, ca_va, fc_va, nb_va], base_te + [cp_te, ca_te, fc_te, nb_te], '  + 遠いキャッシュ + 話題予測(積み上げ)')
    b4 = min(b4, b5, b6)
    log('\n要約(4-gram基準 %.3f): 遠い部品による追加の改善 = %.2f ポイント   [N=%d W=%d %.0fs]'
        % (base_bits, 100 * (b1 - b4) / base_bits, N, W, time.time() - t0))


if __name__ == '__main__':
    main()
