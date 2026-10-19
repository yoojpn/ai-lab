"""
実験D: 記憶の予算(大きさ)と予測精度の曲線。「抽象的な知識(クラス・チャンク)を含む方が、小さな記憶で精度が高いか」。

方法: 数え上げの表から、出現回数が c 回未満の項目を捨てる(c を 1→50 と増やすと記憶が小さくなる)。
      方式ごとに「記憶の大きさ vs テストの bit/語」を並べる。
  W    : 単語4-gramだけ
  W+C  : 単語4-gram + 単語クラス(群)の表 (クラスは 200群 → 表が小さく、1つの項目が多くの出来事を代表する)
  FULL : W+C + チャンク + チャンククラス (=前回のはしご)
  FULL+局所 : FULL + 文書内の繰り返し・索引引き(文脈から作る一時的な索引なので、記憶には数えない)
  NN   : 小さなニューラル(固定サイズ。パラメーター数が記憶の大きさ)  ← 引数で学習量を指定(0なら省略)
記憶の大きさは「保存した項目数×16バイト」で見積もる(キー8+回数4+その他4)。
使い方: LADDER_CORPUS=... python3 exp_budget.py <訓練語数N> [NNのエポック数]
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time, copy
sys.path.insert(0, '/home/claude/hdc')
import numpy as np
from collections import Counter
import exp_ladder as L
import exp_scale as es
import exp_class as ec
import exp_cards as XC
import exp_far as XF

V = es.V; HIST = L.HIST
log = L.log
BYTES = 16


def pruned(model, vy, c):
    new = []
    for (uc, tot, typ, up, pc) in model.t:
        keep = pc >= c
        if not keep.any():
            keep[np.argmax(pc)] = True   # 空の表は扱えないので最大の1項目だけ残す
        up2, pc2 = up[keep], pc[keep]
        uc2, inv = np.unique(up2 // vy, return_inverse=True)
        new.append((uc2, np.bincount(inv, weights=pc2).astype(np.float64),
                    np.bincount(inv).astype(np.float64), up2, pc2))
    m = copy.copy(model)
    m.t = new
    return m


def entries(model):
    return sum(len(t[3]) for t in model.t)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    nn_ep = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
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
    Xtr, ytr, mtr, ptr = XF.lwindows_meta(tr_i, N)
    Xva, yva, mva, pva = XF.lwindows_meta(va_i, es.N_VAL)
    Xte, yte, mte, pte = XF.lwindows_meta(te_i, es.N_TEST)
    log('訓練%d 検証%d テスト%d [%.0fs]' % (len(Xtr), len(Xva), len(Xte), time.time() - t0))
    w3 = lambda X: X[:, HIST - 3:]
    pu = np.bincount(ytr, minlength=V).astype(np.float64) + 1.0
    pu /= pu.sum()
    A = es.NG(w3(Xtr), ytr)
    cls, wc = ec.exchange(mods, K, 6, lambda s: None)
    clsc = np.bincount(cls, weights=wc, minlength=K); wic = np.where(wc > 0, wc / np.maximum(clsc[cls], 1), 1e-9)
    B_ = ec.NGv(cls[w3(Xtr)], cls[ytr], K)
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
    log('準備完了 [%.0fs]' % (time.time() - t0))

    # 各モデルを(モデル, 語彙サイズ, 入力変換)で扱う
    specs = [
        ('A', A, V, lambda X, y, U: (w3(X), y), lambda p, X, y: p),
        ('B', B_, K, lambda X, y, U: (cls[w3(X)], cls[y]), lambda p, X, y: p * wic[y]),
        ('E', E_, V, lambda X, y, U: (cls[w3(X)], y), lambda p, X, y: p),
        ('F', F_, V, lambda X, y, U: (Xf(X), y), lambda p, X, y: p),
        ('C', C_, V, lambda X, y, U: (U, y), lambda p, X, y: p),
        ('D', D_, V, lambda X, y, U: (cu[U], y), lambda p, X, y: p),
    ]
    # B の出力語彙は K(クラス)。pruned の vy は各モデルの出力語彙
    vys = {'A': V, 'B': K, 'E': V, 'F': V, 'C': V, 'D': V}

    # 局所(記憶に数えない)
    cp_va, st_va = XC.copy_probs(Xva, yva, pu); cp_te, st_te = XC.copy_probs(Xte, yte, pu)
    ca_va = XC.cache_probs(Xva, yva, pu); ca_te = XC.cache_probs(Xte, yte, pu)
    fc_va = XF.farcache_probs(va_i, mva, pva, yva, 300, pu); fc_te = XF.farcache_probs(te_i, mte, pte, yte, 300, pu)
    lp_va, _ = XF.local_for_windows(va_i, mva, pva, yva, cls, pu); lp_te, _ = XF.local_for_windows(te_i, mte, pte, yte, cls, pu)
    loc_va = [cp_va, ca_va, fc_va] + [lp_va[n] for n in ['L1', 'L2', 'L3', 'S2', 'LC2', 'LCW']]
    loc_te = [cp_te, ca_te, fc_te] + [lp_te[n] for n in ['L1', 'L2', 'L3', 'S2', 'LC2', 'LCW']]
    log('局所部品を計算 [%.0fs]' % (time.time() - t0))

    def probs_of(name, m, X, y, U):
        _, mod, _, inp, post = [s for s in specs if s[0] == name][0]
        x, yy = inp(X, y, U)
        r = m.probs(x, yy)
        r = r[0] if isinstance(r, tuple) else r
        return post(r, X, y)

    base_bits = None
    rows = []
    for c in [1, 2, 3, 4, 6, 10, 20, 50]:
        pm = {n: (pruned(mod, vys[n], c) if c > 1 else mod) for n, mod, *_ in specs}
        sz = {n: entries(pm[n]) for n in pm}
        pv = {n: probs_of(n, pm[n], Xva, yva, Uva) for n in pm}
        pt = {n: probs_of(n, pm[n], Xte, yte, Ute) for n in pm}
        if c == 1:
            base_bits = -np.log2(pt['A']).mean()

        def mixbits(names, extra_v=(), extra_t=()):
            Pv = np.array([pv[n] for n in names] + list(extra_v)); Pt = np.array([pt[n] for n in names] + list(extra_t))
            w = L.em(Pv)
            return -np.log2((w[:, None] * Pt).sum(0)).mean()
        sets = {'W': ['A'], 'W+C': ['A', 'B', 'E', 'F'], 'FULL': ['A', 'B', 'E', 'F', 'C', 'D']}
        res = {}
        for k, names in sets.items():
            res[k] = (sum(sz[n] for n in names) * BYTES / 1e6, mixbits(names))
        res['FULL+局所'] = (res['FULL'][0], mixbits(sets['FULL'], loc_va, loc_te))
        res['W+局所'] = (res['W'][0], mixbits(['A'], loc_va, loc_te))
        rows.append((c, res))
        log('c=%-3d ' % c + ' | '.join('%s %.1fMB %.3f(%+.1f%%)' % (k, v[0], v[1], 100 * (1 - v[1] / base_bits))
                                       for k, v in res.items()) + '  [%.0fs]' % (time.time() - t0))

    nn_pt = None
    if nn_ep > 0:
        import exp_nn2 as nn2
        rng = np.random.default_rng(0)
        net = nn2.Net(rng, np.zeros(V)); net.qprob = np.ones(V) / V
        X3 = w3(Xtr); Bn = nn2.B; total = int(nn_ep * (len(X3) // Bn)); step = 0
        while step < total:
            perm = rng.permutation(len(X3))
            for i in range(0, len(perm) - Bn + 1, Bn):
                ix = perm[i:i + Bn]
                net.step(X3[ix], ytr[ix], rng, False, 2e-3 if step < 0.6 * total else 1e-3); step += 1
                if step >= total:
                    break
        nn_pt = net.probs_true(w3(Xte), yte)
        nn_pv = net.probs_true(w3(Xva), yva)
        npar = sum(v.size for v in net.p.values())
        b = -np.log2(nn_pt).mean()
        log('\nNN(固定サイズ) パラメーター%d個=%.1fMB: %.3f bit/語 (%+.1f%%) [%.0fs]' % (npar, npar * 4 / 1e6, b, 100 * (1 - b / base_bits), time.time() - t0))
        # NN + FULL(c=1 と c=10 と c=50)
        for c in (1, 10, 50):
            pm = {n: (pruned(mod, vys[n], c) if c > 1 else mod) for n, mod, *_ in specs}
            names = ['A', 'B', 'E', 'F', 'C', 'D']
            Pv = np.array([probs_of(n, pm[n], Xva, yva, Uva) for n in names] + [nn_pv])
            Pt = np.array([probs_of(n, pm[n], Xte, yte, Ute) for n in names] + [nn_pt])
            w = L.em(Pv); bb = -np.log2((w[:, None] * Pt).sum(0)).mean()
            mb = sum(entries(pm[n]) for n in names) * BYTES / 1e6 + npar * 4 / 1e6
            log('  FULL(c=%d) + NN: 合計記憶 %.1fMB  %.3f bit/語 (%+.1f%%)' % (c, mb, bb, 100 * (1 - bb / base_bits)))
    log('\n基準: 単語4-gram(刈らない) %.3f bit/語 = %.1fMB   [N=%d]' % (base_bits, entries(A) * BYTES / 1e6, N))


if __name__ == '__main__':
    main()
