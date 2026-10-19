"""
圧縮の階段(ladder): 下の段が見つけた単位を、上の段が新しい記号として使い、さらに圧縮する。
  段0: 単語
  段1: 単語のクラス(尤度で自動に束ねる) …… 実験で確認済みの 4〜5% 改善
  段2: よく続く語の連なりを新しい単位(チャンク)にする(BPE型。頻度の高い隣り合わせを併合)
       → 同じ「直前3つ」でも、より長い範囲を見られる
  段3: チャンクのクラス(段2の単位を、さらにクラスに束ねる)
各段のモデルは「次の単語」の確率を出し、重みはEM法で検証データから決める(混合)。
問い: 段を積むほど、4-gram に対する改善は積み上がるか?(積み上がれば、時間とともに複利で賢くなる設計)
使い方: python3 exp_ladder.py <訓練語数N> <単語クラス数K> <併合回数M> <チャンククラス数K2>
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time
import numpy as np
from collections import Counter
sys.path.insert(0, '/home/claude/hdc')
import exp_scale as es
import exp_class as ec

es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)
V = es.V; DELTA = es.DELTA; HIST = 12
log = lambda s: print(s, flush=True)


def lwindows(mods, limit=None):
    X, Y, tot = [], [], 0
    for a in mods:
        n = len(a)
        if n <= HIST:
            continue
        X.append(np.stack([a[j:n - HIST + j] for j in range(HIST)], axis=1)); Y.append(a[HIST:]); tot += n - HIST
        if limit and tot >= limit:
            break
    X = np.concatenate(X); Y = np.concatenate(Y)
    return (X[:limit], Y[:limit]) if limit else (X, Y)


class NG2:
    """直前3つの記号(各 < vc)から次の単語(< vy)を当てる。補間つき絶対値割引。短い文脈へ後退する"""
    def __init__(s, X, y, vc, vy):
        s.vc, s.vy, s.t = vc, vy, []
        for k in range(4):
            ck = s.code(X, k)
            up, pc = np.unique(ck * vy + y, return_counts=True)
            uc, tot = np.unique(ck, return_counts=True)
            _, typ = np.unique(up // vy, return_counts=True)
            s.t.append((uc, tot, typ, up, pc))

    def code(s, X, k):
        c = np.zeros(len(X), np.int64)
        for j in range(k):
            c += X[:, 2 - j].astype(np.int64) * (s.vc ** j)
        return c

    def probs(s, X, y):
        p = np.full(len(y), 1.0 / s.vy); alive = np.ones(len(y), bool)
        for k in range(4):
            uc, tot, typ, up, pc = s.t[k]
            cq = s.code(X, k)
            i = np.minimum(np.searchsorted(uc, cq), len(uc) - 1)
            found = alive & (uc[i] == cq)
            pq = cq * s.vy + y
            j = np.minimum(np.searchsorted(up, pq), len(up) - 1)
            c = np.where(up[j] == pq, pc[j], 0)
            p = np.where(found, np.maximum(c - DELTA, 0) / tot[i] + DELTA * typ[i] / tot[i] * p, p)
            alive = found
        return p


def learn_bpe(mods, M, minc=15, crit='count'):
    SEP = V + M; B = V + M + 1
    a = np.concatenate([np.concatenate([m, [SEP]]) for m in mods]).astype(np.int64)
    lex = {}
    for m in range(M):
        ok = (a[:-1] != SEP) & (a[1:] != SEP)
        u, c = np.unique((a[:-1] * B + a[1:])[ok], return_counts=True)
        if crit == 'mdl':
            cu_ = np.bincount(a[a != SEP], minlength=B).astype(np.float64)
            Nt = float((a != SEP).sum())
            xs, ys = u // B, u % B
            sc = np.where(c >= minc, c * np.log2(c * Nt / (cu_[xs] * cu_[ys])) - 2 * np.log2(B), -1e18)
            j = sc.argmax()
            if sc[j] <= 0:
                break
        else:
            j = c.argmax()
            if c[j] < minc:
                break
        x, y = divmod(int(u[j]), B)
        nid = V + m
        idx = np.flatnonzero((a[:-1] == x) & (a[1:] == y))
        if x == y:
            keep, last = [], -2
            for i in idx:
                if i > last + 1:
                    keep.append(i); last = i
            idx = np.array(keep)
        a[idx] = nid
        a = np.delete(a, idx + 1)
        lex[nid] = lex.get(x, (x,)) + lex.get(y, (y,))
    return lex


def make_parser(lex):
    d = {t: u for u, t in lex.items()}
    Lmax = max(len(t) for t in d)
    firsts = {t[0] for t in d}

    def parse(win):
        out, i, n = [], 0, len(win)
        while i < n:
            if win[i] in firsts:
                for L in range(min(Lmax, n - i), 1, -1):
                    u = d.get(win[i:i + L])
                    if u is not None:
                        out.append(u); i += L; break
                else:
                    out.append(win[i]); i += 1
            else:
                out.append(win[i]); i += 1
        return out
    return parse, Lmax


def unit_ctx(X, parse, pad):
    out = np.full((len(X), 3), pad, np.int64)
    for r, row in enumerate(X.tolist()):
        u = parse(tuple(row))[-3:]
        out[r, 3 - len(u):] = u
    return out


def em(P, iters=80):
    w = np.ones(len(P)) / len(P)
    for _ in range(iters):
        mix = (w[:, None] * P).sum(0)
        w = (w[:, None] * P / mix).mean(1)
    return w


def main():
    N, K, M, K2 = (int(x) for x in sys.argv[1:5])
    crit = sys.argv[5] if len(sys.argv) > 5 else 'count'
    nn_ep = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
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
    Xtr, ytr = lwindows(tr_i, N); Xva, yva = lwindows(va_i, es.N_VAL); Xte, yte = lwindows(te_i, es.N_TEST)
    log('訓練%d 検証%d テスト%d [%.0fs]' % (len(Xtr), len(Xva), len(Xte), time.time() - t0))
    w3 = lambda X: X[:, HIST - 3:]
    # --- 段0: 単語4-gram
    A = es.NG(w3(Xtr), ytr)
    pA_va, _ = A.probs(w3(Xva), yva); pA_te, _ = A.probs(w3(Xte), yte)
    log('段0 単語4-gram bit/語: テスト%.3f' % (-np.log2(pA_te).mean()))
    # --- 段1: 単語クラス
    cls, wc = ec.exchange(mods, K, 6, lambda s: None)
    clsc = np.bincount(cls, weights=wc, minlength=K); wic = np.where(wc > 0, wc / np.maximum(clsc[cls], 1), 1e-9)
    B_ = ec.NGv(cls[w3(Xtr)], cls[ytr], K)
    pB = lambda X, y: B_.probs(cls[w3(X)], cls[y]) * wic[y]
    E_ = NG2(cls[w3(Xtr)], ytr, K, V)                                   # クラス文脈から単語を直接
    Xf = lambda X: np.stack([cls[X[:, HIST - 3]], cls[X[:, HIST - 2]], X[:, HIST - 1]], axis=1)
    F_ = NG2(Xf(Xtr), ytr, V, V)                                       # (クラス, クラス, 直前の単語)
    log('段1 クラス学習済み [%.0fs]' % (time.time() - t0))
    # --- 段2: チャンク
    lex = learn_bpe(mods, M, crit=crit)
    parse, Lmax = make_parser(lex)
    lens = Counter(len(t) for t in lex.values())
    ex = sorted(lex.items(), key=lambda kv: -len(kv[1]))[:0]
    inv = {i: w for w, i in vocab.items()}; inv[0] = '<unk>'
    log('段2 チャンク%d個(長さ別 %s) [%.0fs]' % (len(lex), dict(sorted(lens.items())), time.time() - t0))
    for nid in list(lex)[:6] + list(lex)[len(lex) // 2:len(lex) // 2 + 4] + list(lex)[-4:]:
        log('   チャンク: ' + ' '.join(inv.get(w, '?') for w in lex[nid]))
    Vn = V + M + 1; PAD = V + M
    Utr = unit_ctx(Xtr, parse, PAD); Uva = unit_ctx(Xva, parse, PAD); Ute = unit_ctx(Xte, parse, PAD)
    log('   チャンク文脈作成 [%.0fs]' % (time.time() - t0))
    C_ = NG2(Utr, ytr, Vn, V)
    # --- 段3: チャンクのクラス
    unit_mods = []
    for a in mods:
        unit_mods.append(np.array(parse(tuple(a.tolist())), np.int64))
    ec.V = Vn
    cu, _ = ec.exchange(unit_mods, K2, 6, lambda s: None)
    ec.V = V
    cu = np.append(cu, K2)
    D_ = NG2(cu[Utr], ytr, K2 + 1, V)
    log('段3 チャンククラス学習済み [%.0fs]' % (time.time() - t0))
    # --- 評価
    names = ['A 単語4-gram', 'B クラス4-gram(次のクラス→語)', 'E クラス文脈→語', 'F (クラス,クラス,語)→語', 'C チャンク文脈→語', 'D チャンククラス文脈→語']
    pv = [pA_va, pB(Xva, yva), E_.probs(cls[w3(Xva)], yva), F_.probs(Xf(Xva), yva), C_.probs(Uva, yva), D_.probs(cu[Uva], yva)]
    pt = [pA_te, pB(Xte, yte), E_.probs(cls[w3(Xte)], yte), F_.probs(Xf(Xte), yte), C_.probs(Ute, yte), D_.probs(cu[Ute], yte)]
    if nn_ep > 0:
        cache = os.environ.get('LADDER_CORPUS', 'x').replace('.pkl', '') + '_nn%.1f.npz' % nn_ep
        if os.path.exists(cache):
            z = np.load(cache); gv, gt = z['gv'], z['gt']
        else:
            import exp_nn2 as nn2
            rng = np.random.default_rng(0)
            net = nn2.Net(rng, np.zeros(V)); net.qprob = np.ones(V) / V
            X3 = w3(Xtr); Bn = nn2.B; total = int(nn_ep * (len(X3) // Bn)); step = 0
            while step < total:
                perm = rng.permutation(len(X3))
                for i in range(0, len(perm) - Bn + 1, Bn):
                    ix = perm[i:i + Bn]
                    net.step(X3[ix], ytr[ix], rng, False, 2e-3 if step < 0.6 * total else 1e-3); step += 1
                    if step % 500 == 0:
                        log('   NN step %d/%d [%.0fs]' % (step, total, time.time() - t0))
                    if step >= total:
                        break
            gv = net.probs_true(w3(Xva), yva); gt = net.probs_true(w3(Xte), yte)
            np.savez(cache, gv=gv, gt=gt)
        names.append('G 小ニューラル(3語文脈)')
        pv.append(gv); pt.append(gt)
    pv = np.array(pv); pt = np.array(pt)
    base = -np.log2(pt[0]).mean()
    log('\n--- 各モデル単独のテスト bit/語 ---')
    for n_, p in zip(names, pt):
        log('  %-30s %.3f' % (n_, -np.log2(p).mean()))
    log('\n--- 段を積んだ混合(EMで重み決定)。基準=単語4-gram %.3f ---' % base)
    sets = [('段0のみ', [0]), ('段0+段1(クラス)', [0, 1, 2, 3]), ('段0+段1+段2(チャンク)', [0, 1, 2, 3, 4]),
            ('段0+段1+段2+段3(チャンククラス)', [0, 1, 2, 3, 4, 5]), ('段0+段2のみ', [0, 4]), ('段0+段2+段3のみ', [0, 4, 5])]
    if nn_ep > 0:
        sets += [('段0+ニューラル', [0, 6]), ('段0+段1+ニューラル', [0, 1, 2, 3, 6]), ('全部(段0〜3)+ニューラル', [0, 1, 2, 3, 4, 5, 6])]
    prev = None
    for label, idx in sets:
        w = em(pv[idx]); mix = (w[:, None] * pt[idx]).sum(0); b = -np.log2(mix).mean()
        log('  %-32s %.3f  改善 %5.2f%%  重み=%s' % (label, b, 100 * (1 - b / base), np.round(w, 2).tolist()))
    log('(参考: ニューラル混合の改善は、N=100万で8.5パーセント)  [%.0fs]' % (time.time() - t0))


if __name__ == '__main__':
    main()
