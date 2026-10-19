"""
実験B: 「空欄つきのカード」は、数え上げのはしご(+小ニューラル)の上にさらに圧縮を足せるか。

はしご(基準): 単語4-gram + クラス + チャンク + チャンククラス (+ 小ニューラル)  ← 前回 5.96% / 8.92% 改善
カード(新規): 次の単語の予測に使う文脈の「型」を、データから選ぶ。
   直前4語の各位置を { 単語そのもの / その単語のクラス(=空欄の取り得る範囲) / 無視(空欄) } のどれかにした型
   例: 「語 ＿ クラス」= 2つ前は単語そのもの、1つ前は何でもよい…ではなく位置ごとに指定する型。
   型の候補は 3^4-1 = 80 通り。訓練データで数え、検証データでの圧縮の得(ビット数の減り)が最大の型を
   1つずつ足していく(足して得が小さければ止める)。= 圧縮を基準にした規則(カード)選び。
 + 文中の繰り返し規則: 「直前の語が以前に出た箇所の次の語」をそのまま写す規則(=変数つきの規則: A B … A → B)
 + 近い所の語の再出現(キャッシュ)
混合の重みは、状況(繰り返し候補の有無 × 4-gramが見つけた最長文脈の次数)ごとに検証データで決める。
基準側にも同じ状況別の重みを使う(=得が混ぜ方の工夫ではなく、カード自体から来ているかを見分けるため)。
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time, itertools
sys.path.insert(0, '/home/claude/hdc')
import numpy as np
from collections import Counter
import exp_ladder as L
import exp_scale as es
import exp_class as ec

V = es.V; HIST = L.HIST; DELTA = es.DELTA
log = L.log


def bucket_em(P, bucket, nb, iters=60, prior=150):
    """P: (成分数, n)。状況(bucket)ごとにEMで重みを決める。小さい状況は全体の重みに寄せる"""
    wg = L.em(P, iters)
    W = np.tile(wg, (nb, 1))
    for b in range(nb):
        m = bucket == b
        n = int(m.sum())
        if n < 50:
            continue
        w = wg.copy()
        for _ in range(iters):
            mix = (w[:, None] * P[:, m]).sum(0)
            w = (w[:, None] * P[:, m] / mix).mean(1)
        W[b] = (n * w + prior * wg) / (n + prior)
    return W


def mix_bits(P, bucket, W):
    return -np.log2((W[bucket].T * P).sum(0))


class Card:
    """型 pat(長さ4。位置4..1 = 4語前..直前。'-'は無視 'w'は単語 'c'はクラス)で次の語を当てる"""
    def __init__(s, pat, Xtr, ytr, cls, pu):
        s.pat, s.cls, s.pu = pat, cls, pu
        c = s.code(Xtr)
        up, pc = np.unique(c * V + ytr, return_counts=True)
        uc, tot = np.unique(c, return_counts=True)
        _, typ = np.unique(up // V, return_counts=True)
        s.t = (uc, tot, typ, up, pc)

    def code(s, X):
        c = np.zeros(len(X), np.int64)
        k = 0
        for i, t in enumerate(s.pat):
            if t == '-':
                continue
            col = X[:, HIST - 4 + i]
            sym = col if t == 'w' else s.cls[col]
            c += sym.astype(np.int64) * (V ** k)
            k += 1
        return c

    def probs(s, X, y):
        uc, tot, typ, up, pc = s.t
        cq = s.code(X)
        i = np.minimum(np.searchsorted(uc, cq), len(uc) - 1)
        found = uc[i] == cq
        pq = cq * V + y
        j = np.minimum(np.searchsorted(up, pq), len(up) - 1)
        c = np.where(up[j] == pq, pc[j], 0)
        pu = s.pu[y]
        return np.where(found, np.maximum(c - DELTA, 0) / tot[i] + DELTA * typ[i] / tot[i] * pu, pu)

    def name(s):
        nm = {'-': '＿', 'w': '語', 'c': 'ｸﾗｽ'}
        return ' '.join(nm[t] for t in s.pat) + ' → 次'


def copy_feats(X):
    """文中の繰り返し: 直前の語が過去(窓内)に出た箇所の次の語。強さ 0=なし 1=1語一致 2=2語一致"""
    n = len(X)
    last = X[:, HIST - 1]
    pred = np.zeros(n, np.int64)
    strength = np.zeros(n, np.int64)
    done = last == 0
    for j in range(HIST - 2, -1, -1):
        m = (~done) & (X[:, j] == last)
        pred[m] = X[m, j + 1]
        two = m & (j >= 1) & (X[:, max(j - 1, 0)] == X[:, HIST - 2])
        strength[m] = np.where(two[m], 2, 1)
        done |= m
    return pred, strength


def copy_probs(X, y, pu):
    pred, st = copy_feats(X)
    p = np.where(st > 0, 0.85 * (y == pred) + 0.15 * pu[y], pu[y])
    return p, st


def cache_probs(X, y, pu):
    cnt = (X == y[:, None]).sum(1)
    cnt = np.where(y == 0, 0, cnt)
    return 0.6 * cnt / HIST + 0.4 * pu[y]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    K, M, K2 = 200, 1500, 200
    HAVE_NN = (N == 2_000_000)
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
    Xtr, ytr = L.lwindows(tr_i, N); Xva, yva = L.lwindows(va_i, es.N_VAL); Xte, yte = L.lwindows(te_i, es.N_TEST)
    log('訓練%d 検証%d テスト%d [%.0fs]' % (len(Xtr), len(Xva), len(Xte), time.time() - t0))
    w3 = lambda X: X[:, HIST - 3:]
    pu = np.bincount(ytr, minlength=V).astype(np.float64) + 1.0
    pu /= pu.sum()
    # ---- 基準のはしご(前回と同じ)
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
    if HAVE_NN:
        stem = os.environ.get('LADDER_CORPUS', es.CACHE).replace('.pkl', '') + '_nn1.0.npz'
        z = np.load(stem)
        nn_va, nn_te = z['gv'], z['gt']
    base_bits = -np.log2(pA_te).mean()
    log('単語4-gram テスト %.3f bit/語' % base_bits)

    # ---- 状況の分け方: (繰り返し候補の強さ 0/1/2) × (4-gramが見つけた最長文脈 0-1 / 2 / 3)
    cp_va, st_va = copy_probs(Xva, yva, pu); cp_te, st_te = copy_probs(Xte, yte, pu)
    mkb = lambda mk: np.where(mk <= 1, 0, mk - 1)
    bk_va = st_va * 3 + mkb(mk_va); bk_te = st_te * 3 + mkb(mk_te)
    NB = 9
    ca_va = cache_probs(Xva, yva, pu); ca_te = cache_probs(Xte, yte, pu)

    def evaluate(pv, pt, label):
        Pv, Pt = np.array(pv), np.array(pt)
        W = bucket_em(Pv, bk_va, NB)
        b = mix_bits(Pt, bk_te, W).mean()
        log('  %-52s %.3f  4-gram比の改善 %5.2f%%' % (label, b, 100 * (1 - b / base_bits)))
        return b

    log('\n=== 基準(同じ「状況別の重み」を使う。得が混ぜ方でなくカードに由来するかを見分けるため) ===')
    b_lad = evaluate(base_va, base_te, 'はしご(段0〜3)')
    if HAVE_NN:
        b_lad_nn = evaluate(base_va + [nn_va], base_te + [nn_te], 'はしご + 小ニューラル')

    log('\n=== 繰り返し規則・キャッシュだけを足す ===')
    evaluate(base_va + [cp_va], base_te + [cp_te], 'はしご + 繰り返し規則')
    evaluate(base_va + [cp_va, ca_va], base_te + [cp_te, ca_te], 'はしご + 繰り返し規則 + キャッシュ')
    if HAVE_NN:
        evaluate(base_va + [nn_va, cp_va, ca_va], base_te + [nn_te, cp_te, ca_te], 'はしご + ニューラル + 繰り返し + キャッシュ')

    # ---- 型(カード)の候補を作り、検証データでの圧縮の得で貪欲に選ぶ
    pats = [p for p in itertools.product('-wc', repeat=4) if any(t != '-' for t in p)]
    log('\n型の候補 %d 通りを訓練データで数える…' % len(pats))
    cards = {}
    for p in pats:
        c = Card(p, Xtr, ytr, cls, pu)
        cards[p] = (c.name(), c.probs(Xva, yva), c.probs(Xte, yte))
        del c
    log('  数え終わり [%.0fs]' % (time.time() - t0))

    def greedy(base_v, base_t, extra_v, extra_t, tag, max_cards=8, min_gain=0.0005):
        cur_v, cur_t = list(base_v) + list(extra_v), list(base_t) + list(extra_t)
        W = bucket_em(np.array(cur_v), bk_va, NB, iters=25)
        cur = mix_bits(np.array(cur_v), bk_va, W).mean()
        chosen = []
        for step in range(max_cards):
            best = None
            for p, (c, pv_, pt_) in cards.items():
                if p in chosen:
                    continue
                Pv = np.array(cur_v + [pv_])
                Wn = bucket_em(Pv, bk_va, NB, iters=12, prior=150)
                bits = mix_bits(Pv, bk_va, Wn).mean()
                if best is None or bits < best[0]:
                    best = (bits, p)
            gain = (cur - best[0]) / cur
            if gain < min_gain:
                log('  [%s] 次の候補の得 %.2f%% < 閾値。ここで止める' % (tag, 100 * gain))
                break
            chosen.append(best[1])
            cur_v.append(cards[best[1]][1]); cur_t.append(cards[best[1]][2]); cur = best[0]
            log('  [%s] 選んだ型 %d: %-14s 検証での得 %.2f%%' % (tag, len(chosen), cards[best[1]][0], 100 * gain))
        return cur_v, cur_t, chosen

    log('\n=== 型(カード)を圧縮の基準で選ぶ ===')
    log('--- はしご + 繰り返し + キャッシュ の上に積む ---')
    v1, t1, ch1 = greedy(base_v=base_va, base_t=base_te, extra_v=[cp_va, ca_va], extra_t=[cp_te, ca_te], tag='はしご系')
    b1 = evaluate(v1, t1, 'はしご + 繰り返し + キャッシュ + 選んだ型%d個' % len(ch1))
    if HAVE_NN:
        log('--- 小ニューラルも入れた上に積む ---')
        v2, t2, ch2 = greedy(base_v=base_va + [nn_va], base_t=base_te + [nn_te], extra_v=[cp_va, ca_va], extra_t=[cp_te, ca_te], tag='ニューラル系')
        b2 = evaluate(v2, t2, 'はしご + ニューラル + 繰り返し + キャッシュ + 型%d個' % len(ch2))
    log('\n要約(テスト、4-gram基準 %.3f)' % base_bits)
    log('  はしご                     改善 %5.2f%%' % (100 * (1 - b_lad / base_bits)))
    log('  はしご + 全カード          改善 %5.2f%%' % (100 * (1 - b1 / base_bits)))
    if HAVE_NN:
        log('  はしご + ニューラル        改善 %5.2f%%' % (100 * (1 - b_lad_nn / base_bits)))
        log('  はしご + ニューラル + 全カード 改善 %5.2f%%' % (100 * (1 - b2 / base_bits)))
    log('[N=%d  %.0fs]' % (N, time.time() - t0))


if __name__ == '__main__':
    main()
