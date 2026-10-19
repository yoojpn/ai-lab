"""
段階1: 圧縮(尤度)だけを基準に、単語をクラスに自動で束ねる(交換アルゴリズム)。勾配なし。
  基準: クラス・バイグラム言語モデルの対数尤度(= 記述長)が上がる移動だけ採用する。
  使い方: クラスの4-gram(直前3語のクラス列 -> 次の語のクラス)×(クラス内での語の確率) を作り、
          単語4-gramと混ぜたときに、どれだけ良くなるかを測る(ニューラルは約8〜11%)。
  python3 exp_class.py <訓練語数N> <クラス数K> <スイープ数>
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time
import numpy as np
sys.path.insert(0, '/home/claude/hdc')
import exp_scale as es

V = es.V; CTX = es.CTX; DELTA = es.DELTA
f = lambda x: np.where(x > 0, x * np.log(np.maximum(x, 1e-300)), 0.0)


class NGv:
    """exp_scale.NG と同じ補間つき絶対値割引。語彙サイズ v を可変にしたもの"""
    def __init__(s, X, y, v):
        s.v = v; s.t = []
        for k in range(CTX + 1):
            ck = s.code(X, k)
            up, pc = np.unique(ck * v + y, return_counts=True)
            uc, tot = np.unique(ck, return_counts=True)
            _, typ = np.unique(up // v, return_counts=True)
            s.t.append((uc, tot, typ, up, pc))

    def code(s, X, k):
        c = np.zeros(len(X), np.int64)
        for j in range(k):
            c += X[:, CTX - 1 - j].astype(np.int64) * (s.v ** j)
        return c

    def probs(s, X, y):
        v = s.v
        p = np.full(len(y), 1.0 / v); alive = np.ones(len(y), bool)
        for k in range(CTX + 1):
            uc, tot, typ, up, pc = s.t[k]
            cq = s.code(X, k)
            i = np.minimum(np.searchsorted(uc, cq), len(uc) - 1)
            found = alive & (uc[i] == cq)
            pq = cq * v + y
            j = np.minimum(np.searchsorted(up, pq), len(up) - 1)
            c = np.where(up[j] == pq, pc[j], 0)
            p = np.where(found, np.maximum(c - DELTA, 0) / tot[i] + DELTA * typ[i] / tot[i] * p, p)
            alive = found
        return p


def exchange(tok_mods, K, sweeps, log):
    prev = np.concatenate([a[:-1] for a in tok_mods]); nxt = np.concatenate([a[1:] for a in tok_mods])
    wc = np.bincount(np.concatenate(tok_mods), minlength=V).astype(np.float64)
    u, cnt = np.unique(prev * V + nxt, return_counts=True)
    pi, ni = u // V, u % V
    selfc = np.zeros(V); m_self = pi == ni; selfc[pi[m_self]] = cnt[m_self]
    pi2, ni2, cnt2 = pi[~m_self], ni[~m_self], cnt[~m_self].astype(np.float64)
    o_out = np.argsort(pi2, kind='stable'); o_in = np.argsort(ni2, kind='stable')
    out_off = np.searchsorted(pi2[o_out], np.arange(V + 1)); in_off = np.searchsorted(ni2[o_in], np.arange(V + 1))
    rank = np.argsort(-wc, kind='stable')
    cls = np.zeros(V, np.int64); cls[rank] = np.arange(V) % K
    N = np.zeros((K, K)); np.add.at(N, (cls[pi], cls[ni]), cnt)
    left = np.zeros((V, K)); right = np.zeros((V, K))
    np.add.at(right, (pi2, cls[ni2]), cnt2); np.add.at(left, (ni2, cls[pi2]), cnt2)
    R = N.sum(1); C = N.sum(0)
    T = f(N).sum() - f(R).sum() - f(C).sum()
    log('初期 T=%.0f' % T)
    order = [w for w in rank if wc[w] > 0]
    M = 12

    def delta(w, a, b):
        dr, dl, s = right[w], left[w], selfc[w]
        rb = N[[a, b], :].copy(); ra = rb.copy()
        ra[0] -= dr; ra[1] += dr
        ra[0, a] -= dl[a]; ra[0, b] += dl[a]; ra[1, a] -= dl[b]; ra[1, b] += dl[b]
        ra[0, a] -= s; ra[1, b] += s
        mask = np.ones(K, bool); mask[[a, b]] = False
        ca_b = N[:, [a, b]][mask]; ca_a = ca_b.copy()
        ca_a[:, 0] -= dl[mask]; ca_a[:, 1] += dl[mask]
        Ra, Rb = ra[0].sum(), ra[1].sum()
        Ca = ca_a[:, 0].sum() + ra[0, a] + ra[1, a]; Cb = ca_a[:, 1].sum() + ra[0, b] + ra[1, b]
        after = f(ra).sum() + f(ca_a).sum() - f(Ra) - f(Rb) - f(Ca) - f(Cb)
        before = f(rb).sum() + f(ca_b).sum() - f(R[a]) - f(R[b]) - f(C[a]) - f(C[b])
        return after - before

    for sw in range(1, sweeps + 1):
        moved = 0; t0 = time.time()
        for w in order:
            a = cls[w]
            prof = np.concatenate([left[w], right[w]])
            nrm = np.linalg.norm(prof)
            if nrm == 0:
                continue
            P = np.concatenate([N.T, N], axis=1)          # クラスc の(前に来る側, 後に来る側)の分布
            sim = (P @ prof) / (np.linalg.norm(P, axis=1) * nrm + 1e-12)
            cand = [b for b in np.argsort(-sim)[:M] if b != a]
            best_b, best_d = None, 1e-9
            for b in cand:
                d = delta(w, a, b)
                if d > best_d:
                    best_b, best_d = b, d
            if best_b is None:
                continue
            b = best_b
            dr, dl, s = right[w].copy(), left[w].copy(), selfc[w]
            N[a, :] -= dr; N[b, :] += dr; N[:, a] -= dl; N[:, b] += dl; N[a, a] -= s; N[b, b] += s
            R = N.sum(1); C = N.sum(0)
            cls[w] = b
            lo, hi = out_off[w], out_off[w + 1]
            us = ni2[o_out[lo:hi]]; cs = cnt2[o_out[lo:hi]]
            np.add.at(left, (us, np.full(len(us), a)), -cs); np.add.at(left, (us, np.full(len(us), b)), cs)
            lo, hi = in_off[w], in_off[w + 1]
            us = pi2[o_in[lo:hi]]; cs = cnt2[o_in[lo:hi]]
            np.add.at(right, (us, np.full(len(us), a)), -cs); np.add.at(right, (us, np.full(len(us), b)), cs)
            moved += 1
        T = f(N).sum() - f(R).sum() - f(C).sum()
        log('スイープ%d: 移動した語=%d T=%.0f (尤度が上がるほど圧縮が進んだ) [%.0fs]' % (sw, moved, T, time.time() - t0))
    return cls, wc


def main():
    N = int(sys.argv[1]); K = int(sys.argv[2]); sweeps = int(sys.argv[3])
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    corp = es.build_corpus()
    from collections import Counter
    cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr_i, va_i, te_i = conv(corp[2]), conv(corp[1]), conv(corp[0])
    # 訓練語N個ぶんのモジュール(exp_scale.windows と同じ取り方)
    mods, tot = [], 0
    for a in tr_i:
        if len(a) <= CTX:
            continue
        mods.append(a); tot += len(a) - CTX
        if tot >= N:
            break
    Xtr, ytr = es.windows(tr_i, N); Xva, yva = es.windows(va_i, es.N_VAL); Xte, yte = es.windows(te_i, es.N_TEST)
    ng = es.NG(Xtr, ytr)
    pva_w, _ = ng.probs(Xva, yva); pte_w, _ = ng.probs(Xte, yte)
    log('訓練%d 検証%d テスト%d | 単語4-gram bit/語: 検証%.3f テスト%.3f (準備%.0fs)' % (
        len(Xtr), len(Xva), len(Xte), -np.log2(pva_w).mean(), -np.log2(pte_w).mean(), time.time() - t0))
    cls, wc = exchange(mods, K, sweeps, log)
    # 例: 大きいクラスの中身
    for c in np.argsort(-np.bincount(cls, weights=wc, minlength=K))[:0]:
        pass
    inv = {i: w for w, i in vocab.items()}; inv[0] = '<unk>'
    shown = 0
    for c in range(K):
        members = [w for w in np.argsort(-wc) if cls[w] == c and wc[w] > 0][:8]
        if len(members) >= 4 and shown < 12:
            log('  クラス%3d: %s' % (c, ' '.join(inv.get(int(w), '?') for w in members))); shown += 1
    clsc = np.bincount(cls, weights=wc, minlength=K)
    word_in_cls = wc / np.maximum(clsc[cls], 1)
    word_in_cls = np.where(wc > 0, word_in_cls, 1e-9)
    ngc = NGv(cls[Xtr], cls[ytr], K)
    def clsprob(X, y):
        return ngc.probs(cls[X], cls[y]) * word_in_cls[y]
    pva_c, pte_c = clsprob(Xva, yva), clsprob(Xte, yte)
    LAMS = [0, .05, .1, .15, .2, .3, .4, .5]
    lam = max(LAMS, key=lambda l: np.log2((1 - l) * pva_w + l * pva_c).sum())
    b_w = -np.log2(pte_w); b_c = -np.log2(pte_c); b_m = -np.log2((1 - lam) * pte_w + lam * pte_c)
    log('\n[N=%d K=%d] テスト bit/語: 単語4-gram=%.3f クラス4-gram単独=%.3f 混合=%.3f (λ=%.2f) → 混合の改善 %.2f%%  [%.0fs]'
        % (len(Xtr), K, b_w.mean(), b_c.mean(), b_m.mean(), lam, 100 * (1 - b_m.mean() / b_w.mean()), time.time() - t0))
    log('(参考: 同条件のニューラル混合の改善は N=100万で8.5%、25万で10.9%、300万で7.9%)')


if __name__ == '__main__':
    main()
