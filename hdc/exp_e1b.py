"""
E1 (圧縮機械の最初の試験): 「穴あきの型」を自動で選ぶと、n-gramより予測が良くなるか。
  型 = 直前6語の各位置を  E(その語そのもの) / C(語のクラス) / W(何でもよい=穴) のどれかにした文脈。
  各型は「その文脈の次の語の数え上げ表」を持つ。基準 = 1,2,3語前までの普通のn-gram(E, EE, EEE)の混合。
  候補の型を1つずつ試しに足し、検証データの bit/語が下がるものだけ残す(貪欲)。足す価値の基準は「短くなったか」。
  混合は、各表の確からしさ(その文脈を何回見たか)で重みを決める小さな門(ゲート)。
使い方: python3 exp_e1.py [訓練語数の上限=全部] [貪欲の回数=8]
出力: 基準のbit/語、足した型の一覧、足すごとのbit/語、表の項目数、テストでの最終値
"""
import os, sys, time, itertools, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_scale as es
from collections import Counter
es.CACHE = os.environ.get("LADDER_CORPUS", es.CACHE)

V = 5000; K = 6; NCLS = 64; DELTA = 0.75
GRAN = {'B': 16, 'C': 64, 'D': 512}
NB = 6
MUL = np.uint64(0x9E3779B97F4A7C15)


def log(s):
    print(s, flush=True)


def windows(docs, limit=None):
    X, y, tot = [], [], 0
    for a in docs:
        n = len(a)
        p = np.concatenate([np.zeros(K, np.int64), a])
        X.append(np.stack([p[j:j + n] for j in range(K)], 1)); y.append(a)
        tot += n
        if limit and tot >= limit:
            break
    X = np.concatenate(X); y = np.concatenate(y)
    return (X[:limit], y[:limit]) if limit else (X, y)


def make_classes(tr_docs, ncls=64):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds
    from scipy.cluster.vq import kmeans2
    t = np.concatenate(tr_docs)
    a = np.concatenate([t[:-1], t[1:], t[:-2], t[2:]]); b = np.concatenate([t[1:], t[:-1], t[2:], t[:-2]])
    M = coo_matrix((np.ones(len(a), np.float32), (a, b)), shape=(V, V)).tocsr()
    tot = M.sum(); ra = np.asarray(M.sum(1)).ravel(); cb = np.asarray(M.sum(0)).ravel() ** 0.75; cb = cb / cb.sum() * tot
    M = M.tocoo()
    pm = np.log(np.maximum(M.data * tot / (ra[M.row] * cb[M.col] + 1e-9), 1e-9))
    keep = pm > 0
    P = coo_matrix((pm[keep], (M.row[keep], M.col[keep])), shape=(V, V)).tocsr()
    U, S, _ = svds(P.astype(np.float64), k=40)
    E = U * np.sqrt(S); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    np.random.seed(0)
    _, lab = kmeans2(E, ncls, minit='++', seed=0)
    lab = lab.astype(np.int64); lab[0] = 0
    return lab


def pat_key(X, pat, cls):
    """pat: 長さKの文字列(左端が6語前、右端が直前)。E/C/W"""
    h = np.zeros(len(X), np.uint64)
    for j, c in enumerate(pat):
        if c == 'W':
            continue
        s = X[:, j] if c == 'E' else cls[c][X[:, j]]
        h = h * MUL + (s.astype(np.uint64) + np.uint64(1))
    return h


def bucket(T):
    b = np.zeros(len(T), np.int64)
    b[T >= 1] = 1; b[T >= 2] = 2; b[T >= 4] = 3; b[T >= 8] = 4; b[T >= 32] = 5
    return b


class Table:
    def __init__(s, key, y):
        uk, kid = np.unique(key, return_inverse=True)
        pair = kid.astype(np.int64) * V + y
        up, pc = np.unique(pair, return_counts=True)
        tot = np.bincount(kid, minlength=len(uk)).astype(np.float64)
        typ = np.bincount(up // V, minlength=len(uk)).astype(np.float64)
        s.uk, s.up, s.pc, s.tot, s.typ = uk, up, pc, tot, typ
        s.entries = len(up)

    def probs(s, key, y, pback):
        i = np.minimum(np.searchsorted(s.uk, key), len(s.uk) - 1)
        found = s.uk[i] == key
        pq = i.astype(np.int64) * V + y
        j = np.minimum(np.searchsorted(s.up, pq), len(s.up) - 1)
        c = np.where(found & (s.up[j] == pq), s.pc[j], 0)
        T = np.where(found, s.tot[i], 0.0); D = np.where(found, s.typ[i], 0.0)
        Ts = np.maximum(T, 1)
        p = np.where(found, np.maximum(c - DELTA, 0) / Ts + DELTA * D / Ts * pback, pback)
        return p, bucket(T)


def fit_gate(P, B, steps=150, a0=None):
    """P,B: (n,m)。門: 成分ごと×確からしさの段ごとの得点 → softmax で重み。bit/語を最小化"""
    P = torch.from_numpy(P).float(); B = torch.from_numpy(B).long()
    m = P.shape[1]
    a = torch.zeros(m, NB, requires_grad=True) if a0 is None else a0.clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=0.3)
    for _ in range(steps):
        w = torch.softmax(a[torch.arange(m)[None, :], B], 1)
        loss = -torch.log2((w * P).sum(1) + 1e-12).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return a.detach(), float(loss)


def gate_bits(a, P, B):
    P = torch.from_numpy(P).float(); B = torch.from_numpy(B).long(); m = P.shape[1]
    w = torch.softmax(a[torch.arange(m)[None, :], B], 1)
    return float(-torch.log2((w * P).sum(1) + 1e-12).mean())


def candidates():
    out = []
    for pat in itertools.product('EBCDW', repeat=K):
        pat = ''.join(pat)
        nz = sum(c != 'W' for c in pat)
        if pat[-1] == 'W' or nz < 2 or nz > 3:
            continue
        if pat.count('E') == 0 and nz == 2:
            continue
        out.append(pat)
    return out


def main():
    NTR = int(sys.argv[1]) if len(sys.argv) > 1 and int(sys.argv[1]) > 0 else None
    ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    torch.set_num_threads(os.cpu_count() or 2)
    corp = es.build_corpus()
    cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr, va, te = conv(corp[2]), conv(corp[1]), conv(corp[0])
    Xtr, ytr = windows(tr, NTR); Xva, yva = windows(va, 200000); Xte, yte = windows(te, 300000)
    log('訓練%d語 検証%d語 テスト%d語 (%.0fs)' % (len(Xtr), len(Xva), len(Xte), time.time() - t0))
    cls = {k: make_classes(tr, n) for k, n in GRAN.items()}
    log('語のクラス %d個 (%.0fs)  例: %s' % (NCLS, time.time() - t0, [[w for w, i in vocab.items() if cls['C'][i] == c][:5] for c in range(3)]))
    uni = np.bincount(ytr, minlength=V).astype(np.float64) + 1; uni /= uni.sum()
    Pb_va = uni[yva]; Pb_te = uni[yte]

    def comp(pat, Xa, ya, tab, pb):
        return tab.probs(pat_key(Xa, pat, cls), ya, pb)

    base_pats = ['WWWWWE', 'WWWWEE', 'WWWEEE']
    tabs = {}
    tabs_final = {}
    entries = {}

    def get_val(pat, Xt_sub_y):
        pass

    # 検証用に、候補ごとの確率を一度だけ計算(訓練は一部で表を作る)
    sub = min(len(Xtr), 2_000_000)
    Xs, ys = Xtr[:sub], ytr[:sub]
    pv = {}; bv = {}
    def add(pat):
        tab = Table(pat_key(Xs, pat, cls), ys)
        p, b = tab.probs(pat_key(Xva, pat, cls), yva, Pb_va)
        pv[pat], bv[pat] = p.astype(np.float32), b.astype(np.int8); entries[pat] = tab.entries
    for pat in base_pats:
        add(pat)
    uniP = Pb_va[:, None]; uniB = np.zeros((len(yva), 1), np.int8)
    def mats(pats):
        P = np.concatenate([uniP] + [pv[p][:, None] for p in pats], 1)
        B = np.concatenate([uniB] + [bv[p][:, None] for p in pats], 1)
        return P, B
    # 検証データを2つに分ける(選ぶ用 / 確かめる用)
    h = len(yva) // 2
    chosen = list(base_pats)
    P, B = mats(chosen)
    a, bits_sel = fit_gate(P[:h], B[:h])
    base_bits = gate_bits(a, P[h:], B[h:])
    log('基準(1〜3語前のn-gramの混合): 確かめ用bit/語=%.4f (%.0fs)' % (base_bits, time.time() - t0))
    cands = [p for p in candidates() if p not in chosen]
    log('候補の型 %d個 を計算中' % len(cands))
    for i, pat in enumerate(cands):
        add(pat)
    log('候補の確率を計算 (%.0fs)' % (time.time() - t0))
    hist = [dict(step=0, pat=None, bits=base_bits)]
    cur = bits_sel
    for r in range(ROUNDS):
        best, bp = cur, None
        rest = [p for p in cands if p not in chosen]
        q = []
        for pat in rest:
            P, B = mats(chosen + [pat])
            _, bs = fit_gate(P[:h], B[:h], steps=15)
            q.append((bs, pat))
        for _, pat in sorted(q)[:12]:
            P, B = mats(chosen + [pat])
            _, bs = fit_gate(P[:h], B[:h], steps=60)
            if bs < best - 1e-4:
                best, bp = bs, pat
        if bp is None:
            log('これ以上足しても良くならない'); break
        chosen.append(bp)
        P, B = mats(chosen)
        a, cur = fit_gate(P[:h], B[:h])
        vb = gate_bits(a, P[h:], B[h:])
        hist.append(dict(step=r + 1, pat=bp, bits=vb))
        log('足した型 %s: 確かめ用bit/語=%.4f (基準比 %+.2f%%) 項目数+%d (%.0fs)' % (bp, vb, 100 * (vb / base_bits - 1), entries[bp], time.time() - t0))
    # 最終: 全訓練で表を作り、検証全体で門を学び、テストで測る
    def final(pats):
        Ps_va, Bs_va, Ps_te, Bs_te = [Pb_va[:, None]], [np.zeros((len(yva), 1), np.int64)], [Pb_te[:, None]], [np.zeros((len(yte), 1), np.int64)]
        ent = 0
        for pat in pats:
            tab = Table(pat_key(Xtr, pat, cls), ytr); ent += tab.entries
            p, b = tab.probs(pat_key(Xva, pat, cls), yva, Pb_va); Ps_va.append(p[:, None]); Bs_va.append(b[:, None])
            p, b = tab.probs(pat_key(Xte, pat, cls), yte, Pb_te); Ps_te.append(p[:, None]); Bs_te.append(b[:, None])
        Pv, Bv = np.concatenate(Ps_va, 1), np.concatenate(Bs_va, 1)
        Pt, Bt = np.concatenate(Ps_te, 1), np.concatenate(Bs_te, 1)
        a, _ = fit_gate(Pv, Bv, steps=200)
        return gate_bits(a, Pt, Bt), ent
    tb_base, ent_base = final(base_pats)
    tb_new, ent_new = final(chosen)
    res = dict(test_bits_base=tb_base, test_bits_new=tb_new, gain_pct=100 * (1 - tb_new / tb_base),
               entries_base=int(ent_base), entries_new=int(ent_new), chosen=chosen[len(base_pats):], hist=hist)
    log('テスト bit/語: 基準 %.4f (項目%d) → 型を足した %.4f (項目%d)  改善 %.2f%%' % (tb_base, ent_base, tb_new, ent_new, res['gain_pct']))
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
