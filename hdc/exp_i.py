"""
I群: 勾配ゼロで「教えていないこと」ができるようにする工夫を、部品ごとに足して測る。
  文献: infini-gram (arXiv 2401.17377: 最長一致の文脈まで遡る「無限長n-gram」), n-gram heads / induction (arXiv 2401.12973, 2209.11895)。
  部品(すべて数える/引くだけ):
    base   : 直前3語のn-gram + 1語飛ばし2本 (絶対割引0.9)
    main   : 本線を order 語まで伸ばす(ハッシュ。infini-gram風に、一致する最長の文脈まで遡る)
    cls    : 語を K 個のグループに分け(PPMI埋め込みのk-means)、グループで文脈を数える → 表にない並びでも答えられる
    mixcls : 直前1語は実際の語、その前2語はグループ
    ngh    : 文書内の n-gram head (直前1〜3語と同じ並びが前にあれば、その次の語を写す。窓の外も見る)
    far    : 窓内のコピー/キャッシュ + 300語の遠いキャッシュ
    bucket : 混ぜる重みの分け方 none|mk(当たった次数)|conf(次数と件数)|ngh(写しの強さ)
    prune  : 件数がこの値より小さい並びを捨てる(記憶を小さく)
    nn_h/nn_n : 極小NN(勾配あり)を足す
  出力: 全体のbit/語、「3語文脈が表にない窓」のbit/語、「正解語が前の文書内に出ていた窓」のbit/語、表の件数、演算回数
使い方: [LADDER_CORPUS=...] python3 exp_i.py '<json>'
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab, items
from lab import V, HIST, bits, log
import exp_scale as es, exp_cards as XC, exp_far as XF, exp_ladder as L

es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)
MUL = np.uint64(0x9E3779B97F4A7C15)


def code(X, spec, cls):
    c = np.zeros(len(X), np.uint64)
    with np.errstate(over='ignore'):
        for t, p in spec:
            v = X[:, HIST - p]
            if t == 'c': v = cls[v]
            c = c * MUL + (v.astype(np.uint64) + np.uint64(1))
    return c.view(np.int64)


class HChain:
    """絶対割引つきバックオフ。levels = 文脈の取り方の列。各要素は [('r'|'c', p), ...] (r=実際の語, c=グループ, p=直前から何語前か)"""
    def __init__(s, X, y, levels, cls, delta=0.9, prune=0):
        s.levels, s.cls, s.delta, s.t, s.entries = levels, cls, delta, [], 0
        for spec in levels:
            ck = code(X, spec, cls)
            uc, inv, tot = np.unique(ck, return_inverse=True, return_counts=True)
            up, pc = np.unique(inv.astype(np.int64) * V + y, return_counts=True)
            del inv
            typ = np.bincount(up // V, minlength=len(uc))      # 捨てる前の種類数(退避の重みに使う)
            if prune > 1 and len(spec) >= 2:
                keep = pc >= prune
                up, pc = up[keep], pc[keep]
            s.t.append((uc, tot, typ, up, pc)); s.entries += len(up) + len(uc)

    def probs(s, X, y):
        p = np.full(len(y), 1.0 / V); alive = np.ones(len(y), bool); mk = np.zeros(len(y), int); tt = np.zeros(len(y))
        for k, spec in enumerate(s.levels):
            uc, tot, typ, up, pc = s.t[k]
            cq = code(X, spec, s.cls)
            i = np.minimum(np.searchsorted(uc, cq), len(uc) - 1)
            found = alive & (uc[i] == cq)
            pq = i.astype(np.int64) * V + y
            j = np.minimum(np.searchsorted(up, pq), len(up) - 1)
            c = np.where(up[j] == pq, pc[j], 0)
            newp = np.maximum(c - s.delta, 0) / tot[i] + s.delta * typ[i] / tot[i] * p
            p = np.where(found, newp, p); mk = np.where(found, k, mk); tt = np.where(found, tot[i], tt); alive = found
        return p, mk, tt

    def ops(s, N):
        return float(len(s.levels) * N * np.log2(N))


def ngh(docs, mid, pos, y, pu):
    """文書内のn-gram head: 直前k語(k=3,2,1)と同じ並びが前にあれば、その次の語を予測。窓の外も見る"""
    n = len(y); P = pu[y].copy(); S = np.zeros(n, int)
    for q in range(n):
        a = docs[mid[q]]; p = int(pos[q])
        for k in (3, 2, 1):
            if p - k < 0 or a[p - 1] == 0: break
            m = a[:p - 1] == a[p - 1]
            idx = np.nonzero(m)[0]
            idx = idx[idx >= k - 1]
            ok = np.ones(len(idx), bool)
            for d in range(1, k):
                ok &= a[idx - d] == a[p - 1 - d]
            idx = idx[ok]
            if len(idx):
                pred = a[idx[-1] + 1]
                P[q] = 0.9 * (pred == y[q]) + 0.1 * pu[y[q]]; S[q] = k
                break
    return P, S


def ngh2(docs, mid, pos, y, pu, kmax=5):
    """文書内のn-gram head 第2版: 直前kmax〜1語と同じ並びが前に何度かあれば、その次の語を多数決(窓の外も見る)"""
    n = len(y); P = pu[y].copy(); S = np.zeros(n, int)
    for q in range(n):
        a = docs[mid[q]]; p = int(pos[q])
        for k in range(kmax, 0, -1):
            if p - k < 0 or a[p - 1] == 0: continue
            idx = np.nonzero(a[:p - 1] == a[p - 1])[0]
            idx = idx[idx >= k - 1]
            ok = np.ones(len(idx), bool)
            for d in range(1, k):
                ok &= a[idx - d] == a[p - 1 - d]
            idx = idx[ok]
            if len(idx):
                nxt = a[idx + 1]
                P[q] = 0.9 * (nxt == y[q]).mean() + 0.1 * pu[y[q]]; S[q] = k
                break
    return P, S


def far_repeat(docs, mid, pos, y):
    out = np.zeros(len(y), bool)
    for q in range(len(y)):
        a = docs[mid[q]]; p = int(pos[q])
        out[q] = y[q] != 0 and bool((a[:max(0, p - HIST)] == y[q]).any())
    return out


def mix(Pv, Pt, bv, bt, minn=300):
    wg = L.em(Pv)
    if bv is None:
        return (wg[:, None] * Pt).sum(0), wg
    out = np.zeros(Pt.shape[1])
    for b in np.unique(bt):
        mv = bv == b; mt = bt == b
        w = L.em(Pv[:, mv]) if mv.sum() >= minn else wg
        out[mt] = (w[:, None] * Pt[:, mt]).sum(0)
    return out, wg


def run(cfg):
    t0 = time.time(); torch.set_num_threads(os.cpu_count() or 2)
    N = cfg['N']; nq = cfg.get('nq', 20000); delta = cfg.get('delta', 0.9)
    ctx = lab.load(N); N = len(ctx.Xtr)
    res = dict(cfg=cfg, N=N); ops = {}
    pu = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1.0; pu /= pu.sum()
    sl = slice(0, nq); nv = min(nq, len(ctx.yva))
    Xva, yva, Xte, yte = ctx.Xva[:nv], ctx.yva[:nv], ctx.Xte[sl], ctx.yte[sl]
    nt = len(yte)
    # グループ(クラス)
    K = max(cfg.get('cls', 0), cfg.get('mixcls', 0)); cls = np.zeros(V, np.int64)
    E0 = None
    if K or cfg.get('nn_h', 0):
        dsv = lab.DE if cfg.get('nn_h', 0) else cfg.get('svd_dim', lab.DE)
        E0 = items.ppmi_embeddings(ctx, dsv); ops['svd'] = float(2 * V * V * dsv * 7 + 22 * N + 10 * V * V)
    if K:
        from scipy.cluster.vq import kmeans2
        En = E0 / (np.linalg.norm(E0, axis=1, keepdims=True) + 1e-9)
        _, lab_ = kmeans2(En.astype(np.float64), K, minit='++', seed=1)
        cls = lab_.astype(np.int64); ops['kmeans'] = float(10 * V * K * E0.shape[1] * 2)
    R = lambda *ps: [('r', p) for p in ps]
    C = lambda *ps: [('c', p) for p in ps]
    order = cfg.get('main', 3)
    specs = {'main': [[]] + [R(*range(1, k + 1)) for k in range(1, order + 1)],
             }
    if cfg.get('skipA', 1): specs['skipA'] = [[], R(2), R(2, 3), R(2, 3, 4)]
    if cfg.get('skipB', 1): specs['skipB'] = [[], R(1), R(1, 3), R(1, 3, 4)]
    if cfg.get('cls'): specs['cls'] = [[], C(1), C(1, 2), C(1, 2, 3)]
    if cfg.get('mixcls'): specs['mixcls'] = [[], R(1), R(1) + C(2), R(1) + C(2, 3)]
    chains, names = {}, []
    for nm, lv in specs.items():
        chains[nm] = HChain(ctx.Xtr, ctx.ytr, lv, cls, delta, cfg.get('prune', 0)); ops['count_' + nm] = chains[nm].ops(N)
        log('数えた %s 表%d件 [%.0fs]' % (nm, chains[nm].entries, time.time() - t0))
    res['entries'] = {k: int(c.entries) for k, c in chains.items()}; res['entries_total'] = int(sum(res['entries'].values()))
    Pv, Pt, names = [], [], []
    mkv = mkt = ttv = ttt = None
    for nm, ch in chains.items():
        pv, mv_, tv_ = ch.probs(Xva, yva); pt_, mt_, tt_ = ch.probs(Xte, yte)
        Pv.append(pv); Pt.append(pt_); names.append(nm)
        if nm == 'main': mkv, mkt, ttv, ttt = mv_, mt_, tv_, tt_
    # 文書内
    Sv = St = None
    if cfg.get('ngh'):
        pv, Sv = ngh(ctx.va_i, ctx.mva[:nv], ctx.pva[:nv], yva, pu); pt_, St = ngh(ctx.te_i, ctx.mte[sl], ctx.pte[sl], yte, pu)
        Pv.append(pv); Pt.append(pt_); names.append('ngh')
    if cfg.get('ngh2'):
        pv, S2v = ngh2(ctx.va_i, ctx.mva[:nv], ctx.pva[:nv], yva, pu); pt_, S2t = ngh2(ctx.te_i, ctx.mte[sl], ctx.pte[sl], yte, pu)
        Pv.append(pv); Pt.append(pt_); names.append('ngh2')
    if cfg.get('far'):
        cpv, _ = XC.copy_probs(Xva, yva, pu); cpt, _ = XC.copy_probs(Xte, yte, pu)
        Pv += [cpv, XC.cache_probs(Xva, yva, pu), XF.farcache_probs(ctx.va_i, ctx.mva[:nv], ctx.pva[:nv], yva, cfg.get('farW', 300), pu)]
        Pt += [cpt, XC.cache_probs(Xte, yte, pu), XF.farcache_probs(ctx.te_i, ctx.mte[sl], ctx.pte[sl], yte, cfg.get('farW', 300), pu)]
        names += ['copy', 'cache', 'farcache']
    # 極小NN
    net = None
    if cfg.get('nn_h'):
        idx = np.random.default_rng(0).choice(len(ctx.Xtr), min(cfg.get('nn_n', 1000000), len(ctx.Xtr)), replace=False)
        net = lab.train(ctx, cfg['nn_h'], 1.0, E0=E0, idx=idx); net.eval()
        def nnp(X, y):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 4096):
                    pr = torch.softmax(net(torch.from_numpy(X[i:i + 4096])), 1)
                    out.append(pr[torch.arange(len(pr)), torch.from_numpy(y[i:i + 4096])].numpy())
            return np.concatenate(out).astype(np.float64)
        Pv.append(nnp(Xva, yva)); Pt.append(nnp(Xte, yte)); names.append('nn')
        ops['nn_train'] = float(6 * 5576 * cfg['nn_h'] * len(idx))
    # Transformer(勾配あり)を足す
    if cfg.get('tf_d'):
        import exp_t
        tnet, tsec = exp_t.train_tf(ctx, cfg['tf_d'], cfg['tf_nl'], cfg.get('tf_epochs', 1))
        def tfp(X, y):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 4096):
                    pr = torch.softmax(tnet(torch.from_numpy(X[i:i + 4096])), 1)
                    out.append(pr[torch.arange(len(pr)), torch.from_numpy(y[i:i + 4096])].numpy())
            return np.concatenate(out).astype(np.float64)
        ptv, ptt = tfp(Xva, yva), tfp(Xte, yte)
        res['tf_alone'] = bits(ptt); res['tf_train_sec'] = tsec
        Pv.append(ptv); Pt.append(ptt); names.append('tf')
        log('Transformer単独 %.4f (学習%.0fs)' % (res['tf_alone'], tsec))
    # 混ぜる
    bk = cfg.get('bucket', 'none')
    bv = bt = None
    if bk == 'mk': bv, bt = mkv, mkt
    elif bk == 'conf':
        cb = lambda m, t: m * 8 + np.minimum(np.log2(t + 1).astype(int), 7)
        bv, bt = cb(mkv, ttv), cb(mkt, ttt)
    elif bk == 'ngh' and Sv is not None: bv, bt = Sv * 16 + mkv, St * 16 + mkt
    Pv, Pt = np.array(Pv), np.array(Pt)
    pm, w = mix(Pv, Pt, bv, bt)
    res['bits'] = bits(pm); res['names'] = names; res['w'] = [float(x) for x in w]
    unseen = mkt < min(3, order)
    res['frac_unseen'] = float(unseen.mean()); res['bits_unseen'] = bits(pm[unseen]) if unseen.sum() > 50 else None
    fr = far_repeat(ctx.te_i, ctx.mte[sl], ctx.pte[sl], yte)
    res['frac_far'] = float(fr.mean()); res['bits_far'] = bits(pm[fr]) if fr.sum() > 50 else None
    res['bits_notfar'] = bits(pm[~fr])
    res['ops'] = ops; res['ops_total'] = float(sum(ops.values())); res['sec'] = time.time() - t0
    log('bit/語 %.4f | 表にない3語文脈(%.0f%%) %s | 文書内で既出の語(%.0f%%) %s | 表%d件 | 演算%.2e [%.0fs]' % (
        res['bits'], 100 * res['frac_unseen'], res['bits_unseen'] and round(res['bits_unseen'], 3), 100 * res['frac_far'],
        res['bits_far'] and round(res['bits_far'], 3), res['entries_total'], res['ops_total'], time.time() - t0))
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
