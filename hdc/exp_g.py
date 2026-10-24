"""
G群: 勾配ゼロのモデルを、同じ演算回数のまま良くする工夫を並べて測る。
  構成部品(すべて「数える/書く」だけ。勾配なし):
    main   : 本線のn-gram。文脈の長さ 3 または 4 (4は5-gram)
    skip   : 1語飛ばしの文脈を数える線を足す (w-2,w-3.. と w-1,w-3..)
    bucket : 混ぜる重みを、本線が当たった次数ごとに別々に決める(EM)
    key    : 索引の鍵。ppmi=共起のSVD / rand=ランダム(SVD不要) / none=索引なし
    de     : 鍵の1語あたりの次元
    nn_h,nn_n : 極小NN(勾配あり)を足す場合の幅と読む語数 (0なら無し)
    delta  : 絶対割引の値
  出力: bit/語(20000窓)と、学習の演算回数の見積り(内訳つき)
使い方: LADDER_CORPUS=... python3 exp_g.py '<json>'   例 '{"N":16000000,"main":4,"skip":1}'
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab, items
from lab import V, HIST, bits, log
import exp_scale as es, exp_cards as XC, exp_ladder as L

es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


class Chain:
    """絶対割引つきバックオフ。levels = 文脈の取り方の列 (例 [(), (1,), (1,2), (1,2,3)])。1=直前の語, 2=その前..."""
    def __init__(s, X, y, levels, delta=0.75):
        s.levels, s.delta, s.t = levels, delta, []
        for lv in levels:
            ck = s.code(X, lv)
            up, pc = np.unique(ck * V + y, return_counts=True)
            uc, tot = np.unique(ck, return_counts=True)
            _, typ = np.unique(up // V, return_counts=True)
            s.t.append((uc, tot, typ, up, pc))

    @staticmethod
    def code(X, lv):
        c = np.zeros(len(X), np.int64)
        for j, p in enumerate(lv):
            c += X[:, HIST - p].astype(np.int64) * (V ** j)
        return c

    def probs(s, X, y):
        p = np.full(len(y), 1.0 / V); alive = np.ones(len(y), bool); mk = np.zeros(len(y), int)
        for k, lv in enumerate(s.levels):
            uc, tot, typ, up, pc = s.t[k]
            cq = s.code(X, lv)
            i = np.minimum(np.searchsorted(uc, cq), len(uc) - 1)
            found = alive & (uc[i] == cq)
            pq = cq * V + y
            j = np.minimum(np.searchsorted(up, pq), len(up) - 1)
            c = np.where(up[j] == pq, pc[j], 0)
            newp = np.maximum(c - s.delta, 0) / tot[i] + s.delta * typ[i] / tot[i] * p
            p = np.where(found, newp, p); mk = np.where(found, k, mk); alive = found
        return p, mk

    def ops(s, N):
        return float(len(s.levels) * N * np.log2(N))


def mix_bucket(Pva, Pte, bva, bte, minn=300):
    wg = L.em(Pva)
    out = np.zeros(Pte.shape[1])
    for b in np.unique(bte):
        mv = bva == b; mt = bte == b
        w = L.em(Pva[:, mv]) if mv.sum() >= minn else wg
        out[mt] = (w[:, None] * Pte[:, mt]).sum(0)
    return out


def run(cfg):
    t0 = time.time(); torch.set_num_threads(os.cpu_count() or 2)
    N = cfg['N']; nq = cfg.get('nq', 20000)
    main_len = cfg.get('main', 3); skip = cfg.get('skip', 0); bucket = cfg.get('bucket', 0)
    key = cfg.get('key', 'ppmi'); de = cfg.get('de', 64); nn_h = cfg.get('nn_h', 0); nn_n = cfg.get('nn_n', 1000000)
    delta = cfg.get('delta', 0.75)
    ctx = lab.load(N)
    N = len(ctx.Xtr)
    sl = slice(0, nq)
    res = dict(cfg=cfg, N=N); ops = {}
    pu = np.bincount(ctx.ytr, minlength=V).astype(np.float64) + 1.0; pu /= pu.sum()
    yva, yte = ctx.yva[sl], ctx.yte[sl]
    Xva, Xte = ctx.Xva[sl], ctx.Xte[sl]
    # ---- 数える
    levels = [tuple(range(1, k + 1)) for k in range(main_len + 1)]
    main = Chain(ctx.Xtr, ctx.ytr, levels, delta)
    ops['count_main'] = main.ops(N)
    pm_va, mk_va = main.probs(Xva, yva); pm_te, mk_te = main.probs(Xte, yte)
    res['main_only'] = bits(pm_te)
    Pva, Pte, names = [pm_va], [pm_te], ['main']
    if skip:
        for nm, lvs in (('skipA', [(), (2,), (2, 3), (2, 3, 4)]), ('skipB', [(), (1,), (1, 3), (1, 3, 4)])):
            ch = Chain(ctx.Xtr, ctx.ytr, lvs, delta); ops['count_' + nm] = ch.ops(N)
            Pva.append(ch.probs(Xva, yva)[0]); Pte.append(ch.probs(Xte, yte)[0]); names.append(nm)
    log('数える完了 main %.4f [%.0fs]' % (res['main_only'], time.time() - t0))
    # ---- 文書内
    if cfg.get('cache', 1):
        cp_va, _ = XC.copy_probs(Xva, yva, pu); cp_te, _ = XC.copy_probs(Xte, yte, pu)
        ca_va = XC.cache_probs(Xva, yva, pu); ca_te = XC.cache_probs(Xte, yte, pu)
        Pva += [cp_va, ca_va]; Pte += [cp_te, ca_te]; names += ['copy', 'cache']

    def final(Pv, Pt):
        Pv, Pt = np.array(Pv), np.array(Pt)
        if bucket:
            return bits(mix_bucket(Pv, Pt, mk_va, mk_te))
        w = L.em(Pv); return bits((w[:, None] * Pt).sum(0))
    res['no_index'] = final(Pva, Pte)
    log('文書内まで: %.4f [%.0fs]' % (res['no_index'], time.time() - t0))
    # ---- 索引
    E0 = None
    if key != 'none' or nn_h:
        if key == 'rand':
            E0 = np.random.default_rng(1).normal(size=(V, de)).astype(np.float32)
            if nn_h:
                E0 = None
        if E0 is None:
            E0 = items.ppmi_embeddings(ctx, de if not nn_h else lab.DE)
            ops['svd'] = float(2 * V * V * E0.shape[1] * 7 + 22 * N + 10 * V * V)
    if key != 'none':
        Ek = E0 if E0.shape[1] == de or key == 'rand' else E0[:, :de]
        def keys(X):
            k = Ek[X[:, -3:]].reshape(len(X), -1).astype(np.float32)
            return k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)
        Kd = torch.cat([torch.from_numpy(np.round(keys(ctx.Xtr[i:i + 500000]) * 127).astype(np.int8)) for i in range(0, N, 500000)])
        yd = torch.from_numpy(ctx.ytr); ops['index_keys'] = float(N * 3 * Ek.shape[1])
        def knn(Xq, yq, T, k=64):
            Q = torch.from_numpy(keys(Xq)); out = np.zeros(len(Q))
            for qi in range(0, len(Q), 500):
                qb = Q[qi:qi + 500]; bs = bi = None
                for c in range(0, N, 500000):
                    sc = qb @ (Kd[c:c + 500000].float() / 127).T
                    v, ix = sc.topk(k, 1); ix = ix + c
                    if bs is None: bs, bi = v, ix
                    else:
                        bs, j = torch.cat([bs, v], 1).topk(k, 1); bi = torch.cat([bi, ix], 1).gather(1, j)
                wt = torch.softmax(bs / T, 1)
                out[qi:qi + 500] = (wt * (yd[bi] == torch.from_numpy(yq[qi:qi + 500])[:, None])).sum(1).numpy()
            return out * 0.98 + 0.02 * pu[yq]
        best = None
        for T in (0.02, 0.05):
            kv = knn(Xva, yva, T); b = final(Pva + [kv], Pte + [kv])   # 重みの選び方確認用(後でテストに使うのは最良のTだけ)
            if best is None or b < best[0]: best = (b, T, kv)
        _, T, kv = best
        Pva.append(kv); Pte.append(knn(Xte, yte, T)); names.append('index'); res['T'] = T
        res['with_index'] = final(Pva, Pte)
        log('索引まで: %.4f [%.0fs]' % (res['with_index'], time.time() - t0))
    # ---- 極小NN(勾配あり)
    if nn_h:
        idx = np.random.default_rng(0).choice(len(ctx.Xtr), min(nn_n, len(ctx.Xtr)), replace=False)
        net = lab.train(ctx, nn_h, 1.0, E0=E0, idx=idx)
        fv = lab.forward_all(net, ctx, 'va')['p'][sl]; ft = lab.forward_all(net, ctx, 'te')['p'][sl]
        res['nn_alone'] = bits(ft)
        Pva.append(fv); Pte.append(ft); names.append('nn')
        ops['nn_train'] = float(6 * 5576 * nn_h * len(idx))
        res['with_nn'] = final(Pva, Pte)
        log('NNまで: %.4f [%.0fs]' % (res['with_nn'], time.time() - t0))
    res['final'] = res.get('with_nn', res.get('with_index', res['no_index']))
    res['names'] = names; res['ops'] = ops; res['ops_total'] = float(sum(ops.values())); res['sec'] = time.time() - t0
    log('最終 %.4f | 学習の演算回数 約%.2e %s [%.0fs]' % (res['final'], res['ops_total'], {k: '%.1e' % v for k, v in ops.items()}, time.time() - t0))
    log('RESULT ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
