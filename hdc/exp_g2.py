"""
G2: 途中変数 (繰り上がり) を自動で見つけ、 学習に出なかった (一の位,一の位) の組でも十の位を当てられるか。
手順 (勾配なし・全部 数え上げか閉じた形の最小二乗):
  1. 一の位の規則 s1=(a1+b1)%10 を ALS (CP分解) で得る (見ていない組も予測する)。
  2. 潜在変数の発見: 学習に出た各組 (a1,b1) について、 十の位の出力を全ての (a0,b0) にわたって並べた「署名」を作り、
     署名が同じ組を同じクラスにまとめる (予測が同じなら同じ = 予測でまとめる)。 クラス数が自動で決まる。
  3. 見ていない組のクラス: 候補の関係特徴 rel(u,v) (u,v は a1,b1,s1 のうち2つ; rel は <, ==, >) から
     学習組でクラスを最もよく当てる特徴を選ぶ。 (事前知識: 数字の大小比較ができる)
  4. 十の位 = 表(a0,b0,クラス)。
比較/掛け算も同じ手順で試す。 掛け算は ALS だけ (関係特徴では足りない見込み)。
使い方: python3 exp_g2.py '{}'
"""
import sys, json, itertools
import numpy as np
from exp_trace_als import cp_als, predict


def table_fit(keys, y):
    d = {}
    for k, v in zip(map(tuple, keys.tolist()), y.tolist()):
        d.setdefault(k, {}); d[k][v] = d[k].get(v, 0) + 1
    return {k: max(c, key=c.get) for k, c in d.items()}


def rel_feats(a, b, s):
    cols = {'a1': a, 'b1': b, 's1': s}; out = {}
    for (n1, u), (n2, v) in itertools.combinations(cols.items(), 2):
        out['%s<%s' % (n1, n2)] = (u < v).astype(int); out['%s==%s' % (n1, n2)] = (u == v).astype(int); out['%s>%s' % (n1, n2)] = (u > v).astype(int)
    return out


def best_als(idx, Y, dims, r, held_idx, held_y, restarts=6, lam=1e-3, iters=60, seed=100):
    best = None
    for s in range(restarts):
        F, W = cp_als(idx, Y, dims, r, lam, iters, np.random.default_rng(seed + s))
        tr = (predict(F, W, idx)[0] == Y.argmax(1)).mean()
        if best is None or tr > best[0] + 1e-9:
            best = (tr, F, W)
    te = (predict(best[1], best[2], held_idx)[0] == held_y).mean() if len(held_idx) else None
    return best[0], te, best[1], best[2]


def main(cfg):
    rng = np.random.default_rng(0)
    pairs = np.array([(a, b) for a in range(10) for b in range(10)])
    held = pairs[rng.choice(100, cfg.get('nheld', 20), replace=False)]
    hs = set(map(tuple, held.tolist()))
    isheld_pair = np.array([tuple(p) in hs for p in pairs.tolist()])
    obs_pairs = pairs[~isheld_pair]; held_pairs = pairs[isheld_pair]
    res = {}
    # ---- 1. 一の位
    yo = obs_pairs.sum(1) % 10; r_ones = cfg.get('rank_ones', 16)
    tr, te, F1, W1 = best_als(obs_pairs, np.eye(10)[yo], [10, 10], r_ones, held_pairs, held_pairs.sum(1) % 10)
    print('一の位 ALS rank=%d: 学習 %.3f 除いた組 %.3f' % (r_ones, tr, te), flush=True); res['ones'] = dict(train=tr, held=te)
    s_hat = np.zeros(100, int)
    s_hat[~isheld_pair] = yo; s_hat[isheld_pair] = predict(F1, W1, held_pairs)[0]            # 学習組は観測した値、 除いた組は ALS の予測
    # ---- 2. 署名クラスタ (加算の十の位)
    cells = np.array([(a0, b0, a1, b1) for a0 in range(10) for b0 in range(10) for a1 in range(10) for b1 in range(10)])
    carry = (cells[:, 2] + cells[:, 3]) // 10; y = (cells[:, 0] + cells[:, 1] + carry) % 10
    pair_id = cells[:, 2] * 10 + cells[:, 3]
    sig = {}
    for p in range(100):
        if isheld_pair[p]:
            continue
        m = pair_id == p; o = np.lexsort((cells[m, 1], cells[m, 0])); sig[p] = tuple(y[m][o].tolist())
    uniq = {}; cls = {}
    for p, s in sig.items():
        cls[p] = uniq.setdefault(s, len(uniq))
    print('発見した潜在クラス数: %d (観測した組 %d)' % (len(uniq), len(sig)), flush=True); res['n_classes'] = len(uniq)
    true_c = {p: (p // 10 + p % 10) // 10 for p in range(100)}
    # クラスが繰り上がりと一致しているか (ラベルの付け替えを除く)
    agree = len({(cls[p], true_c[p]) for p in cls}) == len(uniq)
    print('クラス = 真の繰り上がり (付け替えを除き一致): %s' % agree, flush=True); res['classes_equal_carry'] = bool(agree)
    # ---- 3. 見ていない組のクラス
    rf = rel_feats(pairs[:, 0], pairs[:, 1], s_hat)
    ob_idx = np.array(sorted(cls)); ob_c = np.array([cls[p] for p in ob_idx])
    sc = []
    for nm, f in rf.items():
        tb = table_fit(f[ob_idx][:, None], ob_c); acc = np.mean([tb.get((int(f[p]),), -1) == c for p, c in zip(ob_idx, ob_c)]); sc.append((acc, nm))
    sc.sort(reverse=True); print('関係特徴の学習組での正解率 上位:', [(round(a, 3), n) for a, n in sc[:4]], flush=True)
    bn = sc[0][1]; f = rf[bn]; tb = table_fit(f[ob_idx][:, None], ob_c)
    held_idx = np.flatnonzero(isheld_pair)
    pred_c = {p: tb.get((int(f[p]),), -1) for p in held_idx}
    truth_c_map = {c: t for (c, t) in {(cls[p], true_c[p]) for p in cls}}               # クラス -> 繰り上がり (評価用)
    cls_acc = np.mean([truth_c_map.get(pred_c[p], -9) == true_c[p] for p in held_idx])
    print('選んだ特徴 %s: 除いた組でのクラス正解率 %.3f' % (bn, cls_acc), flush=True); res['class_feature'] = bn; res['class_acc_held'] = float(cls_acc)
    # ---- 4. 十の位の表 (a0,b0,クラス)
    allc = np.array([cls[p] if p in cls else pred_c[p] for p in pair_id])
    obs_cell = ~isheld_pair[pair_id]
    key = np.stack([cells[:, 0], cells[:, 1], allc], 1)
    tb2 = table_fit(key[obs_cell], y[obs_cell])
    pr = np.array([tb2.get(tuple(k), -1) for k in key[~obs_cell].tolist()])
    held_tens = float((pr == y[~obs_cell]).mean()); obs_tens = float(np.mean([tb2.get(tuple(k), -1) == v for k, v in zip(key[obs_cell].tolist(), y[obs_cell].tolist())]))
    print('十の位: 学習 %.4f, 除いた組 %.4f (n=%d)  [ALSのみ: 71%%]' % (obs_tens, held_tens, int((~obs_cell).sum())), flush=True)
    res['tens'] = dict(train=obs_tens, held=held_tens)
    # ---- 比較 (a<b を 2 桁で): 出力 = a<b。 同じ手順 (署名->クラス->関係特徴)
    lt = (cells[:, 0] * 10 + cells[:, 2] < cells[:, 1] * 10 + cells[:, 3]).astype(int)
    # 入力の並び: (a0,b0,a1,b1) の意味を (a0=a十,b0=b十,a1=a一,b1=b一) とする
    sigc = {}
    for p in range(100):
        if isheld_pair[p]:
            continue
        m = pair_id == p; o = np.lexsort((cells[m, 1], cells[m, 0])); sigc[p] = tuple(lt[m][o].tolist())
    uq = {}; clc = {p: uq.setdefault(s, len(uq)) for p, s in sigc.items()}
    rf2 = rel_feats(pairs[:, 0], pairs[:, 1], s_hat)
    ob2 = np.array(sorted(clc)); oc2 = np.array([clc[p] for p in ob2]); sc2 = []
    for nm, f2 in rf2.items():
        t_ = table_fit(f2[ob2][:, None], oc2); sc2.append((np.mean([t_.get((int(f2[p]),), -1) == c for p, c in zip(ob2, oc2)]), nm))
    sc2.sort(reverse=True); bn2 = sc2[0][1]; f2 = rf2[bn2]; t2 = table_fit(f2[ob2][:, None], oc2)
    allc2 = np.array([clc[p] if p in clc else t2.get((int(f2[p]),), -1) for p in pair_id])
    key2 = np.stack([cells[:, 0], cells[:, 1], allc2], 1); tb3 = table_fit(key2[obs_cell], lt[obs_cell])
    pr2 = np.array([tb3.get(tuple(k), -1) for k in key2[~obs_cell].tolist()])
    print('比較: クラス数 %d, 選んだ特徴 %s (学習 %.3f), 除いた組の正解率 %.4f' % (len(uq), bn2, sc2[0][0], (pr2 == lt[~obs_cell]).mean()), flush=True)
    res['compare'] = dict(n_classes=len(uq), feature=bn2, held=float((pr2 == lt[~obs_cell]).mean()))
    # ---- 掛け算 (1桁×1桁): ALS のみ
    prod = pairs[:, 0] * pairs[:, 1]; res['mult'] = {}
    for nm, yy in (('ones', prod % 10), ('tens', prod // 10)):
        for r in cfg.get('ranks_mult', [8, 16, 24, 32]):
            tr, te, _, _ = best_als(obs_pairs, np.eye(10)[yy[~isheld_pair]], [10, 10], r, held_pairs, yy[isheld_pair], restarts=8)
            print('掛け算 %s の位 rank=%d: 学習 %.3f 除いた組 %.3f' % (nm, r, tr, te), flush=True); res['mult'][nm + str(r)] = dict(train=tr, held=te)
    print('RESULT_G2 ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1] if len(sys.argv) > 1 else '{}'))
