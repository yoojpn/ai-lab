"""
TRACE: 数え上げだけで「変数の値を引く」「足す」を学べるか。 規則は人が書かず、 文脈の候補から数えた正解率で自動に選ぶ。
データ (自動生成):
  const 行 : T := d d : T = d d ;                       (T: 変数名 a-f, d: 数字)
  add   行 : T := X + Y : X = d d Y = d d T = s s ;      (X,Y: 定義済みの変数, s s = (X+Y) mod 100 の 2 桁)
  役割     : syntax(:= + : = ;), copy_name(行の中で名前を写す), lookup(X = ?? の2桁を前の定義から引く),
             sum_tens / sum_ones (十の位 / 一の位), copy_const (const 行の写し)
  ランダムな語 (行頭の変数名, 演算の変数名, const の数字) は 当てる対象にしない。
特徴 (どれも数え上げ用の離散値):
  off_o : o 語前の語 (o=1..16)
  ptr_m_v : 直前の m 語 (m=1,2,3) が プログラム内の前の場所に出ていれば、 その場所の v 語先 (v=1..3) の語 (induction)
規則の探索 : 役割ごとに、 特徴の部分集合 (大きさ1..4) を総当たりし、 訓練で 文脈 -> 最頻の語 の表を作り、 検証の正解率が最も高いものを選ぶ。
             役割は 評価用に分かっているものとして使う (役割の自動判定は 別の課題)。
評価 (教師強制の次語の正解率, 役割ごと): 通常 / 長いプログラム / 学習から除いた (一の位, 一の位) の組を含む行
使い方: python3 exp_trace.py '{"ntrain":30000,"nval":3000,"ntest":3000}'
"""
import sys, json, time, itertools
import numpy as np

TOKS = ['<pad>', '<bos>', ':=', ':', '=', '+', ';', 'a', 'b', 'c', 'd', 'e', 'f'] + [str(i) for i in range(10)]
ID = {t: i for i, t in enumerate(TOKS)}; NT = len(TOKS)
VARS = ['a', 'b', 'c', 'd', 'e', 'f']
ROLES = ['syntax', 'copy_name', 'lookup', 'sum_tens', 'sum_ones', 'copy_const']
R = {r: i for i, r in enumerate(ROLES)}
NOFF = 16; PTRM = (1, 2, 3); PTRV = (1, 2, 3)


def digits(v):
    return [ID[str(v // 10)], ID[str(v % 10)]]


def gen_program(rng, nst, held, exclude=True):
    vals = {}; toks = [ID['<bos>']]; roles = [-1]; hflag = [0]

    def emit(ts, rs, hs=None):
        toks.extend(ts); roles.extend(rs); hflag.extend(hs if hs is not None else [0] * len(ts))
    for s in range(nst):
        defined = list(vals)
        add = (s >= 2 and len(defined) >= 1 and rng.random() < 0.8)
        if add:
            for _ in range(30):
                X = defined[rng.integers(len(defined))]; Y = defined[rng.integers(len(defined))]
                if held is None or (not exclude) or (vals[X] % 10, vals[Y] % 10) not in held:
                    break
            else:
                add = False
        if add:
            T = VARS[rng.integers(len(VARS))]; v = (vals[X] + vals[Y]) % 100
            hf = 1 if (held is not None and (vals[X] % 10, vals[Y] % 10) in held) else 0
            emit([ID[T], ID[':='], ID[X], ID['+'], ID[Y], ID[':'], ID[X], ID['='], *digits(vals[X]), ID[Y], ID['='], *digits(vals[Y]), ID[T], ID['='], *digits(v), ID[';']],
                 [-1, R['syntax'], -1, R['syntax'], -1, R['syntax'], R['copy_name'], R['syntax'], R['lookup'], R['lookup'], R['copy_name'], R['syntax'], R['lookup'], R['lookup'], R['copy_name'], R['syntax'], R['sum_tens'], R['sum_ones'], R['syntax']],
                 [0] * 16 + [hf, hf, 0])
            vals[T] = v
        else:
            T = VARS[rng.integers(len(VARS))]; v = int(rng.integers(100))
            emit([ID[T], ID[':='], *digits(v), ID[':'], ID[T], ID['='], *digits(v), ID[';']],
                 [-1, R['syntax'], -1, -1, R['syntax'], R['copy_name'], R['syntax'], R['copy_const'], R['copy_const'], R['syntax']])
            vals[T] = v
    return toks, roles, hflag


def make_set(rng, n, lo, hi, held, exclude=True):
    T, Ro, H, S = [], [], [], []; pos = 0
    for _ in range(n):
        t, r, h = gen_program(rng, int(rng.integers(lo, hi + 1)), held, exclude)
        T += t; Ro += r; H += h; S += [pos] * len(t); pos += len(t)
    return np.array(T, np.int16), np.array(Ro, np.int8), np.array(H, np.int8), np.array(S, np.int64)


def prev_same(key, start):
    N = len(key); idx = np.lexsort((np.arange(N), key)); ks = key[idx]
    same = np.r_[False, ks[1:] == ks[:-1]]; prev = np.full(N, -1, np.int64)
    prev[idx[same]] = idx[np.flatnonzero(same) - 1]
    ok = (prev >= 0)
    ok[ok] = start[prev[ok]] == start[ok]
    prev[~ok] = -1
    return prev


def features(tok, start):
    """行 i の特徴は、 i 番目の語を当てるために使える (i より前の) 情報だけ"""
    N = len(tok); t = tok.astype(np.int64); F = []; names = []
    ar = np.arange(N)
    for o in range(1, NOFF + 1):
        j = ar - o; ok = j >= start
        F.append(np.where(ok, t[np.maximum(j, 0)], 0)); names.append('off%d' % o)
    # ポインタ: j = i-1 を最後の語とする m 語の並び
    for m in PTRM:
        key = np.zeros(N, np.int64)
        for k in range(m):
            jj = ar - k; key = key * NT + np.where(jj >= 0, t[np.maximum(jj, 0)], 0)
        pv = prev_same(key, start)                        # j における 同じ並びの 前の位置 (その並びの最後の語の位置)
        for v in PTRV:
            # 位置 i の特徴 = 位置 j=i-1 の ポインタ から読んだ値 (v 先が i-1 以下のとき)
            pj = np.full(N, -1, np.int64); pj[1:] = pv[:-1]
            pp = pj + v; val = np.where((pj >= 0) & (pp <= ar - 1), t[np.clip(pp, 0, N - 1)], 0)
            F.append(val); names.append('ptr%d_%d' % (m, v))
    return np.stack(F, 1).astype(np.int8), names


def encode(Fm, sub):
    key = np.zeros(len(Fm), np.int64)
    for s in sub:
        key = key * NT + Fm[:, s].astype(np.int64)
    return key


def fit_table(key, y):
    kl = key * NT + y.astype(np.int64)
    u, c = np.unique(kl, return_counts=True); ku = u // NT; lu = u % NT
    o = np.lexsort((-c, ku)); ku, lu = ku[o], lu[o]
    first = np.r_[True, ku[1:] != ku[:-1]]
    return ku[first], lu[first]


def predict(table, key, default):
    ku, lu = table; i = np.searchsorted(ku, key); i = np.minimum(i, len(ku) - 1)
    hit = ku[i] == key
    return np.where(hit, lu[i], default), hit


def search_role(Ftr, ytr, Fva, yva, nf, maxk=4, target=0.9995, log=print):
    best = None
    for k in range(1, maxk + 1):
        bk = None
        for sub in itertools.combinations(range(nf), k):
            tab = fit_table(encode(Ftr, sub), ytr)
            pr, hit = predict(tab, encode(Fva, sub), 0)
            acc = float((pr == yva).mean())
            if bk is None or acc > bk[0]:
                bk = (acc, sub)
        log('    大きさ%d の最良 検証 %.4f %s' % (k, bk[0], bk[1]))
        if best is None or bk[0] > best[0] + 1e-9:
            best = bk
        if bk[0] >= target:
            break
    return best


def evaluate(tok, roles, hflag, F, sub, table, default, mask_role, only_held=False):
    pos = np.flatnonzero((roles == mask_role) & ((hflag == 1) if only_held else True))
    if len(pos) == 0:
        return None
    pr, hit = predict(table, encode(F[pos], sub), default)
    return float((pr == tok[pos]).mean()), len(pos), float(hit.mean())


def main(cfg):
    rng = np.random.default_rng(0); t0 = time.time()
    log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    HELD = None
    if cfg.get('held', True):
        pairs = [(a, b) for a in range(10) for b in range(10)]
        HELD = set(map(tuple, np.array(pairs)[rng.choice(100, cfg.get('nheld', 20), replace=False)].tolist()))
    sets = {}
    sets['train'] = make_set(rng, cfg['ntrain'], 4, 8, HELD); sets['val'] = make_set(rng, cfg['nval'], 4, 8, HELD)
    sets['test'] = make_set(rng, cfg['ntest'], 4, 8, HELD, False); sets['long'] = make_set(rng, cfg['ntest'], 10, 14, HELD, False)
    # 通常の学習 (除かない) も比較用に
    full = make_set(rng, cfg['ntrain'], 4, 8, None); fullval = make_set(rng, cfg['nval'], 4, 8, None)
    for k, v in sets.items():
        log('%s: %d 語, 当てる対象 %d' % (k, len(v[0]), int((v[1] >= 0).sum())))
    feats = {}; names = None
    for k, v in list(sets.items()) + [('full', full), ('fullval', fullval)]:
        feats[k], names = features(v[0], v[3])
    nf = len(names); log('特徴 %d 個: %s' % (nf, ' '.join(names)))
    res = {}
    for mode, trk, vak in (('除かない学習', 'full', 'fullval'), ('一の位の組20個を学習から除く', 'train', 'val')):
        trd = full if trk == 'full' else sets['train']; vad = fullval if vak == 'fullval' else sets['val']
        log('=== %s ===' % mode); res[mode] = {}
        for role in ['syntax', 'copy_name', 'lookup', 'copy_const', 'sum_ones', 'sum_tens']:
            rid = R[role]
            ptr = np.flatnonzero(trd[1] == rid); vpt = np.flatnonzero(vad[1] == rid)
            if len(ptr) > 60000:
                ptr = ptr[:60000]
            Ftr, ytr = feats[trk][ptr], trd[0][ptr].astype(np.int64); Fva, yva = feats[vak][vpt], vad[0][vpt].astype(np.int64)
            log('  役割 %s: 訓練 %d 検証 %d' % (role, len(ptr), len(vpt)))
            acc, sub = search_role(Ftr, ytr, Fva, yva, nf, log=log)
            allp = np.flatnonzero(trd[1] == rid)
            tab = fit_table(encode(feats[trk][allp], sub), trd[0][allp].astype(np.int64))
            default = int(np.bincount(trd[0][allp].astype(np.int64)).argmax())
            row = dict(sub=[names[s] for s in sub], val=acc)
            for nm in ('test', 'long'):
                r_ = evaluate(sets[nm][0].astype(np.int64), sets[nm][1], sets[nm][2], feats[nm], sub, tab, default, rid)
                row[nm] = r_
            if role in ('sum_ones', 'sum_tens'):
                row['held'] = evaluate(sets['test'][0].astype(np.int64), sets['test'][1], sets['test'][2], feats['test'], sub, tab, default, rid, only_held=True)
            res[mode][role] = row
            log('  -> %s: 規則 %s | 検証 %.4f | 通常 %s | 長い %s | 除いた組 %s' % (role, row['sub'], acc, row['test'] and '%.4f' % row['test'][0], row['long'] and '%.4f' % row['long'][0], row.get('held') and '%.4f(n=%d)' % (row['held'][0], row['held'][1])))
    log('RESULT_TRACE ' + json.dumps(dict(cfg=cfg, held=sorted(HELD) if HELD else None, res=res)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
