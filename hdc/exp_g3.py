"""
G3: 本物の Python を実行して作った実行記録 (トレース) を、 役割を教えずに、 規則被覆で当てられるか。
  プログラム: 変数 a-f、 値 0..99、 文 = 定数代入 / 二項(+,-) / 定数との加減 / if(1文) / for i in range(k)(1文)。
  記録: 実際に sys.settrace で実行して、 実行された代入ごとに  'L<行> 変数 = 値 NL'  を出す (ループは繰り返しごと)。
  入力の並び: [ソース (行ごとに L<n> ... NL)] TRACE [記録]。  記録の語を 1 語ずつ当てる。
  特徴 (形式は探索に合わせない。 一般の「参照」だけ): off_o (o語前), lo(o,v) (その語の前回の出現の v 語先), fo(o,v) (同 初出の v 語先),
      D(f,u) (特徴 f が指す変数の、 記録上の最新の値の u 桁目), since_NL, in_trace。
  評価: 語の正解率 (未説明は不正解), 記録の完全一致(全部の語が正解), 語の種類別。
使い方: python3 exp_g3.py '{"ntrain":6000,"ntest":600}'
"""
import sys, json, time, re
import numpy as np
from rulecover import RuleCover

NAMES = list('abcdef')
TOK = ['<pad>', 'TRACE', 'NL', 'IN', '=', '+', '-', '>', '<', '==', ':', '(', ')', 'if', 'for', 'in', 'range', 'i'] + NAMES + [str(d) for d in range(10)] + ['L%d' % k for k in range(1, 31)]
ID = {t: k for k, t in enumerate(TOK)}; NV = len(TOK)
NAMEIDS = [ID[n] for n in NAMES + ['i']]


def digs(v):
    return list(str(v))


def gen(rng, nst):
    """戻り値: (source行のリスト[(kind,tokens)], python text, 値の検証)"""
    env = {}; lines = []; defined = []

    def stmt_assign(env, tgt=None, allow_i=False):
        names = [n for n in defined] + (['i'] if allow_i else [])
        for _ in range(30):
            t = rng.choice(defined) if tgt is None else tgt
            kind = rng.integers(3)
            if kind == 0 and len(names) >= 1:
                u = rng.choice(names); w = rng.choice(names); op = '+' if rng.random() < 0.6 else '-'
                v = env[u] + env[w] if op == '+' else env[u] - env[w]
                if 0 <= v <= 99:
                    return t, [t, '=', u, op, w], v
            elif kind == 1:
                u = rng.choice(names); k = int(rng.integers(1, 21)); op = '+' if rng.random() < 0.6 else '-'
                v = env[u] + k if op == '+' else env[u] - k
                if 0 <= v <= 99:
                    return t, [t, '=', u, op] + digs(k), v
            else:
                k = int(rng.integers(0, 100)); return t, [t, '=', *digs(k)], k
        return None
    for s in range(nst):
        r = rng.random()
        if len(defined) < 2 or r < 0.4:
            t = NAMES[rng.integers(6)]; k = int(rng.integers(0, 100)); env[t] = k
            if t not in defined:
                defined.append(t)
            lines.append(('assign', t, [t, '=', *digs(k)], None)); continue
        if r < 0.7:
            a = stmt_assign(env)
            if a is None:
                continue
            t, rhs, v = a; env[t] = v; lines.append(('assign', t, rhs, None)); continue
        if r < 0.85:
            u = rng.choice(defined); w = rng.choice(defined); cop = ['>', '<', '=='][rng.integers(3)]
            cond = [u, cop, w] if rng.random() < 0.6 else [u, cop, *digs(int(rng.integers(0, 100)))]
            a = stmt_assign(env)
            if a is None:
                continue
            t, rhs, v = a
            cv = env[cond[2]] if cond[2] in env else int(''.join(cond[2:]))
            truth = {'>': env[u] > cv, '<': env[u] < cv, '==': env[u] == cv}[cop]
            if truth:
                env[t] = v
            lines.append(('if', None, ['if'] + cond + [':'], None)); lines.append(('body', t, rhs, None)); continue
        # for
        k = int(rng.integers(2, 4)); tgt = rng.choice(defined); ok = False
        for _ in range(10):
            e2 = dict(env); good = True; body = None
            if rng.random() < 0.5:
                rhs = [tgt, '=', tgt, '+', 'i']
            else:
                kk = int(rng.integers(1, 6)); rhs = [tgt, '=', tgt, '+', *digs(kk)]
            for it in range(k):
                e2['i'] = it
                val = e2[tgt] + (it if rhs[-1] == 'i' else int(''.join(rhs[4:])))
                if not 0 <= val <= 99:
                    good = False; break
                e2[tgt] = val
            if good:
                ok = True; break
        if not ok:
            continue
        env = e2; lines.append(('for', 'i', ['for', 'i', 'in', 'range', '(', *digs(k), ')', ':'], None)); lines.append(('body', tgt, rhs, None))
    return lines


def to_py(lines):
    out = []
    for kind, tgt, toks, _ in lines:
        t = ' '.join(toks)
        t = re.sub(r'(\d) (\d)', r'\1\2', t); t = re.sub(r'(\d) (\d)', r'\1\2', t)
        t = t.replace('range ( ', 'range(').replace(' ) :', '):').replace(' :', ':')
        out.append(('    ' if kind == 'body' else '') + t)
    return '\n'.join(out) + '\n'


def run_trace(lines, src):
    g = {'__builtins__': {'range': range}}; ev = []; st = {'prev': {}, 'last': None}
    def tr(frame, event, arg):
        if frame.f_code.co_filename != '<p>':
            return None
        if event in ('line', 'return'):
            cur = {k: v for k, v in g.items() if k != '__builtins__'}
            if st['last'] is not None:
                ln = st['last']; kind, tgt = lines[ln - 1][0], lines[ln - 1][1]
                if kind in ('assign', 'body'):
                    ev.append((ln, tgt, cur[tgt]))
                elif kind == 'for' and cur.get('i') != st['prev'].get('i'):
                    ev.append((ln, 'i', cur['i']))
            st['prev'] = cur; st['last'] = frame.f_lineno if event == 'line' else None
        return tr
    import sys as _s
    code = compile(src, '<p>', 'exec'); _s.settrace(tr)
    try:
        exec(code, g)
    finally:
        _s.settrace(None)
    return ev


def make_stream(rng, nst):
    while True:
        lines = gen(rng, nst)
        if len(lines) < 3:
            continue
        src = to_py(lines)
        try:
            ev = run_trace(lines, src)
        except Exception:
            continue
        # 自分の模擬実行との照合: 記録の最後の値が実際の値と同じ (run_trace は本物の実行)
        toks = ['<pad>']
        toks = []
        for n, (kind, tgt, tk, _) in enumerate(lines, 1):
            toks += ['L%d' % n] + (['IN'] if kind == 'body' else []) + tk + ['NL']
        toks += ['TRACE']; tstart = len(toks)
        for ln, nm, val in ev:
            toks += ['L%d' % ln, nm, '='] + digs(val) + ['NL']
        if len(toks) > 300 or ev and any(l > 30 for l, _, _ in ev):
            continue
        return [ID[t] for t in toks], tstart, src


def build_set(rng, n, lo, hi):
    T = []; S = []; TS = []; pos = 0; srcs = []
    for _ in range(n):
        t, ts, src = make_stream(rng, int(rng.integers(lo, hi + 1)))
        T += t; S += [pos] * len(t); TS += [pos + ts] * len(t); pos += len(t); srcs.append(src)
    return np.array(T, np.int16), np.array(S, np.int64), np.array(TS, np.int64), srcs


def prev_occ(t, start):
    N = len(t); idx = np.lexsort((np.arange(N), t)); ks = t[idx]
    same = np.r_[False, ks[1:] == ks[:-1]]; prev = np.full(N, -1, np.int64); prev[idx[same]] = idx[np.flatnonzero(same) - 1]
    ok = prev >= 0; ok[ok] = start[prev[ok]] == start[ok]; prev[~ok] = -1
    first = np.full(N, -1, np.int64)
    # 初出: 同じ (プログラム, 語) の最小位置
    key = start * 256 + t.astype(np.int64); o = np.lexsort((np.arange(N), key)); ky = key[o]
    fs = np.r_[True, ky[1:] != ky[:-1]]; fid = np.maximum.accumulate(np.where(fs, np.arange(N), 0)); first[o] = o[fid]
    return prev, first


def features(tok, start, tstart):
    N = len(tok); t = tok.astype(np.int64); ar = np.arange(N); F = []; names = []
    def at(p, valid):
        return np.where(valid, t[np.clip(p, 0, N - 1)], 0)
    prev, first = prev_occ(t, start)
    for o in range(1, 13):
        j = ar - o; F.append(at(j, j >= start)); names.append('off%d' % o)
    for o in (1, 2):
        for v in (1, 2):
            j = ar - o; p = np.where(j >= 0, prev[np.clip(j, 0, N - 1)], -1); q = p + v
            F.append(at(q, (p >= 0) & (q <= ar - 1))); names.append('lo%d_%d' % (o, v))
    fo_feats = {}
    for o in range(1, 7):
        for v in range(1, 6):
            j = ar - o; p = np.where(j >= 0, first[np.clip(j, 0, N - 1)], -1); q = p + v
            f = at(q, (p >= 0) & (p < j) & (q <= ar - 1)); F.append(f); names.append('fo%d_%d' % (o, v)); fo_feats[(o, v)] = len(F) - 1
    # 記録上の最新の定義 (完成した行のみ): 変数 n ごとに 位置
    EQ, NL, TR = ID['='], ID['NL'], ID['TRACE']
    nl_next = np.full(N, N, np.int64); nlpos = np.flatnonzero(t == NL)
    nxt = np.searchsorted(nlpos, ar, side='right'); nl_next = np.where(nxt < len(nlpos), nlpos[np.minimum(nxt, len(nlpos) - 1)], N)
    last = np.full((len(NAMEIDS), N), -1, np.int64)
    inrace = ar > tstart
    for k, nid in enumerate(NAMEIDS):
        ev = np.flatnonzero((t[:-1] == nid) & (t[1:] == EQ) & (ar[:-1] > tstart[:-1]))
        e = nl_next[ev]; okk = e + 1 < N
        a = np.full(N + 1, -1, np.int64); np.maximum.at(a, (e[okk] + 1), ev[okk])
        l = np.maximum.accumulate(a[:N]); l = np.where((l >= 0) & (l >= start), l, -1); last[k] = l
    nid2k = np.full(NV + 8, -1, np.int64)
    for k, nid in enumerate(NAMEIDS):
        nid2k[nid] = k
    def deref(ft, u):
        k = nid2k[ft]; ok = k >= 0; lp = last[np.maximum(k, 0), ar]; q = lp + 1 + u
        return np.where(ok & (lp >= 0) & (q <= ar - 1), t[np.clip(q, 0, N - 1)], 0)
    srcfeat = [(('off%d' % o), F[o - 1]) for o in range(1, 7)] + [('fo%d_%d' % ov, F[i]) for ov, i in fo_feats.items()]
    for nm, ft in srcfeat:
        for u in (1, 2):
            F.append(deref(ft, u)); names.append('D(%s,%d)' % (nm, u))
    # 位置の特徴
    lastnl = np.maximum.accumulate(np.where(t == NL, ar, -1)); sn = np.minimum(ar - np.where(lastnl < start, start - 1, lastnl), 20)
    F.append(sn); names.append('since_NL'); F.append((ar > tstart).astype(np.int64)); names.append('in_trace')
    return np.stack(F, 1).astype(np.int16), names


def main(cfg):
    rng = np.random.default_rng(cfg.get('seed', 0)); t0 = time.time()
    log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    tr = build_set(rng, cfg['ntrain'], 4, 10); te = build_set(rng, cfg['ntest'], 4, 10); lg = build_set(rng, cfg['ntest'], 11, 13)
    log('例 (ソース):\n' + tr[3][0])
    Ftr, names = features(tr[0], tr[1], tr[2]); Fte, _ = features(te[0], te[1], te[2]); Flg, _ = features(lg[0], lg[1], lg[2])
    log('特徴 %d 個' % len(names))
    # 当てる対象 = 記録の中の語 (TRACE の次から)
    def targets(d):
        ar = np.arange(len(d[0])); return np.flatnonzero(ar > d[2])
    ptr, pte, plg = targets(tr), targets(te), targets(lg)
    log('学習の対象 %d, テスト %d, 長い %d' % (len(ptr), len(pte), len(plg)))
    rc = RuleCover(maxk=cfg.get('maxk', 5), beam=cfg.get('beam', 8), purity=cfg.get('purity', 0.995), minn=cfg.get('minn', 3), sample=cfg.get('sample', 80000), rounds=cfg.get('rounds', 30))
    rc.fit(Ftr[ptr], tr[0][ptr].astype(np.int64), names, log=log)
    res = {}
    for nm, d, F_, p_ in (('通常', te, Fte, pte), ('長い', lg, Flg, plg)):
        pred, rid = rc.predict(F_[p_]); y = d[0][p_].astype(np.int64)
        ok = pred == y; cov = pred >= 0
        # 記録の完全一致
        pid = np.searchsorted(np.unique(d[1]), d[1][p_]); bad = np.bincount(pid, weights=(~ok).astype(float)); em = float((bad == 0).mean())
        kinds = {'L': np.array([TOK[v].startswith('L') and v > ID['NL'] and TOK[v][1:].isdigit() for v in y]), 'name': np.isin(y, NAMEIDS), 'digit': np.isin(y, [ID[str(k)] for k in range(10)]), 'NL': y == ID['NL'], '=': y == ID['=']}
        byk = {k: float(ok[m].mean()) for k, m in kinds.items() if m.any()}
        res[nm] = dict(token_acc=float(ok.mean()), covered=float(cov.mean()), acc_when_covered=float(ok[cov].mean()) if cov.any() else None, exact_trace=em, by_kind=byk)
        log('%s: 語の正解率 %.4f (説明できた %.4f, 説明できた中の正解 %.4f) 記録の完全一致 %.4f 種類別 %s' % (nm, ok.mean(), cov.mean(), ok[cov].mean() if cov.any() else 0, em, {k: round(v, 3) for k, v in byk.items()}))
    log('RESULT_G3 ' + json.dumps(dict(cfg=cfg, res=res, rules=[[names[f] for f in s] for s, _ in rc.rules])))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
