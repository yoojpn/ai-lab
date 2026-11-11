"""手順の学習の実験。 学習器は 1 つ (ostia.learn + 汎用の「見方」の探索) で、 課題ごとの作り込みは無い。
課題の定義 (TASKS) は データを作るためだけに使い、 学習器には 文字列の組 (入力, 出力) しか渡さない。
見方 (view) = 入力の読み方 6 種 x 出力の書き方 2 種。 区切り記号は データから決める (どの入力にも 高々 1 回しか出ない記号)。
選び方: 学習と同じ長さの 検証例での正答率 → 同点なら 状態の少ないもの。 その後、 学習より長い入力で試す。
使い方: python3 procs.py [学習例の数=3000] [課題名,...]"""
import sys, random, time, json
from ostia import learn, Fail

PAD = '_'


def find_seps(xs):
    mx = {}
    for x in xs:
        c = {}
        for ch in x: c[ch] = c.get(ch, 0) + 1
        for ch, n in c.items(): mx[ch] = max(mx.get(ch, 0), n)
    cand = {ch for ch, n in mx.items() if n <= 1}
    if not cand: return None
    ks = {sum(1 for ch in x if ch in cand) for x in xs}
    return cand if len(ks) == 1 and ks != {0} else None


def columns(x, seps, right):
    fields = [[]]; ops = []
    for ch in x:
        if ch in seps: ops.append(ch); fields.append([])
        else: fields[-1].append(ch)
    L = max(len(f) for f in fields)
    if right: fields = [[PAD] * (L - len(f)) + f for f in fields]
    else: fields = [f + [PAD] * (L - len(f)) for f in fields]
    return [tuple(f[i] for f in fields) + tuple(ops) for i in range(L)]


def in_views(seps):
    V = {'そのまま': lambda x: tuple(x), '逆順': lambda x: tuple(reversed(x))}
    if seps:
        V['桁をそろえて右から'] = lambda x: tuple(reversed(columns(x, seps, True)))
        V['桁をそろえて左から(右詰め)'] = lambda x: tuple(columns(x, seps, True))
        V['頭をそろえて左から'] = lambda x: tuple(columns(x, seps, False))
        V['頭をそろえて右から'] = lambda x: tuple(reversed(columns(x, seps, False)))
    return V


OUT_VIEWS = {'そのまま': (lambda y: tuple(y), lambda o: ''.join(o)), '逆順': (lambda y: tuple(reversed(y)), lambda o: ''.join(reversed(o)))}


def acc(T, fin, fdec, pairs):
    ok = 0
    for x, y in pairs:
        o = T.run(fin(x))
        if o is not None and fdec(o) == y: ok += 1
    return ok / max(1, len(pairs))


def fit(train, val, max_red=400, verbose=False):
    """見方を総当たりして 機械を作り、 検証例で選ぶ。 戻り値: (検証の正答率, 状態数, 入力の見方, 出力の見方, T, fin, fdec, 全候補の表)"""
    seps = find_seps([x for x, _ in train]); best = None; table = []
    for iname, fin in in_views(seps).items():
        for oname, (fenc, fdec) in OUT_VIEWS.items():
            t = time.time()
            try:
                T = learn([(fin(x), fenc(y)) for x, y in train], max_red=max_red); va = acc(T, fin, fdec, val); ns = T.n_states()
            except Fail as e:
                table.append((iname, oname, None, None, time.time() - t)); continue
            table.append((iname, oname, va, ns, time.time() - t))
            key = (-va, ns)
            if best is None or key < best[0]: best = (key, va, ns, iname, oname, T, fin, fdec)
    if best is None: return None, table, seps
    return best[1:], table, seps


# ---------- 課題 (データを作るだけ) ----------
def num(rng, L, base=10):
    if L == 1: return rng.randrange(base)
    return rng.randrange(base ** (L - 1), base ** L)


def tob(n, base):
    if n == 0: return '0'
    s = ''
    while n: s = '0123456789'[n % base] + s; n //= base
    return s


def two(rng, lo, hi, base=10):
    return num(rng, rng.randint(lo, hi), base), num(rng, rng.randint(lo, hi), base)


def t_add(rng, lo, hi): a, b = two(rng, lo, hi); return '%d+%d' % (a, b), str(a + b)
def t_add2(rng, lo, hi): a, b = two(rng, lo, hi, 2); return tob(a, 2) + '+' + tob(b, 2), tob(a + b, 2)
def t_subp(rng, lo, hi):
    a, b = two(rng, lo, hi); a, b = max(a, b), min(a, b); return '%d-%d' % (a, b), str(a - b).zfill(len(str(a)))
def t_subs(rng, lo, hi):
    a, b = two(rng, lo, hi); a, b = max(a, b), min(a, b); return '%d-%d' % (a, b), str(a - b)
def t_mul1(rng, lo, hi): a = num(rng, rng.randint(lo, hi)); d = rng.randrange(10); return '%d*%d' % (a, d), str(a * d)
def t_inc(rng, lo, hi): a = num(rng, rng.randint(lo, hi)); return str(a), str(a + 1)
def t_dbl(rng, lo, hi): a = num(rng, rng.randint(lo, hi)); return str(a), str(2 * a)
def t_rev(rng, lo, hi): s = ''.join(rng.choice('abcde') for _ in range(rng.randint(lo, hi))); return s, s[::-1]
def t_cmp(rng, lo, hi):
    a, b = two(rng, lo, hi)
    if rng.random() < 0.3: b = a
    return '%d?%d' % (a, b), '<' if a < b else ('>' if a > b else '=')
def t_mix(rng, lo, hi): return t_add(rng, lo, hi) if rng.random() < 0.5 else t_subp(rng, lo, hi)
def t_xor(rng, lo, hi):
    a, b = two(rng, lo, hi, 2); L = max(len(tob(a, 2)), len(tob(b, 2))); return tob(a, 2) + '^' + tob(b, 2), tob(a ^ b, 2).zfill(L)
def t_mod3(rng, lo, hi): a = num(rng, rng.randint(lo, hi)); return str(a), str(a % 3)
def t_div1(rng, lo, hi):
    a = num(rng, rng.randint(lo, hi)); d = rng.randrange(1, 10); return '%d/%d' % (a, d), str(a // d).zfill(len(str(a)))
def t_mul(rng, lo, hi): a, b = two(rng, lo, hi); return '%d*%d' % (a, b), str(a * b)
def t_max(rng, lo, hi): a, b = two(rng, lo, hi); return '%d,%d' % (a, b), str(max(a, b))


TASKS = {'足し算(10進)': t_add, '足し算(2進)': t_add2, '引き算(桁数そろえ)': t_subp, '引き算(先頭の0を消す)': t_subs, '1桁を掛ける': t_mul1, '+1': t_inc, '2倍': t_dbl,
         '文字列の逆順': t_rev, '大小比較': t_cmp, '足し算と引き算の混合': t_mix, 'XOR(2進)': t_xor, '3で割った余り': t_mod3, '1桁で割る': t_div1, '大きい方': t_max, '掛け算(一般)': t_mul}


def dataset(gen, n, lo, hi, seed, exclude=()):
    rng = random.Random(seed); d = {}; ex = set(exclude); tries = 0
    while len(d) < n and tries < n * 50:
        tries += 1; x, y = gen(rng, lo, hi)
        if x not in ex: d[x] = y
    return list(d.items())


if __name__ == '__main__':
    NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; names = sys.argv[2].split(',') if len(sys.argv) > 2 else list(TASKS)
    TESTL = [(4, 6), (8, 10), (16, 20)]; res = {}
    for name in names:
        gen = TASKS[name]; t0 = time.time()
        lo, hi = (1, 6) if name in ('足し算(2進)', 'XOR(2進)', '文字列の逆順') else (1, 3)
        tr = dataset(gen, NTR, lo, hi, 1); va = dataset(gen, 500, lo, hi, 2, exclude=[x for x, _ in tr])
        if len(va) < 50: va = dataset(gen, 500, lo, hi, 2)
        best, table, seps = fit(tr, va)
        if best is None: print('%s: どの見方でも 機械を作れず' % name, flush=True); res[name] = None; continue
        vacc, ns, iname, oname, T, fin, fdec = best; tests = []
        for a, b in TESTL:
            k = 2 if lo == 1 and hi == 6 else 1
            te = dataset(gen, 300, a * k, b * k, 100 + b); tests.append((b * k, acc(T, fin, fdec, te)))
        print('%-12s 学習 %d 例 (長さ %d-%d) | 選ばれた見方: 入力=%s 出力=%s | 状態 %d 遷移 %d | 検証 %.3f | 長い入力: %s | %.0fs' % (
            name, len(tr), lo, hi, iname, oname, ns, T.n_edges(), vacc, '  '.join('長さ%d: %.3f' % (L, a) for L, a in tests), time.time() - t0), flush=True)
        res[name] = dict(n_train=len(tr), view_in=iname, view_out=oname, states=ns, edges=T.n_edges(), val=vacc, tests=tests, seps=sorted(seps) if seps else None,
                         table=[(i, o, v, s) for i, o, v, s, _ in table])
    json.dump(res, open('procs_result_%d.json' % NTR, 'w'), ensure_ascii=False, indent=1)
