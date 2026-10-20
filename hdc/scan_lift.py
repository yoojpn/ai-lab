"""
実験A: 人間が雛形の言語や構文解析の枠を与えずに、SCANの規則を「データの持ち上げ」だけで作る。

使う部品は次の2つだけ(どの課題にも共通):
  (1) 規則 = 「入力の並びの型(定数語と空欄)」→「出力の並びの型(定数トークンと空欄の参照)」
  (2) 書き換え: 規則に合う箇所を値に置き換えていく(合う箇所が複数ある時の優先順位は後でデータから決める)

規則の見つけ方:
  既知の句(意味が分かっている入力の一部)を、その意味が出力に現れる位置ごと「空欄」に置き換える。
  たとえば  walk twice → WALK WALK  で walk=WALK を知っていれば  「X twice → X X」。
  同じ形の持ち上げが何回も(別の句で)出てきた型だけを規則として採用する(= 例の数で確かめる)。
  短い入力から順に学ぶので、長い入力は、すでに見つけた規則で解けるものは新しい規則を要しない。
"""
import itertools
import sys
import time
from collections import Counter, defaultdict

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scan') + '/'


def load(fn):
    out = []
    for line in open(D + fn):
        a, b = line.strip().split(' OUT: ')
        out.append((tuple(a[4:].split()), tuple(b.split())))
    return out


class Rule:
    def __init__(self, pat, tpl, stage):
        self.pat, self.tpl, self.stage = pat, tpl, stage
        self.consts = [x[1] for x in pat if x[0] == 'c']
        self.head = self.consts[0] if self.consts else None
        self.nvar = sum(1 for x in pat if x[0] == 'v')

    def show(self):
        def nm(k):
            return 'XYZ'[k]
        p = ' '.join(x[1] if x[0] == 'c' else nm(x[1]) for x in self.pat)
        t = ' '.join(x[1] if x[0] == 'c' else nm(x[1]) for x in self.tpl)
        return '%s  →  %s' % (p, t)


class System:
    def __init__(self):
        self.lex = {}
        self.rules = []
        self.ranks = None   # head語 → 優先度(データから後で決める)

    def rank(self, r, i):
        base = self.ranks.get(r.head, 0) if self.ranks is not None else 100 - r.stage
        return (base, len(r.consts), -i)

    def rewrite(self, words):
        toks = [('v', self.lex[w]) if w in self.lex else ('w', w) for w in words]
        for _ in range(200):
            best = None
            for ri, r in enumerate(self.rules):
                n = len(r.pat)
                for i in range(len(toks) - n + 1):
                    ok = True
                    for j, item in enumerate(r.pat):
                        t = toks[i + j]
                        if item[0] == 'c':
                            if t[0] != 'w' or t[1] != item[1]:
                                ok = False
                                break
                        elif t[0] != 'v':
                            ok = False
                            break
                    if ok:
                        key = (self.rank(r, ri), -i)
                        if best is None or key > best[0]:
                            best = (key, ri, i)
            if best is None:
                break
            _, ri, i = best
            r = self.rules[ri]
            vals = {}
            for j, item in enumerate(r.pat):
                if item[0] == 'v':
                    vals[item[1]] = toks[i + j][1]
            out = []
            for item in r.tpl:
                if item[0] == 'c':
                    out.append(item[1])
                else:
                    out.extend(vals[item[1]])
            toks[i:i + len(r.pat)] = [('v', tuple(out))]
        return toks

    def run(self, words):
        t = self.rewrite(words)
        if len(t) == 1 and t[0][0] == 'v':
            return t[0][1]
        return None


def decomps(out, ms, cap=6):
    """出力を、定数トークンと空欄(ms[k]の置き換え)に分ける。定数が最少の分け方を全部返す"""
    n = len(out)
    INF = 10 ** 9
    best = [INF] * (n + 1)
    best[n] = 0
    for pos in range(n - 1, -1, -1):
        b = 1 + best[pos + 1]
        for m in ms:
            if out[pos:pos + len(m)] == m:
                b = min(b, best[pos + len(m)])
        best[pos] = b
    res = []

    def go(pos, acc):
        if len(res) >= cap:
            return
        if pos == n:
            res.append(tuple(acc))
            return
        for k, m in enumerate(ms):
            if out[pos:pos + len(m)] == m and best[pos] == best[pos + len(m)]:
                go(pos + len(m), acc + [('v', k)])
        if best[pos] == 1 + best[pos + 1]:
            go(pos + 1, acc + [('c', out[pos])])
    go(0, [])
    return res


def lift(toks, out):
    vals = [t[1] for t in toks if t[0] == 'v']
    if not any(t[0] == 'w' for t in toks) or not vals or len(vals) > 2:
        return []
    if any(len(v) == 0 for v in vals) or len(set(vals)) != len(vals):
        return []
    pat, k = [], 0
    for t in toks:
        if t[0] == 'w':
            pat.append(('c', t[1]))
        else:
            pat.append(('v', k))
            k += 1
    cands = []
    for tpl in decomps(out, vals):
        used = {x[1] for x in tpl if x[0] == 'v'}
        if len(used) == len(vals):
            cands.append((tuple(pat), tpl))
    return cands


def substrings(seq):
    r = {()}
    for i in range(len(seq)):
        for j in range(i + 1, len(seq) + 1):
            r.add(tuple(seq[i:j]))
    return sorted(r, key=len)


def learn(train, log=print, min_support=3, share=0.6):
    S = System()
    ex = defaultdict(list)
    for a, b in train:
        ex[len(a)].append((a, b))
    for a, b in ex[1]:
        S.lex[a[0]] = b
    log('単語1個の例から:', {w: ' '.join(m) for w, m in S.lex.items()})
    stage = 0
    for L in sorted(ex):
        if L == 1:
            continue
        items = ex[L]
        for rnd in range(8):
            ops = {c for r in S.rules for c in r.consts}
            unknown = sorted({w for a, b in items for w in a} - set(S.lex) - ops)
            for u in unknown:
                tests = [(a, b) for a, b in train if u in a and len(a) <= max(L, 4)
                         and all(w == u or w in S.lex or w in ops for w in a)]
                tests.sort(key=lambda x: len(x[0]))
                tests = tests[:30]
                if not tests:
                    continue
                best, bs = None, 0
                for m in substrings(tests[0][1]):
                    S.lex[u] = m
                    s = sum(S.run(a) == b for a, b in tests)
                    if s > bs:
                        best, bs = m, s
                if best is not None and bs >= 0.9 * len(tests):
                    S.lex[u] = best
                    log('  [長さ%d] 語 %s の意味 → %s (例%d/%d個で一致)' % (L, u, ' '.join(best) or '(空)', bs, len(tests)))
                else:
                    S.lex.pop(u, None)
            fails = [(a, b) for a, b in items if S.run(a) != b]
            if not fails:
                break
            cnt = Counter()
            for a, b in fails:
                seen = set()
                for c in lift(S.rewrite(a), b):
                    if c not in seen:
                        seen.add(c)
                        cnt[c] += 1
            bypat = defaultdict(list)
            for (pat, tpl), c in cnt.items():
                bypat[pat].append((c, tpl))
            added = 0
            for pat, lst in bypat.items():
                lst.sort(key=lambda x: -x[0])
                tot = sum(c for c, _ in lst)
                c, tpl = lst[0]
                if c >= min_support and c >= share * tot:
                    if any(r.pat == pat for r in S.rules):
                        continue
                    stage += 1
                    r = Rule(pat, tpl, stage)
                    S.rules.append(r)
                    added += 1
                    log('  [長さ%d] 規則: %s   (例%d個)' % (L, r.show(), c))
            if not added:
                break
    return S


def accuracy(S, data):
    return sum(S.run(a) == b for a, b in data) / len(data)


def tune_ranks(S, train, log=print):
    heads = sorted({r.head for r in S.rules if r.head})
    cand = [(a, b) for a, b in train if len(a) >= 4]
    step = max(1, len(cand) // 150)
    sample = cand[::step][:150]
    S.ranks = {h: 0 for h in heads}
    for r in S.rules:
        S.ranks[r.head] = max(S.ranks[r.head], 0)

    def score():
        return sum(S.run(a) == b for a, b in sample)
    cur = score()
    for _ in range(4):
        improved = False
        for h in heads:
            keep = S.ranks[h]
            for v in range(len(heads)):
                if v == keep:
                    continue
                S.ranks[h] = v
                s = score()
                if s > cur:
                    cur, keep, improved = s, v, True
            S.ranks[h] = keep
        if cur == len(sample) or not improved:
            break
    log('優先度(大きいほど先に結合): %s  / 検証サンプル %d/%d' % (S.ranks, cur, len(sample)))


def main():
    for name in sys.argv[1:] or ['simple', 'length', 'addprim_jump']:
        print('\n=== 分割: %s ===' % name)
        train = load('tasks_train_%s.txt' % name)
        test = load('tasks_test_%s.txt' % name)
        t0 = time.time()
        S = learn(train)
        tune_ranks(S, train)
        dt = time.time() - t0
        print('規則 %d 個 / 語彙 %d 個 / 学習 %.1f 秒' % (len(S.rules), len(S.lex), dt))
        print('訓練の正答率 %.1f%% / 未見の試験の正答率 %.1f%%'
              % (100 * accuracy(S, train), 100 * accuracy(S, test)))


if __name__ == '__main__':
    main()
