"""例 (入力列 -> 出力列) から 決定的な変換機械 (subsequential transducer) を 状態の併合だけで組み立てる。 勾配なし・反復学習なし。
OSTIA (Oncina, Garcia, Vidal 1993) の 赤青版。 課題ごとの作り込みは無い: 記号は ただの ハッシュ可能な値。
  T = learn(pairs)            pairs: [(入力のタプル, 出力のタプル), ...]
  T.run(x) -> 出力のタプル または None (未知の遷移)
"""
import sys
sys.setrecursionlimit(100000)


class Fail(Exception):
    pass


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return a[:i]


class Transducer:
    def __init__(self):
        self.tr = [{}]          # tr[q][記号] = [出力タプル, 次の状態]
        self.fin = [None]       # fin[q] = 終わりの出力 (未定義は None)
        self.pre = ()           # 最初に必ず出す出力
        self.red = []

    def run(self, x):
        q = 0; out = list(self.pre)
        for a in x:
            e = self.tr[q].get(a)
            if e is None: return None
            out.extend(e[0]); q = e[1]
        if self.fin[q] is None: return None
        out.extend(self.fin[q]); return tuple(out)

    def n_states(self): return len(self.red)

    def n_edges(self): return sum(len(self.tr[q]) for q in self.red)


def build_ptt(pairs):
    T = Transducer(); tr, fin = T.tr, T.fin
    for x, y in pairs:
        q = 0
        for a in x:
            e = tr[q].get(a)
            if e is None:
                tr.append({}); fin.append(None); e = [(), len(tr) - 1]; tr[q][a] = e
            q = e[1]
        if fin[q] is not None and fin[q] != tuple(y): raise Fail('同じ入力に違う出力 (関数でない)')
        fin[q] = tuple(y)
    # onward 化: 出力を できるだけ 根の側へ 寄せる (後ろから)
    order = []; st = [0]
    while st:
        q = st.pop(); order.append(q)
        for a in tr[q]: st.append(tr[q][a][1])
    pref = {}
    for q in reversed(order):
        for a, e in tr[q].items(): e[0] = pref[e[1]]
        outs = [e[0] for e in tr[q].values()]
        if fin[q] is not None: outs.append(fin[q])
        p = outs[0]
        for o in outs[1:]:
            p = lcp(p, o)
            if not p: break
        k = len(p)
        if k:
            for e in tr[q].values(): e[0] = e[0][k:]
            if fin[q] is not None: fin[q] = fin[q][k:]
        pref[q] = p
    T.pre = pref[0]
    return T


def learn(pairs, max_red=400):
    T = build_ptt(pairs); tr, fin = T.tr, T.fin
    isred = [False] * len(tr); isred[0] = True; red = [0]; log = []

    def set_edge(e, i, v): log.append((e, i, e[i])); e[i] = v

    def set_fin(q, v): log.append((fin, q, fin[q])); fin[q] = v

    def add_edge(q, a, e): log.append((tr[q], a, None)); tr[q][a] = e

    def undo(mark):
        while len(log) > mark:
            o, k, old = log.pop()
            if old is None and isinstance(o, dict): del o[k]
            else: o[k] = old

    def pushback(q, s):          # q は 入る辺が 1 本だけの 木の状態
        if not s: return
        for e in tr[q].values(): set_edge(e, 0, s + e[0])
        if fin[q] is not None: set_fin(q, s + fin[q])

    def fold(p, q):
        if fin[q] is not None:
            if fin[p] is None: set_fin(p, fin[q])
            elif fin[p] != fin[q]: raise Fail()
        for a, eq in list(tr[q].items()):
            ep = tr[p].get(a)
            if ep is None: add_edge(p, a, eq); continue
            wp, tp = ep; wq, tq = eq
            if isred[tp]:
                if wq[:len(wp)] != wp: raise Fail()
                pushback(tq, wq[len(wp):])
            else:
                u = lcp(wp, wq); pushback(tp, wp[len(u):]); pushback(tq, wq[len(u):]); set_edge(ep, 0, u)
            fold(tp, tq)

    while True:
        blue = None
        for r in red:                       # 赤の状態から出る辺の先で、 まだ赤でない状態 (青) を 順に
            for a in sorted(tr[r], key=repr):
                t = tr[r][a][1]
                if not isred[t]: blue = (r, a, t); break
            if blue: break
        if blue is None: break
        r, a, q = blue; e = tr[r][a]; merged = False
        for p in red:
            mark = len(log)
            try:
                set_edge(e, 1, p); fold(p, q); merged = True; break
            except Fail:
                undo(mark)
        if merged: log.clear()
        else:
            isred[q] = True; red.append(q)
            if len(red) > max_red: raise Fail('状態が多すぎる (max_red 超え)')
    T.red = red
    return T
