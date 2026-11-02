"""R3: 本物の文章で 「同じ型の関係」 が どれくらい出てくるかを数える。
型 (例 " is the capital of") ごとに、 型の直前の語 X と 直後の語 Y の組 (X, Y) を集める (語は トークン 1 つ)。
  回数    = 型が出た回数
  組の種類 = 異なる (X, Y) の数
  1回だけの組の割合 = 組のうち 1 回しか出なかったものの割合
  1回だけの組が占める出現の割合 = 出現のうち その組が 1 回だけの組に当たるものの割合
  Xの種類 / Y の種類
使い方: python3 exp_rel_count.py
"""
import sys, json, collections
import numpy as np
from tokenizers import Tokenizer

W = '/tmp/tok'
TEMPL = [
    ' is the capital of', ' was born in', ' is located in', ' is a type of', ' is the opposite of', ' is the author of',
    ' was founded in', ' is the largest', ' is a member of', ' is the president of', ' is known for', ' is the name of',
    'の首都は', 'の出身', 'の作者は',
]


def main():
    tk = Tokenizer.from_file(W + '/tok.json')
    T = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); n = len(T)
    res = []
    for tm in TEMPL:
        ids = tk.encode(tm).ids; m = len(ids)
        # 一番まれなトークンを探索の目印にする (ここでは先頭以外も含め、 出現数が小さいもの)
        cnt = {}
        for t in set(ids): cnt[t] = 0
        pairs = collections.Counter(); total = 0
        # まず目印を決めるため 小さな標本で数える
        samp = np.asarray(T[:20_000_000]); best = min(set(ids), key=lambda t: int((samp == t).sum())); off = ids.index(best)
        CH = 40_000_000
        for a in range(0, n, CH):
            ch = np.asarray(T[a:min(n, a + CH + m + 2)]).astype(np.int64)
            cand = np.flatnonzero(ch[:CH] == best) - off
            cand = cand[(cand >= 1) & (cand + m + 1 < len(ch))]
            ok = np.ones(len(cand), bool)
            for j, t in enumerate(ids): ok &= ch[cand + j] == t
            c = cand[ok]
            for X, Y in zip(ch[c - 1].tolist(), ch[c + m].tolist()): pairs[(X, Y)] += 1
            total += len(c)
        if total == 0:
            print('%-24s 出現 0' % repr(tm)); res.append(dict(templ=tm, total=0)); continue
        npair = len(pairs); once = sum(1 for v in pairs.values() if v == 1)
        nx = len({x for x, _ in pairs}); ny = len({y for _, y in pairs})
        r = dict(templ=tm, total=total, pairs=npair, once_pairs=once, once_pair_frac=once / npair, occ_in_once=once / total, X=nx, Y=ny)
        res.append(r)
        print('%-24s 出現 %8d | 組の種類 %8d | 1回だけの組 %5.1f%% | 出現のうち1回だけの組 %5.1f%% | Xの種類 %7d | Yの種類 %7d' % (repr(tm), total, npair, 100 * r['once_pair_frac'], 100 * r['occ_in_once'], nx, ny), flush=True)
    print('RESULT_REL ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
