"""R3b: ウェブの文章 (FineWeb の標本) で 「同じ型の関係」 を数える。 手元の小説データでは 事実の文が少なすぎたので 取り直し。
型の直前の語 X と 直後の語 Y の組 (X, Y) を 数える (語は 英数字の並び 1 つ)。
使い方: python3 exp_rel_web.py /tmp/fw/sample/10BT/000_00000.parquet [文書数の上限]
"""
import sys, re, json, collections
import pyarrow.parquet as pq

TEMPL = ['is the capital of', 'was born in', 'is located in', 'is a type of', 'is the opposite of', 'is the author of', 'was founded in',
         'is the largest', 'is a member of', 'is the president of', 'is known for', 'is the name of', 'is the founder of', 'is a city in', 'is a village in']
RX = re.compile(r'(\w+) (' + '|'.join(re.escape(t) for t in TEMPL) + r') (\w+)')


def main(path, maxdocs=None):
    pf = pq.ParquetFile(path); pairs = {t: collections.Counter() for t in TEMPL}; nd = 0; nw = 0
    for rg in range(pf.num_row_groups):
        tx = pf.read_row_group(rg, columns=['text']).column('text').to_pylist()
        for t in tx:
            nd += 1; nw += t.count(' ') + 1
            for X, tm, Y in RX.findall(t): pairs[tm][(X, Y)] += 1
            if maxdocs and nd >= maxdocs: break
        if maxdocs and nd >= maxdocs: break
        if rg % 50 == 0: print('  ... 文書 %d, 語 %.1fM' % (nd, nw / 1e6), flush=True)
    print('文書 %d, 語 約 %.1fM' % (nd, nw / 1e6)); res = []
    for tm in TEMPL:
        c = pairs[tm]; total = sum(c.values())
        if not total: print('%-22s 出現 0' % tm); continue
        npair = len(c); once = sum(1 for v in c.values() if v == 1)
        r = dict(templ=tm, total=total, pairs=npair, once_pair_frac=once / npair, occ_in_once=once / total, X=len({x for x, _ in c}), Y=len({y for _, y in c}), top=[(f'{x}→{y}', v) for (x, y), v in c.most_common(3)])
        res.append(r)
        print('%-22s 出現 %7d | 組の種類 %7d | 1回だけの組 %5.1f%% | 出現のうち1回だけの組 %5.1f%% | X %6d | Y %6d | 最多 %s' % (tm, total, npair, 100 * r['once_pair_frac'], 100 * r['occ_in_once'], r['X'], r['Y'], r['top']), flush=True)
    print('RESULT_RELWEB ' + json.dumps(dict(docs=nd, words=nw, res=res), ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
