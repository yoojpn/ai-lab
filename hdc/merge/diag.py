import sys
from procs import *
name = sys.argv[1]; n = int(sys.argv[2]); gen = TASKS[name]
tr = dataset(gen, n, 1, 3, 1); va = dataset(gen, 500, 1, 3, 2, exclude=[x for x, _ in tr])
best, table, seps = fit(tr, va); vacc, ns, iname, oname, T, fin, fdec = best
te = dataset(gen, 2000, 16, 20, 120); unk = wrong = 0; ex = []
for x, y in te:
    o = T.run(fin(x))
    if o is None: unk += 1
    elif fdec(o) != y:
        wrong += 1
        if len(ex) < 3: ex.append((x, y, fdec(o)))
print(name, n, '見方', iname, oname, '状態', ns, '遷移', T.n_edges(), '検証', vacc, '| 長さ20: 未知の遷移 %d, 誤答 %d / %d' % (unk, wrong, len(te)), ex)
for q in T.red: print('  状態', q, '遷移数', len(T.tr[q]), '終わりの出力', T.fin[q])
