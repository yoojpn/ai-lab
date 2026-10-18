"""複数ワーカーの学習結果(文脈の数え上げ)を足し合わせて、1つの予測器にする。
使い方: python3 merge_states.py <出力.pkl> <入力1.pkl> <入力2.pkl> ..."""
import os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import forager
main_mod = sys.modules['__main__']
for n in ('State', 'Domain', 'PPM'):
    setattr(main_mod, n, getattr(forager, n))
out, ins = sys.argv[1], sys.argv[2:]
base = None
for path in ins:
    with open(path, 'rb') as f:
        st = pickle.load(f)
    if base is None:
        base = st
        continue
    for k, tab in enumerate(st.model.tabs):
        bt = base.model.tabs[k]
        for key, e in tab.items():
            b = bt.get(key)
            if b is None:
                bt[key] = e
            else:
                for s, c in e[0].items():
                    b[0][s] = b[0].get(s, 0) + c
                b[1] += e[1]
    base.pages += st.pages
    base.bytes += st.bytes
    base.fetches += st.fetches
    base.elapsed += st.elapsed
    for h, d in st.domains.items():
        base.domains.setdefault(h, d)
    del st
base.model.prune(2_000_000)
with open(out, 'wb') as f:
    pickle.dump(base, f, protocol=pickle.HIGHEST_PROTOCOL)
print('結合: %d個 pages=%d contexts=%d' % (len(ins), base.pages, base.model.n_contexts()))
