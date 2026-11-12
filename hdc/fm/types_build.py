"""返答の型 (潜在変数): ユーザー発話を 語の埋め込みの平均で k-means して 型に分け、 型ごとに 『質問 + 答え』 の n-gram 表を 作る。 数え上げのみ。
出力: /tmp/typ/cent.npy (K, 48), /tmp/typ/z<k>/ (sub.bin, final_o1..5)。 使い方: python3 types_build.py <24M の WORK> [K=8]"""
import sys, os, pickle, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); import engine as E
W24 = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 8; V = 32768; OUT = '/tmp/typ'; os.makedirs(OUT, exist_ok=True)
Em = np.load(W24 + '/E.npy'); uni = np.bincount(np.fromfile(W24 + '/sub.bin', dtype=np.uint16)[8:], minlength=V).astype(float) + 1; u = uni / uni.sum(); idf = np.log(1 / np.maximum(u, 1e-9)); idf = (idf / idf.max()).astype(np.float32)
def uvec(seg):
    v = (Em[seg] * idf[seg][:, None]).sum(0); n = np.linalg.norm(v); return v / n if n > 0 else v
convs = pickle.load(open('/tmp/al/convs.pkl', 'rb')); tr = np.load('/tmp/al/tr_idx.npy'); seen = {}; sel = []
for i in tr:
    k = hash(tuple(convs[i][0][1:13].tolist())); seen[k] = seen.get(k, 0) + 1
    if seen[k] <= 3: sel.append(i)
pairs = []; vecs = []
for i in sel:
    S, st = convs[i]
    for ui in range(0, len(st) - 1, 2):
        if ui + 1 >= len(st): break
        e = st[ui + 2] if ui + 2 < len(st) else len(S); q = S[st[ui]:st[ui + 1]]; a = S[st[ui + 1]:e]
        if len(q) < 3 or len(a) < 3: continue
        pairs.append((q[:80], a)); vecs.append(uvec(q))
X = np.array(vecs, np.float32); print('pairs', len(pairs), flush=True)
rng = np.random.default_rng(0); C = X[rng.choice(len(X), K, replace=False)].copy()
for it in range(20):
    lab = np.argmax(X @ C.T, 1)
    for k in range(K):
        m = lab == k
        if m.any(): c = X[m].mean(0); C[k] = c / max(np.linalg.norm(c), 1e-9)
print('cluster sizes', np.bincount(lab, minlength=K), flush=True); np.save(OUT + '/cent.npy', C)
for k in range(K):
    d = OUT + '/z%d' % k; os.makedirs(d, exist_ok=True)
    arr = [np.zeros(8, np.uint16)]
    for (q, a), l in zip(pairs, lab):
        if l == k: arr += [np.array([0], np.uint16), q.astype(np.uint16), np.array([2], np.uint16), a.astype(np.uint16)]
    arr = np.concatenate(arr); arr.tofile(d + '/sub.bin'); n_tok = len(arr) - 8
    for n in (1, 2, 3, 4, 5):
        if os.path.exists(d + '/final_o%d_k.bin' % n): continue
        runs = E.spill(d + '/sub.bin', n_tok, d + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], d + '/final_o%d' % n, 1 if n <= 2 else 2, 20_000_000, lambda s: None)
        for f in glob.glob(d + '/runs/*'): os.remove(f)
    print('built', k, n_tok, flush=True)
print('TYPES_DONE', flush=True)
