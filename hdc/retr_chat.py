"""検索型の会話 (数え上げのみ, 勾配なし)。 学習データの「発言 -> 返事」の組から、 いまの発言に近い発言を探して その返事を返す。
  索引: 各組の「発言」の BPE 語を TF-IDF (頻出語は除く) の疎行列に。  検索: 疎行列と問いの内積 -> 上位 k 件 -> 返事を復号。
  使い方: python3 retr_chat.py build '{"np":6000000}'   /   python3 retr_chat.py ask '{"q":["Hello"],"k":5}'"""
import sys, json, time, os
import numpy as np
from scipy import sparse
from tokenizers import Tokenizer

W = '/tmp/tok'; D = '/tmp/retr'; V = 32768


def build(cfg):
    t0 = time.time(); lg = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True); os.makedirs(D, exist_ok=True)
    T = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); sp = []
    for a in range(0, len(T), 50_000_000):
        x = np.asarray(T[a:a + 50_000_000]); sp.append(np.flatnonzero((x == 1) | (x == 2)) + a)
    sp = np.concatenate(sp); tp = np.asarray(T[sp]); lg('区切り %d' % len(sp))
    st = np.r_[-1, sp[:-1]] + 1                                                    # 各発言の開始
    st = st + (np.asarray(T[np.minimum(st, len(T) - 1)]) == 0)                      # 先頭の <bos> を除く
    ln = sp - st; j = np.flatnonzero((tp[:-1] == 2) & (ln[:-1] >= 2) & (ln[:-1] <= 60) & (ln[1:] >= 2) & (ln[1:] <= 80))
    rng = np.random.default_rng(0)
    if len(j) > cfg['np']: j = np.sort(rng.choice(j, cfg['np'], replace=False))
    lg('組 %d' % len(j)); cs, cl, rs, rl = st[j], ln[j], st[j + 1], ln[j + 1]; n = len(j)
    rows = np.repeat(np.arange(n), cl); off = np.arange(cl.sum()) - np.repeat(np.cumsum(cl) - cl, cl); cols = np.asarray(T[np.repeat(cs, cl) + off]).astype(np.int64)
    X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, V)); X.sum_duplicates(); df = np.bincount(X.indices, minlength=V); idf = np.log(n / np.maximum(df, 1)).astype(np.float32); idf[df > 0.02 * n] = 0
    X.data = (1 + np.log(X.data)) * idf[X.indices]; nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-9; X = sparse.diags(1 / nr) @ X; X = X.tocsr().astype(np.float32)
    sparse.save_npz(D + '/X.npz', X); np.savez(D + '/meta.npz', idf=idf, rs=rs, rl=rl, cs=cs, cl=cl); lg('索引 %d 組, 非零 %d' % (n, X.nnz))


def load():
    X = sparse.load_npz(D + '/X.npz'); m = np.load(D + '/meta.npz'); T = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); return X, m, T, Tokenizer.from_file(W + '/tok.json')


def rerank(cands, qids_list, Md):
    """候補の返事を n-gram(会話の表) で採点: 文脈つきの確率 - 文脈なしの確率 (PMI)。 長さで割って 偏りを抑える"""
    import dlg_retr
    parts = [(np.array([0] + qids_list + [2]), np.array(c, np.uint16) if False else np.array(c + [1])) for c in cands]
    null = [(np.array([0, 2]), np.array(c + [1])) for c in cands]
    a = dlg_retr.score(parts, None, Md, None, None, None)['lm']; b = dlg_retr.score(null, None, Md, None, None, None)['lm']; ln = np.array([len(c) + 1 for c in cands])
    return (a - b) / np.sqrt(ln), a / ln


def ask(cfg):
    import engine as E
    X, m, T, tk = load(); idf = m['idf']
    lam = {int(k): v for k, v in json.load(open(W + '/lam.json')).items()}; Md = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    for q in cfg['q']:
        ids = np.array(tk.encode(q).ids); cnt = np.bincount(ids, minlength=V).astype(np.float32); w = np.where(cnt > 0, (1 + np.log(np.maximum(cnt, 1))) * idf, 0); nrm = np.linalg.norm(w)
        if nrm == 0: print('Q:', q, '-> (索引に使える語が無い)'); continue
        s = X @ sparse.csr_matrix((w / nrm)[:, None]); s = s.toarray().ravel(); top = np.argsort(-s)[:cfg.get('pool', 200)]; seen = set(); cand = []; sims = []; txt = []
        for i in top:
            ids_r = np.asarray(T[m['rs'][i]:m['rs'][i] + m['rl'][i]]).astype(int).tolist(); r = tk.decode(ids_r)
            if r in seen: continue
            seen.add(r); cand.append(ids_r); sims.append(float(s[i])); txt.append(r)
        pmi, avg = rerank(cand, ids.tolist(), Md); sims = np.array(sims); final = sims + cfg.get('alpha', 0.12) * pmi
        order = np.argsort(-final)[:cfg.get('k', 5)]
        print('Q:', q)
        for i in order: print('  [sim %.2f pmi %.2f] %s' % (sims[i], pmi[i], txt[i]))


if __name__ == '__main__':
    {'build': build, 'ask': ask}[sys.argv[1]](json.loads(sys.argv[2]))
