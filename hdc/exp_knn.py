"""刷新の実験 1: 「似た文脈の統計を共有する」仕組み。 勾配なし。
  1. 学習データの一部 (nidx 語) から 単語ベクトル = 周囲の語の共起 (PPMI) の SVD
  2. 文脈ベクトル = 直前 K 語の単語ベクトルの減衰つき和 (正規化)
  3. 学習側の全位置の文脈ベクトルを 近傍検索の索引 (faiss IVF-PQ) に入れ、 次の語を記録 (kNN-LM 型)
  4. 予測: 近い文脈の次の語の重み付き頻度 P_knn と、 同じデータで作った n-gram(1..5) P_lm を、 LM の自信度の区分ごとの割合で混ぜる (検証で格子探索)
  5. 比較: 同じ nidx 語の n-gram / +kNN / 3.8 億語の n-gram (参考)
使い方: python3 exp_knn.py '{"nidx":40000000,"dim":96,"K":8,"ntest":60000}'"""
import sys, os, json, time, shutil
import numpy as np, faiss
from scipy import sparse
from scipy.sparse.linalg import svds
import engine as E

V = 32768


def log(t0):
    return lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)


def embed(S, top, dim, lg):
    freq = np.bincount(S.astype(np.int64), minlength=V); freq[:3] = 0
    ids = np.argsort(-freq, kind='stable')[:top]; ids = ids[freq[ids] > 0]; K = len(ids); rank = np.full(V, -1, np.int64); rank[ids] = np.arange(K)
    r = rank[S.astype(np.int64)]; M = sparse.csr_matrix((K, 2 * K), dtype=np.float32)
    for d in (1, 2, 3):
        for a in range(0, len(r) - d, 20_000_000):
            x = r[a:a + 20_000_000]; y = r[a + d:a + d + 20_000_000]; n = min(len(x), len(y)); x = x[:n]; y = y[:n]; ok = (x >= 0) & (y >= 0); x = x[ok]; y = y[ok]
            M = M + sparse.coo_matrix((np.ones(2 * len(x), np.float32), (np.r_[x, y], np.r_[y, K + x])), shape=(K, 2 * K)).tocsr()
    tot = M.sum(); rs = np.asarray(M.sum(1)).ravel() + 1e-9; cs = np.asarray(M.sum(0)).ravel() ** 0.75; cs = cs / cs.sum() * tot + 1e-9
    C = M.tocoo(); v = np.log(C.data * tot / (rs[C.row] * cs[C.col])); keep = v > 0
    P = sparse.csr_matrix((v[keep], (C.row[keep], C.col[keep])), shape=M.shape); U, Sg, _ = svds(P, k=dim, random_state=0); Em = (U * np.sqrt(Sg)).astype(np.float32)
    Em /= (np.linalg.norm(Em, axis=1, keepdims=True) + 1e-9); emb = np.zeros((V, dim), np.float32); emb[ids] = Em; lg('単語ベクトル %d 語 x %d 次元' % (K, dim)); return emb


def ctxvec(T, idx, emb, K, a, cat=0):
    if cat:                                                                  # 語順を保つ: 直前 cat 語を 連結 (近い語ほど重い)
        parts = [emb[T[idx - k]] * (a ** (k - 1)) for k in range(1, cat + 1)]; out = np.concatenate(parts, axis=1); n = np.linalg.norm(out, axis=1, keepdims=True); return out / np.maximum(n, 1e-9), (n[:, 0] > 1e-9)
    out = np.zeros((len(idx), emb.shape[1]), np.float32); w = 1.0
    for k in range(1, K + 1):
        out += w * emb[T[idx - k]]; w *= a
    n = np.linalg.norm(out, axis=1, keepdims=True); return out / np.maximum(n, 1e-9), (n[:, 0] > 1e-9)


def main(cfg):
    t0 = time.time(); lg = log(t0); W = '/tmp/tok'; D = '/tmp/knn'; os.makedirs(D, exist_ok=True)
    nidx = cfg.get('nidx', 40_000_000); dim = cfg.get('dim', 96); K = cfg.get('K', 8); a = cfg.get('a', 0.75); cat = cfg.get('cat', 0); rng = np.random.default_rng(0)
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = nidx // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]); lg('索引用の学習データ %d 語' % (len(S) - 8))
    emb = embed(S, cfg.get('top', 20000), dim, lg)
    # n-gram (同じ S)
    S.tofile(D + '/sub.bin'); runs = E.spill(D + '/sub.bin', len(S) - 8, D + '/runs', [1, 2, 3, 4, 5], 8_000_000, lambda s: None); MINC = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
    for n in (1, 2, 3, 4, 5): E.merge(runs[n], D + '/final_o%d' % n, MINC[n], 20_000_000, lambda s: None)
    shutil.rmtree(D + '/runs', ignore_errors=True); lam = {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}
    Msmall = E.Model({n: '%s/final_o%d' % (D, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    Mfull = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}, smooth='wb', lam={int(k): v for k, v in json.load(open(W + '/lam.json')).items()}, kn_base=True); lg('n-gram 済み')
    # 索引
    d = dim * cat if cat else dim; nlist = 4096; q = faiss.IndexFlatL2(d); index = faiss.IndexIVFPQ(q, d, nlist, 32 if cat else 16, 8)
    pos_all = np.arange(8 + K, len(S)); trn = pos_all[rng.choice(len(pos_all), 400_000, replace=False)]; xv, _ = ctxvec(S, trn, emb, K, a, cat); index.train(xv); lg('索引の学習済み')
    nxt = np.zeros(len(S), np.uint16); B = 1_000_000; kept = []
    for s in range(8 + K, len(S), B):
        idx = np.arange(s, min(len(S), s + B)); x, ok = ctxvec(S, idx, emb, K, a, cat); index.add(x[ok]); kept.append(S[idx[ok]])
    nxt = np.concatenate(kept); lg('索引に %d 件' % index.ntotal); index.nprobe = cfg.get('nprobe', 16)
    def evalset(sp, n):
        T = np.fromfile('%s/%s.bin' % (W, sp), dtype=np.uint16); p = rng.integers(16, len(T), int(n * 1.02)); p = p[T[p] != 0][:n]; y = T[p].astype(np.int64)
        x, ok = ctxvec(T, p, emb, K, a, cat); Dd, I = index.search(x, 32); return T, p, y, ok, Dd, I
    sets = {sp: evalset(sp, cfg.get('ntest', 60000)) for sp in ('val', 'test')}; lg('検索済み')
    def pknn(y, Dd, I, tau):
        w = np.exp(-(Dd - Dd[:, :1]) / tau); hit = (nxt[I] == y[:, None]); return (w * hit).sum(1) / w.sum(1)
    res = {}
    for sp in ('val', 'test'):
        T, p, y, ok, Dd, I = sets[sp]; res[sp] = dict(lm_small=2.0 ** Msmall.logprob_at(T, p), lm_full=2.0 ** Mfull.logprob_at(T, p))
    # tau と 割合 (LM の自信度の区分ごと) を検証で選ぶ
    GR = np.r_[0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6]
    T_, p_, y_, ok_, Dd_, I_ = sets['val']; best = None
    for tau in (0.005, 0.01, 0.02, 0.05, 0.1):
        pk = pknn(y_, Dd_, I_, tau); plm = res['val']['lm_small']; ll = 0
        b = np.clip((-np.log2(plm) // 2).astype(int), 0, 7)
        for k in range(8):
            m = b == k
            if m.any(): ll += max(np.log2((1 - g) * plm[m] + g * pk[m]).sum() for g in GR)
        if best is None or ll > best[0]: best = (ll, tau)
    tau = best[1]; pk = pknn(y_, Dd_, I_, tau); plm = res['val']['lm_small']; b = np.clip((-np.log2(plm) // 2).astype(int), 0, 7); lam_b = np.zeros(8)
    for k in range(8):
        m = b == k
        if m.any(): lam_b[k] = GR[int(np.argmax([np.log2((1 - g) * plm[m] + g * pk[m]).sum() for g in GR]))]
    T, p, y, ok, Dd, I = sets['test']; pk = pknn(y, Dd, I, tau); plm = res['test']['lm_small']; b = np.clip((-np.log2(plm) // 2).astype(int), 0, 7); g = lam_b[b]
    pm = (1 - g) * plm + g * pk; bits = lambda P: float(-np.log2(P).mean())
    out = dict(nidx=nidx, tau=tau, lam=lam_b.tolist(), lm_small=bits(plm), knn_only=bits(np.maximum(pk, 1e-6)), mix=bits(pm), lm_full=bits(res['test']['lm_full']), n_test=int(len(y)))
    lg('RESULT_KNN ' + json.dumps(out)); json.dump(out, open('/tmp/knn_result_%d_cat%d.json' % (nidx, cat), 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
