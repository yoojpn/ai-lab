"""日本語の読解質問応答 (JSQuAD) を 数え上げ + 鍵の自動探索 だけで解く 基準線。 勾配なし・網なし・CPU。
文脈の各文字について 「答えの始まり」 「答えの終わり」 かどうかを 特徴の鍵 (3 個までの組を自動探索) の回数表で予測し、 始まり+終わりの得点が最大の区間を答えにする。
評価: JSQuAD 検証 (EM / 文字 F1, 正解が複数あれば最大)。 使い方: python3 jsq_count.py [N_TRAIN] [N_VAL] [TOPK]"""
import sys, re, itertools, time, unicodedata, collections
import numpy as np, pandas as pd

D = '/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad/jsq/data'
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 12000; NVA = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 6
t0 = time.time(); lg = lambda s: print('[%4ds] %s' % (time.time() - t0, s), flush=True)
tr = pd.read_parquet(D + '/train-00000-of-00001.parquet'); va = pd.read_parquet(D + '/validation-00000-of-00001.parquet')
rng = np.random.default_rng(0); tr = tr.iloc[rng.permutation(len(tr))].reset_index(drop=True); va = va.iloc[rng.permutation(len(va))].reset_index(drop=True)
cnt = collections.Counter(); [cnt.update(c) for c in tr['context'][:20000]]
CH = {c: i + 1 for i, (c, _) in enumerate(cnt.most_common(3000))}
QT = ['何', 'いつ', '誰', 'だれ', 'どこ', 'どの', 'どれ', 'いくつ', 'どう', 'なぜ', 'どんな', 'いくら', '何年', '何人', '何番目', '何を', '何が', '何と', '何に', '何で', 'どちら', 'どのような', '何世紀', '何日', '何月']


def cls(ch):
    o = ord(ch)
    if ch.isdigit(): return 1
    if 0x4E00 <= o <= 0x9FFF: return 2
    if 0x3040 <= o <= 0x309F: return 3
    if 0x30A0 <= o <= 0x30FF: return 4
    if ch.isascii() and ch.isalpha(): return 5
    if unicodedata.category(ch).startswith('P') or ch in '　 ': return 6
    return 7


def qtype(q):
    best = (len(q) + 1, 0)
    for i, w in enumerate(QT):
        p = q.find(w)
        if p >= 0 and (p < best[0] or (p == best[0] and len(w) > len(QT[best[1] - 1]))): best = (p, i + 1)
    return best[1]


def bucket(d):
    return np.digitize(d, [0, 1, 2, 3, 5, 9, 17, 33, 65])


def featurize(ctx, q):
    n = len(ctx); ids = np.array([CH.get(c, 0) for c in ctx], np.int64); cl = np.array([cls(c) for c in ctx], np.int64); pad = lambda a, k: np.r_[a[k:], np.zeros(k, np.int64)] if k > 0 else np.r_[np.zeros(-k, np.int64), a[:n + k]]
    qb = set(q[i:i + 2] for i in range(len(q) - 1)); m = np.array([ctx[i:i + 2] in qb for i in range(n)]); m1 = np.array([c in q for c in ctx])
    pos = np.flatnonzero(m); P = np.arange(n)
    if len(pos):
        j = np.searchsorted(pos, P); dl = P - pos[np.maximum(j - 1, 0)]; dl[j == 0] = 999; dr = pos[np.minimum(j, len(pos) - 1)] - P; dr[j == len(pos)] = 999
    else: dl = np.full(n, 999); dr = np.full(n, 999)
    sid = np.cumsum(np.r_[0, [c == '。' for c in ctx[:-1]]]); sm = np.bincount(sid, weights=m.astype(float))[sid]; sep = ctx.find('[SEP]'); title = (P < (sep if sep >= 0 else 0)).astype(np.int64)
    bi = lambda a, b: (a * 31 + b) % 8191 + 1
    F = [ids, pad(ids, -1), pad(ids, 1), pad(ids, -2), pad(ids, 2), cl, pad(cl, -1), pad(cl, 1), bi(pad(ids, -1), ids), bi(ids, pad(ids, 1)), np.full(n, qtype(q)), m.astype(np.int64) + 2 * m1.astype(np.int64), bucket(dl), bucket(dr), np.digitize(sm, [0, 1, 2, 3, 5, 9]), (P * 8 // max(n, 1)), title]
    return np.stack(F, 1)


FN = ['c0', 'c-1', 'c+1', 'c-2', 'c+2', 'cls0', 'cls-1', 'cls+1', 'bi-', 'bi+', 'qtype', 'inq', 'dleft', 'dright', 'sentmatch', 'relpos', 'title']


def build(df, n):
    X = []; S = []; E = []; idx = []
    for i in range(min(n, len(df))):
        r = df.iloc[i]; ctx = r['context']; a = r['answers']; st = int(a['answer_start'][0]); en = st + len(a['text'][0]) - 1
        if en >= len(ctx) or ctx[st:en + 1] != a['text'][0]: continue
        f = featurize(ctx, r['question']); y1 = np.zeros(len(ctx), np.int64); y2 = np.zeros(len(ctx), np.int64); y1[st] = 1; y2[en] = 1; X.append(f); S.append(y1); E.append(y2); idx.append(np.full(len(ctx), i))
    return np.concatenate(X), np.concatenate(S), np.concatenate(E), np.concatenate(idx)


def ll_subset(key_tr, y_tr, key_va, y_va, a=30.0):
    u, inv = np.unique(key_tr, return_inverse=True); n = np.bincount(inv).astype(float); pos = np.bincount(inv, weights=y_tr); p0 = y_tr.mean(); p = (pos + a * p0) / (n + a)
    j = np.minimum(np.searchsorted(u, key_va), len(u) - 1); seen = u[j] == key_va; pv = np.where(seen, p[j], p0); pv = np.clip(pv, 1e-6, 1 - 1e-6)
    return float((y_va * np.log(pv) + (1 - y_va) * np.log(1 - pv)).mean()), (u, p, p0)


def search(X, y, Xv, yv, kmax=3, top=TOPK, sub_rows=1_500_000):
    nf = X.shape[1]; V = 1 << 14; res = []; r = np.random.default_rng(1); ii = r.choice(len(X), min(sub_rows, len(X)), replace=False); Xs = X[ii]; ys = y[ii]
    for k in range(1, kmax + 1):
        for sub in itertools.combinations(range(nf), k):
            kt = np.zeros(len(Xs), np.int64); kv = np.zeros(len(Xv), np.int64)
            for s in sub: kt = kt * V + Xs[:, s]; kv = kv * V + Xv[:, s]
            ll, _ = ll_subset(kt, ys, kv, yv); res.append((ll, sub))
    res.sort(key=lambda t: -t[0]); return res[:top]


def fit_models(X, y, subs):
    V = 1 << 14; ms = []
    for _, sub in subs:
        k = np.zeros(len(X), np.int64)
        for s in sub: k = k * V + X[:, s]
        u, inv = np.unique(k, return_inverse=True); n = np.bincount(inv).astype(float); pos = np.bincount(inv, weights=y); p0 = y.mean(); ms.append((sub, u, (pos + 30 * p0) / (n + 30), p0))
    return ms


def score(ms, X):
    V = 1 << 14; tot = np.zeros(len(X))
    for sub, u, p, p0 in ms:
        k = np.zeros(len(X), np.int64)
        for s in sub: k = k * V + X[:, s]
        j = np.minimum(np.searchsorted(u, k), len(u) - 1); pv = np.clip(np.where(u[j] == k, p[j], p0), 1e-6, 1 - 1e-6); tot += np.log(pv / (1 - pv))
    return tot / len(ms)


def f1(a, b):
    ca = collections.Counter(a); cb = collections.Counter(b); c = sum((ca & cb).values())
    if c == 0: return 0.0
    p = c / len(a); r = c / len(b); return 2 * p * r / (p + r)


def main():
    X, S, E, _ = build(tr, NTR); lg('学習の文字 %d (%d 問から)' % (len(X), NTR)); nh = len(X) // 5; Xh, Sh, Eh = X[-nh:], S[-nh:], E[-nh:]; Xt, St, Et = X[:-nh], S[:-nh], E[:-nh]
    ss = search(Xt, St, Xh, Sh); lg('始まりの鍵 (上位): ' + ' | '.join('%s %.4f' % ('+'.join(FN[s] for s in sub), ll) for ll, sub in ss[:3]))
    es = search(Xt, Et, Xh, Eh); lg('終わりの鍵 (上位): ' + ' | '.join('%s %.4f' % ('+'.join(FN[s] for s in sub), ll) for ll, sub in es[:3]))
    ms_s = fit_models(X, S, ss); ms_e = fit_models(X, E, es); em = f1s = 0.0; n = 0; ex = []
    for i in range(min(NVA, len(va))):
        r = va.iloc[i]; ctx = r['context']; f = featurize(ctx, r['question']); s_ = score(ms_s, f); e_ = score(ms_e, f); best = (-1e9, 0, 0)
        for L in range(0, 40):
            if L >= len(ctx): break
            sc = s_[:len(ctx) - L] + e_[L:]; j = int(np.argmax(sc))
            if sc[j] > best[0]: best = (sc[j], j, j + L)
        pred = ctx[best[1]:best[2] + 1]; refs = list(r['answers']['text']); em += max(float(pred == g) for g in refs); f1s += max(f1(pred, g) for g in refs); n += 1
        if len(ex) < 5: ex.append((r['question'], pred, refs[0]))
    lg('JSQUAD_RESULT 学習 %d 問 / 検証 %d 問: EM %.3f  文字F1 %.3f' % (NTR, n, em / n, f1s / n))
    for q, p, g in ex: print('  問: %s | 予測: %s | 正解: %s' % (q, p, g))


main()
