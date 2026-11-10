"""人が書いた日本語雑談 (RealPersonaChat, CC BY-SA 4.0) を 数えて 返事を選ぶ会話の試作。 勾配なし・網なし・CPU。
返事は 1 発話まるごと 選ぶ (数語先で話題が流れる問題を 発話単位で避ける)。 選び方 = 「前の発話の文字 2-gram」 と 「返事の文字 2-gram」 の 結びつき (PPMI) を 学習の対話から 数えて 得点にする。
評価: 検証の対話から (前の発話, 本当の返事) を取り、 他の 99 個の返事と混ぜて 順位をつける (R@1: 本当の返事が 1 位になる割合, ランダムは 0.01)。
使い方: python3 rpc_chat.py [発話1 発話2 ...]   (引数があれば 会話の例を出す)"""
import sys, json, glob, math, collections, time
import numpy as np
from scipy import sparse

R = '/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad/rpc/repo/real_persona_chat/dialogues'
t0 = time.time(); lg = lambda s: print('[%4ds] %s' % (time.time() - t0, s), flush=True)
fs = sorted(glob.glob(R + '/*.json')); dl = [[u['text'] for u in json.load(open(f))['utterances']] for f in fs]
test_d = dl[-400:]; train_d = dl[:-400]
pairs = lambda D: [(d[i - 1], d[i], d[i - 2] if i >= 2 else '') for d in D for i in range(1, len(d))]
ptr = pairs(train_d); pte = pairs(test_d); lg('対話 %d (学習 %d / 検証 %d), 学習の組 %d, 検証の組 %d' % (len(dl), len(train_d), len(test_d), len(ptr), len(pte)))
bg = lambda s: [s[i:i + 2] for i in range(len(s) - 1)] + [s[i] for i in range(len(s))]
cnt = collections.Counter(); [cnt.update(set(bg(c)) | set(bg(r))) for c, r, _ in ptr]
vocab = {g: i for i, g in enumerate([g for g, n in cnt.items() if n >= 3])}; V = len(vocab); lg('特徴 (文字 1-gram, 2-gram) %d 個' % V)


def mat(texts):
    rows = []; cols = []
    for i, t in enumerate(texts):
        for g in set(bg(t)):
            j = vocab.get(g)
            if j is not None: rows.append(i); cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(texts), V))


Xc = mat([c for c, _, _ in ptr]); Yr = mat([r for _, r, _ in ptr]); N = Xc.shape[0]
C = (Xc.T @ Yr).tocoo(); rs = np.asarray(Xc.sum(0)).ravel(); cs = np.asarray(Yr.sum(0)).ravel()
keep = C.data >= 2; pm = np.log(C.data[keep] * N / (rs[C.row[keep]] * cs[C.col[keep]] + 1e-9)); ok = pm > 0.5
W = sparse.csr_matrix((pm[ok].astype(np.float32), (C.row[keep][ok], C.col[keep][ok])), shape=(V, V)); lg('結びつきの表 %d 項目' % W.nnz)
idf = np.log(N / (1 + cs)).astype(np.float32)


def scores(ctx, Ycand, alpha=0.5):
    x = mat([ctx]); q = np.asarray((x @ W).todense()).ravel(); ln = np.asarray(Ycand.sum(1)).ravel() + 1
    return (Ycand @ q) / ln ** alpha


def evaluate(K=99, seed=0):
    rng = np.random.default_rng(seed); n = min(3000, len(pte)); idx = rng.choice(len(pte), n, replace=False); allr = [p[1] for p in pte]; res = collections.defaultdict(list)
    for i in idx:
        c, r, _ = pte[i]; neg = [allr[j] for j in rng.choice(len(allr), K, replace=False) if allr[j] != r][:K - 1]; cand = [r] + neg; Y = mat(cand)
        for name, s in (('結びつき PPMI', scores(c, Y)), ('文字の重なり (tf-idf)', np.asarray(((mat([c]).multiply(idf)) @ Y.T).todense()).ravel() / (np.asarray(Y.sum(1)).ravel() + 1) ** 0.5), ('ランダム', rng.random(len(cand)))):
            rank = int((s > s[0]).sum() + 1); res[name].append(rank)
    for name, rk in res.items(): rk = np.array(rk); lg('RPC_RESULT %s: R@1 %.3f  R@5 %.3f  (評価 %d 組, 候補 %d 個)' % (name, (rk == 1).mean(), (rk <= 5).mean(), len(rk), K))


Yall = mat([r for _, r, _ in ptr]); utt = [r for _, r, _ in ptr]


def reply(ctx, k=5, seed=0, avoid=()):
    s = scores(ctx, Yall); top = np.argsort(-s)[:k * 4]; out = [utt[i] for i in top if utt[i] not in avoid and utt[i] != ctx]; return out[:k]


if __name__ == '__main__':
    evaluate()
    for q in sys.argv[1:]:
        print('あなた: %s' % q); [print('   返事候補: %s' % a) for a in reply(q, 3)]
