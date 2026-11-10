"""1 語ずつ生成する会話 (数え上げのみ)。 雑談の対話 (RealPersonaChat) で 次数 1..6 の表 (単発も残す) を作り、
「前の発言の語」と「次に出る語」の 結びつき (PPMI) を確率に 足して (最大エントロピー型: P(w) ∝ P_ngram(w) * exp(beta * 結びつき(w))) 返事を 1 語ずつ生成する。
評価: (1) 返事の語の bit (n-gram 単独 / 結びつきを足す), (2) 候補 100 個の順位づけ (返事の対数尤度 - 前の発言なしの対数尤度)、 返事選び (R@1 0.216) と比較。
使い方: python3 rpc_gen.py [発話1 発話2 ...]   (引数があれば 生成の例)"""
import sys, os, json, glob, shutil, time, collections
import numpy as np
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E, chat
from tokenizers import Tokenizer

S = '/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad'; R = S + '/rpc/repo/real_persona_chat/dialogues'; WD = S + '/rpcgen'; os.makedirs(WD, exist_ok=True)
t0 = time.time(); lg = lambda s: print('[%4ds] %s' % (time.time() - t0, s), flush=True)
tk = Tokenizer.from_file('/tmp/tokhf/tok/tok.json'); V = 32768
fs = sorted(glob.glob(R + '/*.json')); dl = [[u['text'] for u in json.load(open(f))['utterances']] for f in fs]; train_d = dl[:-400]; test_d = dl[-400:]
enc = lambda t: tk.encode(t).ids


def stream(D):
    out = [0] * 8; spans = []
    for d in D:
        out.append(0)
        for u in d:
            ids = enc(u); a = len(out); out += ids + [2]; spans.append((a, len(out)))
    return np.array(out, np.uint16), spans


Ttr, _ = stream(train_d); Tte, spte = stream(test_d); lg('学習の語 %d, 検証の語 %d' % (len(Ttr), len(Tte)))
pad = WD + '/tr.bin'; Ttr.tofile(pad); pre = {}
for n in (1, 2, 3, 4, 5, 6):
    pf = '%s/o%d' % (WD, n)
    if not os.path.exists(pf + '_k.bin'):
        runs = E.spill(pad, len(Ttr) - 8, WD + '/runs', [n], 2_000_000, lambda s: None); E.merge(runs[n], pf, 1, 20_000_000, lambda s: None); shutil.rmtree(WD + '/runs', ignore_errors=True)
    pre[n] = pf
lam = {2: 1.0, 3: 2.0, 4: 3.0, 5: 3.0, 6: 3.0}; M = E.Model(pre, smooth='wb', lam=lam, kn_base=True); lg('次数 1-6 の表 済み')
# 結びつき: 前の発言の語 -> 返事の語 (PPMI)
rows = []; cols = []; rows2 = []; cols2 = []; k = 0
for d in train_d:
    for i in range(1, len(d)):
        for g in set(enc(d[i - 1])): rows.append(k); cols.append(g)
        for g in set(enc(d[i]) + [2]): rows2.append(k); cols2.append(g)
        k += 1
X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(k, V)); Y = sparse.csr_matrix((np.ones(len(rows2), np.float32), (rows2, cols2)), shape=(k, V))
C = (X.T @ Y).tocoo(); rs = np.asarray(X.sum(0)).ravel(); cs = np.asarray(Y.sum(0)).ravel(); keep = C.data >= 3
pm = np.log(C.data[keep] * k / (rs[C.row[keep]] * cs[C.col[keep]] + 1e-9)); ok = pm > 0.3
Wm = sparse.csr_matrix((pm[ok].astype(np.float32), (C.row[keep][ok], C.col[keep][ok])), shape=(V, V)); lg('結びつきの表 %d 項目 (%d 組から)' % (Wm.nnz, k))


def assoc(prev_ids):
    ids = list(set(prev_ids))
    if not ids: return np.zeros(V, np.float32)
    return np.asarray(Wm[ids].sum(0)).ravel() / np.sqrt(len(ids))


def dist(T, a, beta, cache=0.0, prev=None):
    P = chat.next_dist(M, T)
    if beta > 0 and a is not None: P = P * np.exp(beta * np.minimum(a, 12.0)); P = P / P.sum()
    return P


def eval_bits(betas=(0.0, 0.05, 0.1, 0.2, 0.3), n=1500, seed=0):
    rng = np.random.default_rng(seed); items = []
    for (a, b) in spte:                                 # 返事 (= 各発話。 最初の発話は 前の発言が無いので除く)
        pass
    # 検証の発話の位置 (a, b): 前の発話 = その 1 つ前の区間
    pos = []
    for j in range(1, len(spte)):
        a, b = spte[j]; pa, pb = spte[j - 1]
        if pb == a and pa > 8 and Tte[pa - 1] != 0 and Tte[a - 1] == 2: pos.append((pa, pb, a, b))
    sel = rng.choice(len(pos), min(n, len(pos)), replace=False); res = {b_: [] for b_ in betas}
    for i in sel:
        pa, pb, a, b = pos[i]; prev = [int(x) for x in Tte[pa:pb - 1]]; av = assoc(prev)
        for t in range(a, b):
            T = Tte[:t]; y = int(Tte[t])
            for b_ in betas:
                P = dist(T, av, b_); res[b_].append(-np.log2(max(P[y], 1e-12)))
    for b_ in betas: lg('GEN_RESULT beta=%.2f: 返事の語 %d 個で %.4f bit/token' % (b_, len(res[b_]), np.mean(res[b_])))


def generate(hist, beta=0.2, temp=0.8, topk=30, maxlen=40, seed=0, norep=True):
    rng = np.random.default_rng(seed); T = [0] * 8 + [0]
    for u in hist: T += enc(u) + [2]
    av = assoc(enc(hist[-1])); gen = []
    for step in range(maxlen):
        P = dist(np.array(T, np.uint16), av, beta); P[0] = 0
        if step < 1: P[2] = 0
        for b in chat.blocked(gen, 3): P[b] = 0
        idx = np.argsort(-P)[:topk]; p = P[idx] ** (1 / temp); p /= p.sum(); x = int(rng.choice(idx, p=p))
        if x == 2: break
        T.append(x); gen.append(x)
    return tk.decode(gen)


if __name__ == '__main__':
    eval_bits()
    for q in sys.argv[1:]:
        print('あなた: %s' % q)
        for s in range(3): print('   返事: %s' % generate([q], seed=s))
