"""会話の 返答の 語の 予測 (ローカル 24M 索引 + 特徴): 基準 / 文書特徴 / アライメント帰納 / 会話だけの n-gram (PoE) の 重みを 数え上げ (座標降下) で 決めて 検証。
使い方: python3 dlg_poe.py <24M の WORK> <会話だけの WORK (sub.bin と final_o*)> <align.pkl>"""
import sys, os, pickle, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import align_copy as A, chat2 as C
from tokenizers import Tokenizer
V = 32768
W24, WD, ALP = sys.argv[1:4]; NF = 4
al = pickle.load(open(ALP, 'rb')); A.D = len(al.S)
convs = pickle.load(open('/tmp/al/convs.pkl', 'rb')); rng = np.random.default_rng(0); te = [convs[i] for i in np.load('/tmp/al/te_clean_idx.npy')]
tk = Tokenizer.from_file('/tmp/tokhf/tok/tok.json'); Em = np.load(W24 + '/E.npy'); lam = np.load(W24 + '/combo_lam.npy')[:9]
uni = np.bincount(np.fromfile(W24 + '/sub.bin', dtype=np.uint16)[8:], minlength=V).astype(float) + 1; ch = C.Chat2(None, Em, uni, lam, np.load(W24 + '/cls64.npy'), tk)
db = C.local_dist(W24, None); dd = C.local_dist(WD, None)
CENT = np.load('/tmp/typ/cent.npy'); KT = len(CENT); dts = [C.local_dist('/tmp/typ/z%d' % k, None) for k in range(KT)]; idf = ch.idf
def uvec(seg):
    v = (Em[seg] * idf[seg][:, None]).sum(0); n = np.linalg.norm(v); return v / n if n > 0 else v
NPOS = int(os.environ.get('NPOS', 2400)); pos = []
for ci, (S, starts) in enumerate(te):
    for ui in range(1, len(starts)):
        s0 = starts[ui]; e = starts[ui + 1] if ui + 1 < len(starts) else len(S)
        if ui % 2 == 1:
            for t in range(s0 + 1, e, 7): pos.append((ci, t))
rng.shuffle(pos); pos = pos[:NPOS]; print('positions', len(pos), flush=True)
Pb = np.zeros((len(pos), V), np.float32); F = np.zeros((NF, len(pos), V), np.float16); Y = np.zeros(len(pos), np.int64); t0 = time.time()
for k, (ci, t) in enumerate(pos):
    S = te[ci][0]; ctx = S[:t].tolist(); Y[k] = S[t]; p = db(ctx); Pb[k] = p; F[0, k] = ch.score(ctx)
    he = te[ci][1][[i for i in range(len(te[ci][1])) if te[ci][1][i] <= t][-1]]; jj, sc = al.cand_scores(S[:t], he); Fa = np.zeros(V); np.add.at(Fa, S[:t][jj], np.exp(sc)); F[1, k] = np.log1p(Fa)
    F[2, k] = np.clip(np.log(np.maximum(dd(ctx), 1e-12)) - np.log(np.maximum(p, 1e-12)), -8, 8)
    stt = te[ci][1]; ui = max(i for i in range(len(stt)) if stt[i] <= t); z = int(np.argmax(CENT @ uvec(S[stt[ui - 1]:stt[ui]]))) if ui >= 1 else 0; pz = np.maximum(dts[z](ctx), 1e-12); pg = np.maximum(dd(ctx), 1e-12); F[3, k] = np.clip(np.log(pz) - np.log(pg), -3, 3)
    if k % 300 == 0: print(k, time.time() - t0, flush=True)
half = len(pos) // 2; tu = slice(0, half); tt = slice(half, len(pos))


def bits(w, sl, ws):
    tot = 0.0
    for k in range(sl.start, sl.stop):
        s = sum(w[i] * F[i, k].astype(np.float32) for i in range(NF) if w[i] != 0) if any(w) else np.zeros(V, np.float32)
        e = Pb[k] * np.exp(s - s.max()); z = e.sum(); tot -= np.log2(max(e[Y[k]] / z, 1e-30))
    return tot / (sl.stop - sl.start)
w = [0.0] * NF; print('基準 tune/test bits', bits(w, tu, w), bits(w, tt, w), flush=True)
for sweep in range(2):
    for i in range(NF):
        best = (bits(w, tu, w), w[i])
        for v in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            w2 = list(w); w2[i] = v; b = bits(w2, tu, w2)
            if b < best[0]: best = (b, v)
        w[i] = best[1]; print(sweep, i, w, best[0], flush=True)
print('最終 重み', w, 'tune', bits(w, tu, w), 'test', bits(w, tt, w), flush=True)
for name, ww in (('文書のみ', [w[0], 0, 0, 0]), ('アライメントのみ', [0, w[1], 0, 0]), ('会話n-gramのみ', [0, 0, w[2], 0]), ('文書+アライメント', [w[0], w[1], 0, 0]), ('型のみ', [0, 0, 0, w[3]]), ('型なし', [w[0], w[1], w[2], 0]), ('型+文書+アライメント', [w[0], w[1], 0, w[3]])):
    print(name, 'test', bits(ww, tt, ww), flush=True)
