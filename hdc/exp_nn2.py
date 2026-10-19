"""
実験4: 計算量を減らす2つの工夫を、小さなニューラル言語モデルに入れて「品質 対 計算量」を測る。
  gate  : 驚き(n-gramが外した単語)だけで学習する。驚きは「その単語自身を除いた」n-gramの予測で測る(無料)。
          対照 rand は、同じ割合をランダムに選ぶ。
  samp  : 出力層(5000語から選ぶ部分、計算の98%)を、学習時だけ「正解+負例256語」に絞る(サンプル化softmax)。
  評価は常に全語彙のsoftmaxで、同じテスト位置。計算量は実際に行った積和の数(順伝播+逆伝播)を数える。
使い方: python3 exp_nn2.py <full|gate|rand|samp|gatesamp> [残す割合=0.4] [エポック数]
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import sys, time, copy
import numpy as np
from collections import Counter
sys.path.insert(0, '/home/claude/hdc')
import exp_word as ew
from exp_nn import windows

V = ew.V; CTX = ew.CTX; DE = 64; H = 256; B = 512
S_NEG = 256


class Net:
    def __init__(s, rng, qlog):
        s.p = dict(E=rng.normal(0, .1, (V, DE)), W1=rng.normal(0, 1 / np.sqrt(CTX * DE), (CTX * DE, H)),
                   b1=np.zeros(H), W2=rng.normal(0, 1 / np.sqrt(H), (H, V)), b2=np.zeros(V))
        s.m = {k: np.zeros_like(v) for k, v in s.p.items()}
        s.v = {k: np.zeros_like(v) for k, v in s.p.items()}
        s.t = 0; s.macs = 0.0; s.qlog = qlog

    def hidden(s, X):
        p = s.p
        x = p['E'][X].reshape(len(X), -1)
        return x, np.tanh(x @ p['W1'] + p['b1'])

    def probs_true(s, X, y):
        out = []
        for i in range(0, len(X), 2048):
            _, h = s.hidden(X[i:i + 2048])
            z = h @ s.p['W2'] + s.p['b2']
            z -= z.max(1, keepdims=True); e = np.exp(z)
            yy = y[i:i + 2048]
            out.append(e[np.arange(len(yy)), yy] / e.sum(1))
        return np.concatenate(out)

    def adam(s, k, g, cols=None, lr=2e-3):
        p = s.p
        if cols is None:
            s.m[k] = .9 * s.m[k] + .1 * g; s.v[k] = .999 * s.v[k] + .001 * g * g
            p[k] -= lr * (s.m[k] / (1 - .9 ** s.t)) / (np.sqrt(s.v[k] / (1 - .999 ** s.t)) + 1e-8)
        else:  # 出力層は使った列だけ更新
            m = s.m[k]; v = s.v[k]
            if k == 'W2':
                m[:, cols] = .9 * m[:, cols] + .1 * g; v[:, cols] = .999 * v[:, cols] + .001 * g * g
                p[k][:, cols] -= lr * (m[:, cols] / (1 - .9 ** s.t)) / (np.sqrt(v[:, cols] / (1 - .999 ** s.t)) + 1e-8)
            else:
                m[cols] = .9 * m[cols] + .1 * g; v[cols] = .999 * v[cols] + .001 * g * g
                p[k][cols] -= lr * (m[cols] / (1 - .9 ** s.t)) / (np.sqrt(v[cols] / (1 - .999 ** s.t)) + 1e-8)

    def step(s, X, y, rng, sampled, lr):
        p = s.p; n = len(X)
        x, h = s.hidden(X)
        if sampled:
            neg = rng.choice(V, S_NEG, p=s.qprob)
            cols = np.unique(np.concatenate([y, neg]))
            z = h @ p['W2'][:, cols] + p['b2'][cols] - (s.qlog[cols] + np.log(S_NEG))
            tgt = np.searchsorted(cols, y)
            nc = len(cols)
        else:
            cols = None
            z = h @ p['W2'] + p['b2']; tgt = y; nc = V
        z -= z.max(1, keepdims=True); e = np.exp(z); pr = e / e.sum(1, keepdims=True)
        dz = pr; dz[np.arange(n), tgt] -= 1; dz /= n
        W2c = p['W2'] if cols is None else p['W2'][:, cols]
        gW2 = h.T @ dz; gb2 = dz.sum(0)
        da = (dz @ W2c.T) * (1 - h * h)
        gW1 = x.T @ da; gb1 = da.sum(0)
        dx = (da @ p['W1'].T).reshape(n, CTX, DE)
        gE = np.zeros_like(p['E']); np.add.at(gE, X, dx)
        s.t += 1
        s.adam('E', gE, lr=lr); s.adam('W1', gW1, lr=lr); s.adam('b1', gb1, lr=lr)
        s.adam('W2', gW2, cols, lr); s.adam('b2', gb2, cols, lr)
        s.macs += 3.0 * n * (CTX * DE * H + H * nc)


def main():
    mode = sys.argv[1]; frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    t0 = time.time()
    tr, va, te = ew.corpus()
    tr = ew.take(tr, ew.N_TRAIN); va = ew.take(va, ew.N_VAL); te = ew.take(te, ew.N_TEST)
    cnt = Counter(t for m in tr for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr_i, va_i, te_i = conv(tr), conv(va), conv(te)
    Xtr, ytr = windows(tr_i); Xva, yva = windows(va_i); Xte, yte = windows(te_i)
    ng = ew.NGram()
    for a in tr_i:
        ng.train(a)
    ngp = lambda Xs, ys: np.array([ng.prob(tuple(map(int, x)), int(y))[0] for x, y in zip(Xs, ys)])
    pva_ng, pte_ng = ngp(Xva, yva), ngp(Xte, yte)

    def loo(ctx, w):
        p = 1.0 / V
        for k in range(CTX + 1):
            e = ng.t[k].get(ctx[CTX - k:] if k else ())
            cnt_, tot = e
            c = cnt_.get(w, 0) - 1; tot2 = tot - 1
            if tot2 <= 0:
                break
            d = len(cnt_) - (1 if c == 0 else 0)
            p = max(c - ew.DELTA, 0) / tot2 + ew.DELTA * d / tot2 * p
        return p
    sur = None
    if mode in ('gate', 'gatesamp'):
        sur = -np.log2(np.array([loo(tuple(map(int, x)), int(y)) for x, y in zip(Xtr, ytr)]))
        keep_gate = np.argsort(-sur)[:int(frac * len(Xtr))]
    rng = np.random.default_rng(0)
    uni = np.bincount(ytr, minlength=V).astype(float) + 1
    q = uni ** .75; q /= q.sum()
    net = Net(rng, np.log(q)); net.qprob = q
    sampled = mode in ('samp', 'gatesamp')
    print('モード=%s 割合=%.2f | 4-gram bit/語 テスト%.3f | 準備%.0fs' % (mode, frac, -np.log2(pte_ng).mean(), time.time() - t0), flush=True)
    best = (9e9, None, 0, 0)
    for ep in range(1, epochs + 1):
        if mode in ('gate', 'gatesamp'):
            idx_all = rng.permutation(keep_gate)
        elif mode == 'rand':
            idx_all = rng.permutation(len(Xtr))[:int(frac * len(Xtr))]
        else:
            idx_all = rng.permutation(len(Xtr))
        lr = 2e-3 if ep <= 3 else 1e-3
        for i in range(0, len(idx_all) - B + 1, B):
            ix = idx_all[i:i + B]
            net.step(Xtr[ix], ytr[ix], rng, sampled, lr)
        pv = net.probs_true(Xva, yva)
        bv = -np.log2(pv + 1e-12).mean()
        lam = max(ew.LAMS + [.95], key=lambda l: np.log2((1 - l) * pva_ng + l * pv).sum())
        bm = -np.log2((1 - lam) * pva_ng + lam * pv).mean()
        print('epoch %d 累計計算量=%.2f TMAC | 検証 bit/語: NN単独=%.3f 混合=%.3f(λ=%.2f) [%.0fs]' % (ep, net.macs / 1e12, bv, bm, lam, time.time() - t0), flush=True)
        if bm < best[0]:
            best = (bm, copy.deepcopy(net.p), lam, net.macs)
    net.p = best[1]; lam = best[2]
    pt = net.probs_true(Xte, yte)
    b_ng = -np.log2(pte_ng).mean(); b_nn = -np.log2(pt + 1e-12).mean()
    b_mx = -np.log2((1 - lam) * pte_ng + lam * pt).mean()
    print('\n[%s] テスト bit/語: 4-gram=%.3f NN単独=%.3f 混合=%.3f (改善%.1f%%) | 選んだ時点の計算量=%.2f TMAC'
          % (mode, b_ng, b_nn, b_mx, 100 * (1 - b_mx / b_ng), best[3] / 1e12))


if __name__ == '__main__':
    main()
