"""
実験3: 「LLMと似たやり方」=誤差を減らす方向に重みを動かす学習(勾配降下)を、CPUだけの小さな
ニューラル言語モデルでやってみる。同じコーパス・同じテスト位置で 4-gram と比べ、計算量も数える。
モデル: 直前3語→埋め込み(64次元)→全結合(256, tanh)→語彙5000のsoftmax。手書きの逆伝播+Adam(numpyのみ)。
"""
import sys, time
import numpy as np
from collections import Counter
sys.path.insert(0, '/home/claude/hdc')
import exp_word as ew

V = ew.V; CTX = ew.CTX; DE = 64; H = 256; B = 512; EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6


def windows(mods):
    X, y = [], []
    for a in mods:
        n = len(a)
        X.append(np.stack([a[CTX - 1 - j:n - 1 - j] for j in range(CTX)][::-1], axis=1))  # 古い語→新しい語
        y.append(a[CTX:])
    return np.concatenate(X), np.concatenate(y)


class Net:
    def __init__(s, rng):
        s.p = dict(E=rng.normal(0, .1, (V, DE)), W1=rng.normal(0, 1 / np.sqrt(CTX * DE), (CTX * DE, H)),
                   b1=np.zeros(H), W2=rng.normal(0, 1 / np.sqrt(H), (H, V)), b2=np.zeros(V))
        s.m = {k: np.zeros_like(v) for k, v in s.p.items()}
        s.v = {k: np.zeros_like(v) for k, v in s.p.items()}
        s.t = 0

    def fwd(s, X):
        p = s.p
        x = p['E'][X].reshape(len(X), -1)
        h = np.tanh(x @ p['W1'] + p['b1'])
        z = h @ p['W2'] + p['b2']
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return x, h, e / e.sum(1, keepdims=True)

    def step(s, X, y, lr=2e-3):
        p = s.p
        x, h, pr = s.fwd(X)
        n = len(X)
        loss = -np.log2(pr[np.arange(n), y] + 1e-12).mean()
        dz = pr; dz[np.arange(n), y] -= 1; dz /= n
        g = {}
        g['W2'] = h.T @ dz; g['b2'] = dz.sum(0)
        da = (dz @ p['W2'].T) * (1 - h * h)
        g['W1'] = x.T @ da; g['b1'] = da.sum(0)
        dx = (da @ p['W1'].T).reshape(n, CTX, DE)
        gE = np.zeros_like(p['E']); np.add.at(gE, X, dx); g['E'] = gE
        s.t += 1
        for k in p:
            s.m[k] = .9 * s.m[k] + .1 * g[k]
            s.v[k] = .999 * s.v[k] + .001 * g[k] ** 2
            mh = s.m[k] / (1 - .9 ** s.t); vh = s.v[k] / (1 - .999 ** s.t)
            p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    def probs_true(s, X, y):
        out = []
        for i in range(0, len(X), 2048):
            out.append(s.fwd(X[i:i + 2048])[2][np.arange(len(X[i:i + 2048])), y[i:i + 2048]])
        return np.concatenate(out)


def main():
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
    def ng_p(Xs, ys):
        return np.array([ng.prob(tuple(map(int, x)), int(y))[0] for x, y in zip(Xs, ys)])
    pva_ng, pte_ng = ng_p(Xva, yva), ng_p(Xte, yte)
    print('訓練%d例 検証%d テスト%d | 4-gram bit/語: 検証%.3f テスト%.3f'
          % (len(Xtr), len(Xva), len(Xte), -np.log2(pva_ng).mean(), -np.log2(pte_ng).mean()), flush=True)
    rng = np.random.default_rng(0)
    net = Net(rng)
    macs_tok = CTX * DE * H + H * V
    for ep in range(1, EPOCHS + 1):
        perm = rng.permutation(len(Xtr))
        tl = []
        for i in range(0, len(perm) - B + 1, B):
            idx = perm[i:i + B]
            tl.append(net.step(Xtr[idx], ytr[idx], lr=2e-3 if ep <= 3 else 1e-3))
        pv = net.probs_true(Xva, yva)
        print('epoch %d 訓練loss=%.3f 検証 bit/語: NN=%.3f  [%.0fs]' % (ep, np.mean(tl), -np.log2(pv + 1e-12).mean(), time.time() - t0), flush=True)
    pv = net.probs_true(Xva, yva); pt = net.probs_true(Xte, yte)
    lam = max(ew.LAMS + [.95], key=lambda l: np.log2((1 - l) * pva_ng + l * pv).sum())
    b_ng = -np.log2(pte_ng); b_nn = -np.log2(pt + 1e-12); b_mx = -np.log2((1 - lam) * pte_ng + lam * pt)
    print('\nテスト bit/語: 4-gram=%.3f  小NN=%.3f  混合(λ=%.2f)=%.3f  (混合の改善 %.1f%%)' % (b_ng.mean(), b_nn.mean(), lam, b_mx.mean(), 100 * (1 - b_mx.mean() / b_ng.mean())))
    print('学習の計算量: 約 %.2f TMAC(積和)= 1語あたり約%.1f MMAC×3(順+逆)×%d語×%dエポック' % (3 * macs_tok * len(Xtr) * EPOCHS / 1e12, macs_tok / 1e6, len(Xtr), EPOCHS))
    print('パラメータ数 %.2f M' % (sum(v.size for v in net.p.values()) / 1e6))


if __name__ == '__main__':
    main()
