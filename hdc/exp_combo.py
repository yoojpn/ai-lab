"""
(実験F: 統合の第1版 = 小さなNN(司書) + 刈った棚(カード) + 文書内の索引。足し算の効果を測る)
実験D: 記憶の予算(大きさ)と予測精度の曲線。「抽象的な知識(クラス・チャンク)を含む方が、小さな記憶で精度が高いか」。

方法: 数え上げの表から、出現回数が c 回未満の項目を捨てる(c を 1→50 と増やすと記憶が小さくなる)。
      方式ごとに「記憶の大きさ vs テストの bit/語」を並べる。
  W    : 単語4-gramだけ
  W+C  : 単語4-gram + 単語クラス(群)の表 (クラスは 200群 → 表が小さく、1つの項目が多くの出来事を代表する)
  FULL : W+C + チャンク + チャンククラス (=前回のはしご)
  FULL+局所 : FULL + 文書内の繰り返し・索引引き(文脈から作る一時的な索引なので、記憶には数えない)
  NN   : 小さなニューラル(固定サイズ。パラメーター数が記憶の大きさ)  ← 引数で学習量を指定(0なら省略)
記憶の大きさは「保存した項目数×16バイト」で見積もる(キー8+回数4+その他4)。
使い方: LADDER_CORPUS=... python3 exp_budget.py <訓練語数N> [NNのエポック数]
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time, copy
sys.path.insert(0, '/home/claude/hdc')
import numpy as np
from collections import Counter
import exp_ladder as L
import exp_scale as es
import exp_class as ec
import exp_cards as XC
import exp_far as XF
import math, torch

V = es.V; HIST = L.HIST
log = L.log
BYTES = 16


def pruned(model, vy, c):
    new = []
    for (uc, tot, typ, up, pc) in model.t:
        keep = pc >= c
        if not keep.any():
            keep[np.argmax(pc)] = True   # 空の表は扱えないので最大の1項目だけ残す
        up2, pc2 = up[keep], pc[keep]
        uc2, inv = np.unique(up2 // vy, return_inverse=True)
        new.append((uc2, np.bincount(inv, weights=pc2).astype(np.float64),
                    np.bincount(inv).astype(np.float64), up2, pc2))
    m = copy.copy(model)
    m.t = new
    return m


def entries(model):
    return sum(len(t[3]) for t in model.t)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    task = sys.argv[2] if len(sys.argv) > 2 else 'all'   # 'c=<しきい値>' / 'nn' / 'all'
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    K, M, K2 = 200, 1500, 200
    t0 = time.time()
    corp = es.build_corpus()
    cnt = Counter(t for m in corp[2] for t in m)
    vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(V - 1))}
    conv = lambda ms: [np.array([vocab.get(t, 0) for t in m], np.int64) for m in ms]
    tr_i, va_i, te_i = conv(corp[2]), conv(corp[1]), conv(corp[0])
    mods, tot = [], 0
    for a in tr_i:
        if len(a) <= HIST:
            continue
        mods.append(a); tot += len(a) - HIST
        if tot >= N:
            break
    Xtr, ytr, mtr, ptr = XF.lwindows_meta(tr_i, N)
    Xva, yva, mva, pva = XF.lwindows_meta(va_i, es.N_VAL)
    Xte, yte, mte, pte = XF.lwindows_meta(te_i, es.N_TEST)
    log('訓練%d 検証%d テスト%d [%.0fs]' % (len(Xtr), len(Xva), len(Xte), time.time() - t0))
    w3 = lambda X: X[:, HIST - 3:]
    pu = np.bincount(ytr, minlength=V).astype(np.float64) + 1.0
    pu /= pu.sum()
    A = es.NG(w3(Xtr), ytr)
    cls, wc = ec.exchange(mods, K, 6, lambda s: None)
    clsc = np.bincount(cls, weights=wc, minlength=K); wic = np.where(wc > 0, wc / np.maximum(clsc[cls], 1), 1e-9)
    B_ = ec.NGv(cls[w3(Xtr)], cls[ytr], K)
    E_ = L.NG2(cls[w3(Xtr)], ytr, K, V)
    Xf = lambda X: np.stack([cls[X[:, HIST - 3]], cls[X[:, HIST - 2]], X[:, HIST - 1]], axis=1)
    F_ = L.NG2(Xf(Xtr), ytr, V, V)
    lex = L.learn_bpe(mods, M)
    parse, Lmax = L.make_parser(lex)
    Vn = V + M + 1; PAD = V + M
    Utr = L.unit_ctx(Xtr, parse, PAD); Uva = L.unit_ctx(Xva, parse, PAD); Ute = L.unit_ctx(Xte, parse, PAD)
    C_ = L.NG2(Utr, ytr, Vn, V)
    unit_mods = [np.array(parse(tuple(a.tolist())), np.int64) for a in mods]
    ec.V = Vn
    cu, _ = ec.exchange(unit_mods, K2, 6, lambda s: None)
    ec.V = V
    cu = np.append(cu, K2)
    D_ = L.NG2(cu[Utr], ytr, K2 + 1, V)
    log('準備完了 [%.0fs]' % (time.time() - t0))

    # 各モデルを(モデル, 語彙サイズ, 入力変換)で扱う
    specs = [
        ('A', A, V, lambda X, y, U: (w3(X), y), lambda p, X, y: p),
        ('B', B_, K, lambda X, y, U: (cls[w3(X)], cls[y]), lambda p, X, y: p * wic[y]),
        ('E', E_, V, lambda X, y, U: (cls[w3(X)], y), lambda p, X, y: p),
        ('F', F_, V, lambda X, y, U: (Xf(X), y), lambda p, X, y: p),
        ('C', C_, V, lambda X, y, U: (U, y), lambda p, X, y: p),
        ('D', D_, V, lambda X, y, U: (cu[U], y), lambda p, X, y: p),
    ]
    # B の出力語彙は K(クラス)。pruned の vy は各モデルの出力語彙
    vys = {'A': V, 'B': K, 'E': V, 'F': V, 'C': V, 'D': V}

    # 局所(記憶に数えない)
    cp_va, st_va = XC.copy_probs(Xva, yva, pu); cp_te, st_te = XC.copy_probs(Xte, yte, pu)
    ca_va = XC.cache_probs(Xva, yva, pu); ca_te = XC.cache_probs(Xte, yte, pu)
    fc_va = XF.farcache_probs(va_i, mva, pva, yva, 300, pu); fc_te = XF.farcache_probs(te_i, mte, pte, yte, 300, pu)
    lp_va, _ = XF.local_for_windows(va_i, mva, pva, yva, cls, pu); lp_te, _ = XF.local_for_windows(te_i, mte, pte, yte, cls, pu)
    loc_va = [cp_va, ca_va, fc_va] + [lp_va[n] for n in ['L1', 'L2', 'L3', 'S2', 'LC2', 'LCW']]
    loc_te = [cp_te, ca_te, fc_te] + [lp_te[n] for n in ['L1', 'L2', 'L3', 'S2', 'LC2', 'LCW']]
    log('局所部品を計算 [%.0fs]' % (time.time() - t0))

    def probs_of(name, m, X, y, U):
        _, mod, _, inp, post = [s for s in specs if s[0] == name][0]
        x, yy = inp(X, y, U)
        r = m.probs(x, yy)
        r = r[0] if isinstance(r, tuple) else r
        return post(r, X, y)

    base_bits = -np.log2(probs_of('A', A, Xte, yte, Ute)).mean()
    # ---- NN(司書): 直前8語は位置つき、その前の4語は平均 ----
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    DE, NEAR = 64, 8

    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.E = torch.nn.Embedding(V, DE); s.l1 = torch.nn.Linear((NEAR + 1) * DE, H); s.l2 = torch.nn.Linear(H, V)
        def forward(s, X):
            e = s.E(X)
            h = torch.relu(s.l1(torch.cat([e[:, -NEAR:].reshape(len(X), -1), e[:, :-NEAR].mean(1)], 1)))
            return s.l2(h)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Tr, Ytr = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    steps = len(Tr) // 512; st = 0
    for i in torch.randperm(len(Tr)).split(512):
        if len(i) < 512: break
        for g in opt.param_groups: g['lr'] = 2e-3 if st < 0.6 * steps else 1e-3
        loss = torch.nn.functional.cross_entropy(net(Tr[i]), Ytr[i]); opt.zero_grad(); loss.backward(); opt.step(); st += 1
    def nnp(X, y):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                z = net(torch.from_numpy(X[i:i + 4096])); lp = torch.log_softmax(z, 1)
                out.append(lp[torch.arange(len(z)), torch.from_numpy(y[i:i + 4096])].exp().numpy())
        return np.concatenate(out).astype(np.float64)
    nn_v, nn_t = nnp(Xva, yva), nnp(Xte, yte)
    npar = sum(p.numel() for p in net.parameters()); nn_mb = npar * 4 / 1e6
    macs = (NEAR + 1) * DE * H + H * V
    B = lambda p: -np.log2(p).mean()
    log('NN単独: %.3f bit/語 (%+.1f%%) 記憶%.1fMB 1語の積和%.2fM [%.0fs]' % (B(nn_t), 100 * (1 - B(nn_t) / base_bits), nn_mb, macs / 1e6, time.time() - t0))
    def mix(Pv, Pt):
        w = L.em(np.array(Pv)); return B((w[:, None] * np.array(Pt)).sum(0)), w
    b, w = mix([nn_v] + loc_va, [nn_t] + loc_te)
    log('NN + 局所(索引): %.3f (%+.1f%%) 記憶%.1fMB' % (b, 100 * (1 - b / base_bits), nn_mb))
    names = ['A', 'B', 'E', 'F', 'C', 'D']
    for c in (1, 3, 10, 50):
        pm = {n: (pruned(mod, vys[n], c) if c > 1 else mod) for n, mod, *_ in specs}
        pv = [probs_of(n, pm[n], Xva, yva, Uva) for n in names]; pt = [probs_of(n, pm[n], Xte, yte, Ute) for n in names]
        mb = sum(entries(pm[n]) for n in names) * BYTES / 1e6
        b0, _ = mix(pv, pt)
        b1, _ = mix(pv + [nn_v], pt + [nn_t])
        b2, w2 = mix(pv + [nn_v] + loc_va, pt + [nn_t] + loc_te)
        log('棚(c=%d, %.1fMB): 棚のみ %.3f (%+.1f%%) | 棚+NN %.3f (%+.1f%%) | 棚+NN+索引 %.3f (%+.1f%%) 合計記憶 %.1fMB' % (
            c, mb, b0, 100 * (1 - b0 / base_bits), b1, 100 * (1 - b1 / base_bits), b2, 100 * (1 - b2 / base_bits), mb + nn_mb))
    log('\n基準: 単語4-gram(刈らない) %.3f bit/語 = %.1fMB [N=%d, H=%d]' % (base_bits, entries(A) * BYTES / 1e6, N, H))


if __name__ == '__main__':
    main()
