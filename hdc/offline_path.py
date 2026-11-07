"""経路の住所で数える試験 (学習なし)。
学習済みの網 (d512c_base.pt) に位置ごとの文脈を通し、 4 層それぞれで選ばれた最上位の釘 (0..65535) の列を 住所 にする。
住所の先頭 j 個 (j=1..4) で 回数表 (住所, 次の語) を作り、 val の位置で バックオフ (Witten-Bell) して 損失と被覆率を測る。
比較: 同じ N 個の学習位置から作った 語の 2 語文脈 / 3 語文脈 の回数表。
使い方: python3 offline_path.py 重み.pt N_train N_val [出力.npz]"""
import sys, time, numpy as np, torch
import exp_ctx as X
import chat_net

import os
TRAIN = os.environ.get('TRAIN', '/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad/dl/tok/train.bin')
VAL = os.environ.get('VAL', '/tmp/tokhf/tok/val.bin')
DEVN = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def ds_local(T, p, back=4096):
    """各位置 p の文書の開始 (0 の語が文書の区切り)。 back 語より前は見ない"""
    out = np.empty(len(p), dtype=np.int64)
    for i, q in enumerate(p):
        lo = max(q - back, 0); w = T[lo:q]; z = np.nonzero(w == 0)[0]
        out[i] = lo + z[-1] + 1 if len(z) else lo
    return out


def collect(net, cfg, T, pos, bs=256):
    c = cfg.get('c', 8); feats = list(cfg.get('feats', [])); offs8 = np.arange(1, c + 1); o1 = [1 + (np.arange(X.NS) * w) // X.NS for w in X.W1]; offs2 = np.arange(c + 1, X.W2 + 1)
    ds = ds_local(T, pos); rows = np.zeros((len(pos), net.L), dtype=np.int64); t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(pos), bs):
            p = pos[i:i + bs]; inp = X.make_inputs(T, p, ds[i:i + bs], feats, offs8, o1, offs2)
            inp = {k_: v_.to(DEVN) for k_, v_ in inp.items()}
            a, g = net.cache(inp); cv = net.ctxv(inp, a); net.hidden(inp, cv)
            for l in range(net.L): rows[i:i + len(p), l] = net._rows[l].view(len(p), -1)[:, 0].cpu().numpy()
            if (i // bs) % 20 == 0: print('  %d/%d  %.0f 位置/秒' % (i, len(pos), (i + len(p)) / (time.time() - t0)), flush=True)
    return rows


def tune(levels, y):
    """段ごとの lam を 座標ごとに 格子で選ぶ (val で調整: 1 段 1 個なので 過剰適合は小さい)"""
    lam = [1.0] * len(levels)
    for _ in range(2):
        for k in range(len(levels)):
            best = None
            for v in (0.25, 0.5, 1, 2, 4, 8, 16, 32):
                l2 = list(lam); l2[k] = v; ls = -np.log2(np.maximum(wb_chain(levels, y, lam=l2), 1e-12)).mean()
                if best is None or ls < best[0]: best = (ls, v)
            lam[k] = best[1]
    return lam, -np.log2(np.maximum(wb_chain(levels, y, lam=lam), 1e-12)).mean()


def keyof(cols):
    """整数の列を 1 つの 64 ビットの住所に (混ぜて 衝突をほぼ無くす)"""
    h = np.zeros(len(cols[0]), dtype=np.uint64)
    for c_ in cols: h = (h * np.uint64(0x9E3779B97F4A7C15) + c_.astype(np.uint64) + np.uint64(1)) * np.uint64(0xBF58476D1CE4E5B9)
    return h


class Counts:
    """住所 -> 次の語の回数。 (住所, 語) を 1 つの 64 ビットの鍵にして並べて数える"""
    def __init__(s, addr, y):
        s.ctx, ci = np.unique(addr, return_inverse=True); s.tot = np.bincount(ci).astype(np.float64)
        pk = keyof([addr, y]); s.pk, s.cnt = np.unique(pk, return_counts=True)
        up = np.unique(ci.astype(np.int64) * 32768 + y.astype(np.int64)); s.kind = np.bincount(up // 32768, minlength=len(s.ctx)).astype(np.float64)   # 住所ごとの 次の語の種類数

    def lookup(s, addr, y):
        """各位置の p(y | 住所) の材料: 合計, 種類数, y の回数, 住所が表にあるか"""
        i = np.minimum(np.searchsorted(s.ctx, addr), len(s.ctx) - 1); seen = s.ctx[i] == addr
        tot = np.where(seen, s.tot[i], 0.0); kind = np.where(seen, s.kind[i], 0.0)
        k = keyof([addr, y]); j = np.minimum(np.searchsorted(s.pk, k), len(s.pk) - 1); cy = np.where(s.pk[j] == k, s.cnt[j], 0).astype(np.float64)
        return tot, kind, cy, seen


def wb_chain(levels, y, V=32768, lam=1.0):
    lam = lam if isinstance(lam, (list, tuple)) else [lam] * len(levels)
    """levels: 粗い順 (短い住所 -> 長い住所) の (tot,kind,cy) 。 Witten-Bell バックオフ。 最下位は 一様"""
    p = np.full(len(y), 1.0 / V)
    for (tot, kind, cy, seen), lm in zip(levels, lam):
        w = tot / (tot + lm * kind + 1e-12); p = np.where(seen, w * cy / np.maximum(tot, 1) + (1 - w) * p, p)
    return p


if __name__ == '__main__':
    net, cfg = chat_net.load(sys.argv[1]); X.DEV = DEVN; net = net.to(DEVN); N = int(sys.argv[2]); NV = int(sys.argv[3]); out = sys.argv[4] if len(sys.argv) > 4 else None
    Ttr = np.memmap(TRAIN, dtype=np.uint16, mode='r'); Tv = np.fromfile(VAL, dtype=np.uint16).astype(np.int64)
    BL = 2000; nbk = max(N // BL, 1); N = nbk * BL; rng = np.random.default_rng(5); st = np.sort(rng.integers(5000, len(Ttr) - BL - 10, nbk))
    segs = [np.asarray(Ttr[q - 4100:q + BL]).astype(np.int64) for q in st]; Ttrain = np.concatenate(segs)                # 学習位置は BL 個ずつの連続ブロック (各ブロックの前に 4100 語の窓)
    ptr_c = np.concatenate([i * (4100 + BL) + 4100 + np.arange(BL) for i in range(nbk)])
    ytr = Ttrain[ptr_c]
    ok = ytr != 0; ptr_c, ytr = ptr_c[ok], ytr[ok]
    rv = np.random.default_rng(777); pv = np.sort(rv.integers(5000, len(Tv) - 10, int(NV * 1.05))); pv = pv[Tv[pv] != 0][:NV]; yv = Tv[pv]
    Rtr = collect(net, cfg, Ttrain, ptr_c, 256); Rv = collect(net, cfg, Tv, pv, 256)
    if out: np.savez(out, Rtr=Rtr, ytr=ytr, Rv=Rv, yv=yv, ptr_c=ptr_c, pv=pv)
    # 経路の住所
    L = net.L; res = {}
    levels = []
    for j in range(1, L + 1):
        atr = keyof([Rtr[:, l] for l in range(j)]); av = keyof([Rv[:, l] for l in range(j)])
        C = Counts(atr, ytr); lv = C.lookup(av, yv); levels.append(lv)
        lam, ls = tune(levels, yv)
        print('経路 先頭 %d 個: 被覆 %.3f  損失 %.4f (バックオフ込み, lam=%s)' % (j, np.mean(lv[3]), ls, lam), flush=True)
    # 語の文脈 (同じ学習位置から)
    lv_t = []
    for o in (1, 2, 3):
        ctr = [Ttrain[ptr_c - k] for k in range(1, o + 1)]; cv_ = [Tv[pv - k] for k in range(1, o + 1)]
        C = Counts(keyof(ctr), ytr); lv = C.lookup(keyof(cv_), yv); lv_t.append(lv); lam, ls = tune(lv_t, yv)
        print('語 %d 個: 被覆 %.3f  損失 %.4f (lam=%s)' % (o, np.mean(lv[3]), ls, lam), flush=True)
    # 一様 (語の分布のみ) の損失
    C = Counts(np.zeros(len(ytr), dtype=np.uint64), ytr); lv = C.lookup(np.zeros(len(yv), dtype=np.uint64), yv); p = wb_chain([lv], yv)
    print('語の頻度のみ: 損失 %.4f' % (-np.log2(np.maximum(p, 1e-12)).mean()))

    # 経路 と 語 の鎖を 混ぜる / 経路 + 語 の結合住所
    Pl = []
    for j in range(1, L + 1): Pl.append(Counts(keyof([Rtr[:, l] for l in range(j)]), ytr).lookup(keyof([Rv[:, l] for l in range(j)]), yv))
    lp_, _ = tune(Pl, yv); pp = wb_chain(Pl, yv, lam=lp_); Tl = []
    for o_ in (1, 2, 3): Tl.append(Counts(keyof([Ttrain[ptr_c - k] for k in range(1, o_ + 1)]), ytr).lookup(keyof([Tv[pv - k] for k in range(1, o_ + 1)]), yv))
    lt_, lst = tune(Tl, yv); pt = wb_chain(Tl, yv, lam=lt_); bits = lambda p_: -np.log2(np.maximum(p_, 1e-12)).mean()
    print('PATH_RESULT 経路4個 %.4f (被覆 %.3f) | 語3個 %.4f (被覆 %.3f) | 混合 0.5: %.4f' % (bits(pp), Pl[-1][3].mean(), lst, Tl[-1][3].mean(), bits(0.5 * pp + 0.5 * pt)), flush=True)
    jl = lambda j, o_: Counts(keyof([Rtr[:, l] for l in range(j)] + [Ttrain[ptr_c - k] for k in range(1, o_ + 1)]), ytr).lookup(keyof([Rv[:, l] for l in range(j)] + [Tv[pv - k] for k in range(1, o_ + 1)]), yv)
    chain = Tl[:2] + [jl(2, 1), jl(L, 1), jl(2, 2), jl(L, 2)]; lc_, lsc_ = tune(chain, yv)
    print('PATH_RESULT 結合の鎖 %.4f' % lsc_, flush=True)
