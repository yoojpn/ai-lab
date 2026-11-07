"""経路の住所で数える試験 (学習なし)。
学習済みの網に 位置ごとの文脈を通し、 4 層それぞれで選ばれた最上位の釘 (0..65535) の列を 住所 にする。
住所の先頭 j 個 (j=1..4) で 回数表 (住所, 次の語) を作り、 val の位置で バックオフ (Witten-Bell) して 損失と被覆率を測る。
比較: 同じ学習位置から作った 語の 1〜3 語文脈の回数表。 経路と語の混合、 経路+語の結合住所の鎖。
再帰層 (lrec) の網は 連続した 512 位置の列として通す (学習と同じ)。 学習位置は BL 位置ずつの連続ブロック、 val は 512 位置の連続ブロック。
使い方: python3 offline_path.py 重み.pt N_train N_val   (環境変数 TRAIN, VAL で語の列の場所を指定)"""
import os, sys, time, numpy as np, torch
import exp_ctx as X
import chat_net

TRAIN = os.environ.get('TRAIN', '/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad/dl/tok/train.bin')
VAL = os.environ.get('VAL', '/tmp/tokhf/tok/val.bin')
DEVN = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SEQ = 512


def ds_all(T):
    """各位置の文書の開始 (0 の語が文書の区切り: 直前の 0 の次)"""
    idx = np.arange(len(T)); lz = np.maximum.accumulate(np.where(T == 0, idx, -1)); ds = np.zeros(len(T), dtype=np.int64); ds[1:] = lz[:-1] + 1; return ds


def collect(net, cfg, T, pos, bs=4096):
    """pos の各位置で 4 層の最上位の釘を集める。 再帰層の網では pos が SEQ 位置ずつ連続している必要がある"""
    c = cfg.get('c', 8); feats = list(cfg.get('feats', [])); dev = DEVN
    offs8 = torch.arange(1, c + 1, device=dev); o1 = [(1 + (torch.arange(X.NS) * w) // X.NS).to(dev) for w in X.W1]; offs2 = torch.arange(c + 1, X.W2 + 1, device=dev)
    Tg = torch.from_numpy(T).to(dev); dsg = torch.from_numpy(ds_all(T)).to(dev); pg = torch.from_numpy(pos).to(dev)
    rows = np.zeros((len(pos), net.L), dtype=np.int64); t0 = time.time(); rec = bool(X.LREC)
    if rec: bs = (bs // SEQ) * SEQ
    with torch.no_grad():
        for i in range(0, len(pos), bs):
            p = pg[i:i + bs]; inp = X.make_inputs_t(Tg, p, dsg[p], feats, offs8, o1, offs2)
            if rec:
                assert len(p) % SEQ == 0; net._seq = (len(p) // SEQ, SEQ)
            a, g = net.cache(inp); cv = net.ctxv(inp, a); net.hidden(inp, cv); net._seq = None
            for l in range(net.L): rows[i:i + len(p), l] = net._rows[l].view(len(p), -1)[:, 0].cpu().numpy()
    print('  集めた %d 位置 (%.0f 位置/秒)' % (len(pos), len(pos) / (time.time() - t0)), flush=True)
    return rows


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
    """levels: 粗い順 (短い住所 -> 長い住所) の (tot,kind,cy,seen) 。 Witten-Bell バックオフ。 最下位は 一様"""
    lam = lam if isinstance(lam, (list, tuple)) else [lam] * len(levels)
    p = np.full(len(y), 1.0 / V)
    for (tot, kind, cy, seen), lm in zip(levels, lam):
        w = tot / (tot + lm * kind + 1e-12); p = np.where(seen, w * cy / np.maximum(tot, 1) + (1 - w) * p, p)
    return p


bits = lambda p_: -np.log2(np.maximum(p_, 1e-12)).mean()


def tune(levels, y):
    """段ごとの lam を 座標ごとに 格子で選ぶ (val で調整: 1 段 1 個なので 過剰適合は小さい)"""
    lam = [1.0] * len(levels)
    for _ in range(2):
        for k in range(len(levels)):
            best = None
            for v in (0.25, 0.5, 1, 2, 4, 8, 16, 32):
                l2 = list(lam); l2[k] = v; ls = bits(wb_chain(levels, y, lam=l2))
                if best is None or ls < best[0]: best = (ls, v)
            lam[k] = best[1]
    return lam, bits(wb_chain(levels, y, lam=lam))


if __name__ == '__main__':
    net, cfg = chat_net.load(sys.argv[1]); X.DEV = DEVN; net = net.to(DEVN); N = int(sys.argv[2]); NV = int(sys.argv[3]); L = net.L
    Ttr = np.memmap(TRAIN, dtype=np.uint16, mode='r'); Tv = np.fromfile(VAL, dtype=np.uint16).astype(np.int64)
    BL = 2048; nbk = max(N // BL, 1); rng = np.random.default_rng(5); st = np.sort(rng.integers(5000, len(Ttr) - BL - 10, nbk))
    Ttrain = np.concatenate([np.asarray(Ttr[q - 4100:q + BL]).astype(np.int64) for q in st])                 # 学習位置は BL 個ずつの連続ブロック (各ブロックの前に 4100 語の窓)
    ptr_all = np.concatenate([i * (4100 + BL) + 4100 + np.arange(BL) for i in range(nbk)])
    nvb = -(-NV // SEQ); rv = np.random.default_rng(777); sv = np.sort(rv.integers(5000, len(Tv) - SEQ - 10, nvb)); pv_all = (sv[:, None] + np.arange(SEQ)[None]).reshape(-1)
    print('lrec=%d  学習位置 %d  val 位置 %d' % (X.LREC, len(ptr_all), len(pv_all)), flush=True)
    Rtr_all = collect(net, cfg, Ttrain, ptr_all); Rv_all = collect(net, cfg, Tv, pv_all)
    ok = Ttrain[ptr_all] != 0; ptr_c, Rtr, ytr = ptr_all[ok], Rtr_all[ok], Ttrain[ptr_all[ok]]                       # 文書の区切り (0) を予測する位置は 除く
    okv = Tv[pv_all] != 0; pv, Rv, yv = pv_all[okv], Rv_all[okv], Tv[pv_all[okv]]; yv_n = len(yv)
    Pl = []
    for j in range(1, L + 1):
        Pl.append(Counts(keyof([Rtr[:, l] for l in range(j)]), ytr).lookup(keyof([Rv[:, l] for l in range(j)]), yv)); lam, ls = tune(Pl, yv)
        print('経路 先頭 %d 個: 被覆 %.3f  損失 %.4f (バックオフ込み, lam=%s)' % (j, Pl[-1][3].mean(), ls, lam), flush=True)
    pp = wb_chain(Pl, yv, lam=lam); Tl = []
    for o_ in (1, 2, 3):
        Tl.append(Counts(keyof([Ttrain[ptr_c - k] for k in range(1, o_ + 1)]), ytr).lookup(keyof([Tv[pv - k] for k in range(1, o_ + 1)]), yv)); lt_, lst = tune(Tl, yv)
        print('語 %d 個: 被覆 %.3f  損失 %.4f (lam=%s)' % (o_, Tl[-1][3].mean(), lst, lt_), flush=True)
    pt = wb_chain(Tl, yv, lam=lt_)
    print('PATH_RESULT 経路4個 %.4f (被覆 %.3f) | 語3個 %.4f (被覆 %.3f) | 混合 0.5: %.4f' % (bits(pp), Pl[-1][3].mean(), bits(pt), Tl[-1][3].mean(), bits(0.5 * pp + 0.5 * pt)), flush=True)
    jl = lambda j, o_: Counts(keyof([Rtr[:, l] for l in range(j)] + [Ttrain[ptr_c - k] for k in range(1, o_ + 1)]), ytr).lookup(keyof([Rv[:, l] for l in range(j)] + [Tv[pv - k] for k in range(1, o_ + 1)]), yv)
    chain = Tl[:2] + [jl(2, 1), jl(L, 1), jl(2, 2), jl(L, 2)]; lc_, lsc_ = tune(chain, yv)
    print('PATH_RESULT 結合の鎖 %.4f  (評価位置 %d)' % (lsc_, yv_n), flush=True)
