"""票の数え上げ (attention の代わりを 数え表だけで作る) の試験。 CPU。
各位置 t の次の語 w の予測 = 過去の語 a_d = T[t-d] (d は 距離) ごとに 票 S_d(w) = log((c_d(a_d, w) + beta*pu(w)) / (tot_d(a_d) + beta)) を集め、
log p(w) ∝ sum_d alpha_d * S_d(w)  (alpha_d は 距離ごとの 重み 15 個だけ 学習する)。 表 c_d は 数えるだけ (勾配なし)。
比較: 同じ位置から数えた 語 1〜3 個の n-gram (Witten-Bell), 混合。
使い方: python3 vote_count.py N_train N_val"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_path import keyof, Counts, wb_chain, tune, bits, ds_all, TRAIN, VAL

V = 32768
DS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128, 256]
BETA = 1.0


def build(Ttrain, ptr_c, ytr, d):
    a = Ttrain[ptr_c - d]; key = a * V + ytr; ks, cs = np.unique(key, return_counts=True); aa = ks // V; ww = ks % V
    tot = np.bincount(aa, weights=cs, minlength=V); st = np.searchsorted(aa, np.arange(V), 'left'); en = np.searchsorted(aa, np.arange(V), 'right')
    return dict(w=ww.astype(np.int64), c=cs.astype(np.float32), tot=tot.astype(np.float32), st=st, en=en)


def votes(tb, a, valid, logpu):
    """位置ごとの 票 S_d (n, V)。 無効な位置 (文書の外) は 0 (全部の語に同じ値 = 何も言わない)"""
    n = len(a); tot = torch.from_numpy(tb['tot'][a]); S = (np.log(BETA) - torch.log(tot + BETA))[:, None].expand(n, V).clone()            # 語の事前確率を 引いた 票 (PMI): 事前確率は 別に 1 回だけ足す
    ln = tb['en'][a] - tb['st'][a]; tl = int(ln.sum())
    if tl:
        off = np.cumsum(ln) - ln; idx = np.arange(tl) - np.repeat(off, ln) + np.repeat(tb['st'][a], ln); rows = np.repeat(np.arange(n), ln); w = tb['w'][idx]; c = torch.from_numpy(tb['c'][idx])
        S = S.clone(); S[rows, w] = torch.log(c + BETA * torch.exp(logpu[w])) - torch.log(tot[rows] + BETA) - logpu[w]
    return S * torch.from_numpy(valid.astype(np.float32))[:, None]


def main():
    N = int(sys.argv[1]); NV = int(sys.argv[2]); Ttr = np.memmap(TRAIN, dtype=np.uint16, mode='r'); Tv = np.fromfile(VAL, dtype=np.uint16).astype(np.int64)
    BL = 4096; nbk = max(N // BL, 1); rng = np.random.default_rng(5); st = np.sort(rng.integers(5000, len(Ttr) - BL - 10, nbk))
    Ttrain = np.concatenate([np.asarray(Ttr[q - 4100:q + BL]).astype(np.int64) for q in st]); ptr_all = np.concatenate([i * (4100 + BL) + 4100 + np.arange(BL) for i in range(nbk)])
    ok = Ttrain[ptr_all] != 0; ptr_c = ptr_all[ok]; ytr = Ttrain[ptr_c]; dst = ds_all(Ttrain)[ptr_c]
    nvb = -(-NV // 4096); rv = np.random.default_rng(777); sv = np.sort(rv.integers(5000, len(Tv) - 4096 - 10, nvb)); pv_all = (sv[:, None] + np.arange(4096)[None]).reshape(-1); pv = pv_all[Tv[pv_all] != 0]; yv = Tv[pv]; dsv = ds_all(Tv)[pv]
    print('学習位置 %d  val 位置 %d' % (len(ptr_c), len(pv)), flush=True); t0 = time.time()
    pu = np.bincount(ytr, minlength=V).astype(np.float64) + 1.0; pu /= pu.sum(); logpu = torch.from_numpy(np.log(pu).astype(np.float32))
    tbs = [build(Ttrain, ptr_c, ytr, d) for d in DS]; print('表を作った (%.0f秒)' % (time.time() - t0), flush=True)
    alpha = torch.nn.Parameter(torch.full((len(DS),), 0.15)); gamma = torch.nn.Parameter(torch.ones(())); opt = torch.optim.Adam([alpha, gamma], lr=0.03)

    def chunk_S(pos, ds_pos):
        return [votes(tbs[k], Tv[pos - d], (pos - d) >= ds_pos, logpu) for k, d in enumerate(DS)]

    # alpha を val の前半で 学習 (15 個だけ)。 評価は 後半だけ
    half = len(pv) // 2; fit = np.arange(0, half, 4)
    for ep in range(6):
        tot = 0.0; nn_ = 0
        for i in range(0, len(fit), 250):
            ii = fit[i:i + 250]; Ss = chunk_S(pv[ii], dsv[ii]); Z = gamma * logpu[None, :] + sum(alpha[k] * Ss[k] for k in range(len(DS)))
            loss = torch.nn.functional.cross_entropy(Z, torch.from_numpy(yv[ii])); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(ii); nn_ += len(ii)
        print('  alpha 学習 %d 回目: 損失 %.4f bit (%.0f秒)' % (ep + 1, tot / nn_ / np.log(2), time.time() - t0), flush=True)
    print('alpha =', np.round(alpha.detach().numpy(), 3).tolist(), ' gamma =', round(float(gamma), 3), flush=True)
    ev = np.arange(half, len(pv)); lp = np.zeros(len(ev))
    with torch.no_grad():
        for i in range(0, len(ev), 250):
            ii = ev[i:i + 250]; Ss = chunk_S(pv[ii], dsv[ii]); Z = gamma * logpu[None, :] + sum(alpha[k] * Ss[k] for k in range(len(DS))); lp[i:i + len(ii)] = torch.log_softmax(Z, 1)[torch.arange(len(ii)), torch.from_numpy(yv[ii])].numpy()
    p_vote = np.exp(lp)
    # 同じ位置の 語 n-gram (1〜3 個) の Witten-Bell
    Tl = []
    for o_ in (1, 2, 3):
        Tl.append(Counts(keyof([Ttrain[ptr_c - k] for k in range(1, o_ + 1)]), ytr).lookup(keyof([Tv[pv - k] for k in range(1, o_ + 1)]), yv)); lt_, _ = tune([tuple(x[ev] for x in lv) for lv in Tl], yv[ev])
    lv_ev = [tuple(x[ev] for x in lv) for lv in Tl]; pt = wb_chain(lv_ev, yv[ev], lam=lt_)
    print('VOTE_RESULT 数える位置 %d (評価 %d 位置): 票の数え上げ %.4f | 語3個 %.4f | 混合 0.5: %.4f | 混合 0.3(票): %.4f' % (len(ptr_c), len(ev), bits(p_vote), bits(pt), bits(0.5 * p_vote + 0.5 * pt), bits(0.3 * p_vote + 0.7 * pt)), flush=True)


if __name__ == '__main__':
    main()
