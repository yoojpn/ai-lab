"""
D2: 下の層を「直近m語の表引き」に置き換えると、品質と計算はどうなるか。
 1. 密モデル(d, nl=4)を普通に学習。
 2. 学習データの窓で、下k層の出力を、各位置の「直近m語」をキーに平均して表にする。
 3. テストでは、全位置について、キーが表にあり(出現cmin回以上)れば表の平均で置き換え、なければ実際に計算(厳密)。
    上の層は普通に計算。 → bit/語 と、置き換えた割合(hit)、節約できる計算量を報告。
節約 = hit * k * (1層のMACs) / (全MACs)。 表引き自体の計算は0とみなし、表の大きさ(MB)も報告する。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import lab
from lab import V, log
import exp_y


def lower(net, x, k):
    n = x.shape[1]; mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
    h = net.E(x) + net.pos[:n]
    for L in net.layers[:k]: h = L(h, src_mask=mask, is_causal=True)
    return h, mask


def upper(net, h, k, mask):
    for L in net.layers[k:net.active]: h = L(h, src_mask=mask, is_causal=True)
    return h


def keys_of(x, m):
    n = x.shape[1]; key = torch.zeros(x.shape[0], n, dtype=torch.long); ok = torch.zeros(x.shape[0], n, dtype=torch.bool)
    for j in range(m):
        sh = torch.nn.functional.pad(x, (j, 0))[:, :n]
        key = key * 1000003 + sh
    ok[:, m - 1:] = True
    return key, ok


def main(cfg):
    net, Xte, yt = exp_y.run(cfg)
    net.eval(); d = cfg.get('d', 128); nl = cfg.get('nl', 4); seq = cfg.get('seq', 64); S = seq + 1
    ctx = lab.load(cfg['N']); stream = np.concatenate(ctx.tr_i)[:cfg['N']]
    M = len(stream) // S; chunks = torch.from_numpy(stream[:M * S].reshape(M, S).astype(np.int64))
    Mw = min(M, cfg.get('Mw', 30000)); n = min(20000, len(Xte)); Xq, yq = Xte[:n], yt[:n]
    lm = 12 * d * d + seq * d; total = nl * lm + d * V
    res = dict(cfg=cfg, macs_total=total, macs_layer=lm); res['rows'] = []
    def bits_of(X, ks, mode):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 500):
                x = X[i:i + 500]; h, mask = lower(net, x, ks)
                if mode is not None:
                    tab, u, cnt, m, cmin = mode; key, ok = keys_of(x, m)
                    pos = torch.searchsorted(u, key.reshape(-1)).clamp(max=len(u) - 1).reshape(key.shape)
                    hit = ok & (u[pos] == key) & (cnt[pos] >= cmin)
                    h = torch.where(hit[..., None], tab[pos], h); stats.append(hit.float().mean().item())
                z = net.out(upper(net, h, ks, mask)[:, -1])
                lp = torch.log_softmax(z, 1); out.append(lp[torch.arange(len(x)), yq[i:i + 500]].exp().numpy())
        return float(-np.log2(np.concatenate(out).astype(np.float64)).mean())
    stats = []
    base = bits_of(Xq, 1, None); res['bits_base'] = base; log('base %.4f' % base)
    for k in (1, 2):
        for m in (2, 3):
            keys, hs = [], []
            with torch.no_grad():
                for i in range(0, Mw, 500):
                    b = chunks[i:i + 500, :-1]; h, _ = lower(net, b, k); kk, ok = keys_of(b, m)
                    keys.append(kk[ok]); hs.append(h[ok])
            keys = torch.cat(keys); H = torch.cat(hs)
            u, inv, cnt = torch.unique(keys, return_inverse=True, return_counts=True)
            tab = torch.zeros(len(u), d).index_add_(0, inv, H) / cnt[:, None]
            for cmin in (1, 3):
                stats.clear(); b = bits_of(Xq, k, (tab, u, cnt, m, cmin)); hit = float(np.mean(stats))
                row = dict(k=k, m=m, cmin=cmin, bits=b, dbits=b - base, hit=hit, saved=hit * k * lm / total, table_mb=len(u) * d * 4 / 1e6, keys=int(len(u)))
                res['rows'].append(row); log('k=%d m=%d cmin=%d bits %.4f (%+.4f) hit %.3f saved %.3f table %.0fMB' % (k, m, cmin, b, b - base, hit, row['saved'], row['table_mb']))
    log('RESULT_TAB ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
