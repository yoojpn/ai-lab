"""数え上げの索引の 容量の分析 (実測)。 次数 1..8 (単発も残す) の表を作り、
(1) 行数・単発の割合・文脈ごとの行数, (2) 符号化ごとの容量 (今の 12 バイト / 文脈をまとめて差分符号 / 単発は鍵なしの静的関数 / 元の文章+ハッシュ索引),
(3) 「下の次数で すでに予測できる」 単発の行の割合 (1 つ抜き: 自分の出現を除いた下の次数の確率) = 削っても損の小さい行 を測る。
使い方: python3 index_analysis.py TOK_DIR WORK_DIR N_TOKENS    (等間隔 40 ブロックから N_TOKENS 語)"""
import os, sys, time, shutil
import numpy as np
import engine as E

TOK = sys.argv[1]; D = sys.argv[2]; NTOK = int(sys.argv[3]); ORD = (1, 2, 3, 4, 5, 6, 8)
t0 = time.time(); lg = lambda s: print('[%4ds] %s' % (time.time() - t0, s), flush=True); os.makedirs(D, exist_ok=True)
Ttr = np.memmap(TOK + '/train.bin', dtype=np.uint16, mode='r'); NT = len(Ttr); nb = 40; bl = NTOK // nb; st = np.linspace(8, NT - bl - 8, nb).astype(np.int64)
pad = D + '/sub.bin'
with open(pad, 'wb') as f:
    np.zeros(8, np.uint16).tofile(f)
    for s in st: np.asarray(Ttr[s:s + bl]).tofile(f)
nt = nb * bl; T = np.fromfile(pad, dtype=np.uint16); lg('語 %d' % nt)
tabs = {}
for n in ORD:
    pf = '%s/o%d' % (D, n)
    if not os.path.exists(pf + '_k.bin'):
        runs = E.spill(pad, nt, D + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], pf, 1, 20_000_000, lambda s: None); shutil.rmtree(D + '/runs', ignore_errors=True)
    k = np.fromfile(pf + '_k.bin', dtype=np.uint64); c = np.fromfile(pf + '_c.bin', dtype=np.uint32); tabs[n] = (k, c)
lg('表を作った')
gam = lambda x: 2 * np.floor(np.log2(np.maximum(x, 1))) + 1                      # Elias ガンマ符号の長さ (ビット)
tot_cur = tot_grp = tot_sf = 0.0
print('次数 | 行数 | 行/位置 | 単発の行(回数1) | 文脈数 | 1 行だけの文脈 | 今(12B/行) | 文脈まとめ+差分(B/行) | +単発は静的関数(B/行)')
for n in ORD:
    k, c = tabs[n]; E_ = len(k); ctx = k >> np.uint64(16); u, first, cnt = np.unique(ctx, return_index=True, return_counts=True); C_ = len(u)
    one = (cnt == 1); single = (c == 1); only1 = np.repeat(one, cnt)                                  # 1 行だけの文脈の行
    sing_ctx = only1 & single                                                                           # 文脈も 1 回しか出ず、 次の語も 1 つ: 「元の文章の 1 か所」 を指すだけの行
    ef_ctx = np.ceil(np.log2((2.0 ** 48) / C_)) + 2                                                      # 文脈の鍵 (Elias-Fano: 1 文脈あたり ビット)
    tok_ = (k & np.uint64(0xFFFF)).astype(np.int64)
    tokbits = np.where(np.repeat(one, cnt), 15.0, 9.0)                                                  # 次の語: 1 行なら 15 ビット、 複数なら 差分で 平均 9 ビットと仮定
    bits_grp = C_ * ef_ctx + tokbits.sum() + gam(c).sum()
    sf_rows = sing_ctx.sum(); bits_sf = (bits_grp - (sing_ctx * (ef_ctx + 15.0 + 1.0)).sum()) + sf_rows * (15 + 8) * 1.23
    tot_cur += E_ * 12; tot_grp += bits_grp / 8; tot_sf += bits_sf / 8
    print('%d | %d | %.3f | %.3f | %d | %.3f | %.2f MB | %.2f MB (%.1f B/行) | %.2f MB (%.1f B/行)' % (n, E_, E_ / nt, single.mean(), C_, one.mean(), E_ * 12 / 1e6, bits_grp / 8e6, bits_grp / 8 / E_, bits_sf / 8e6, bits_sf / 8 / E_), flush=True)
print('合計 (次数 %s): 今 %.1f MB (%.1f B/語) | 文脈まとめ+差分 %.1f MB (%.1f B/語) | +単発は静的関数 %.1f MB (%.1f B/語) | 元の文章+ハッシュ索引 (全次数) %.1f MB (7 B/語, 推定)' % (ORD, tot_cur / 1e6, tot_cur / nt, tot_grp / 1e6, tot_grp / nt, tot_sf / 1e6, tot_sf / nt, 7 * nt / 1e6))
# (3) 下の次数で すでに予測できる 単発の行 (1 つ抜き)
rng = np.random.default_rng(0); pos = rng.integers(40, nt, 300000); pos = pos[T[8:][pos] != 0] if False else pos; y = T[8 + pos].astype(np.uint64); pos8 = pos + 8
cs = {n: np.concatenate([[0], np.cumsum(tabs[n][1].astype(np.int64))]) for n in ORD}


def look(n, ctxpos, w):
    """次数 n の表で (文脈, 語) の回数 と その文脈の合計"""
    k, c = tabs[n]; h = E.ctx_hash(T, ctxpos, n - 1, n); lo = np.searchsorted(k, h << np.uint64(16)); hi = np.searchsorted(k, (h + np.uint64(1)) << np.uint64(16)); tot = cs[n][hi] - cs[n][lo]
    key = (h << np.uint64(16)) | w; j = np.minimum(np.searchsorted(k, key), len(k) - 1); cw = np.where(k[j] == key, c[j], 0); return cw, tot


print('次数 | 位置のうち その次数の行が単発 (回数1, 文脈も1回) | そのうち 1 つ下の次数が (自分を除いて) 語を 確率 0.5 以上 / 0.2 以上 / 0.05 以上 で予測 | 下の次数が その語を 知らない')
for n in (4, 5, 6, 8):
    cw, tot = look(n, pos8, y); m = (cw == 1) & (tot == 1)
    if m.sum() == 0: continue
    lowern = ORD[ORD.index(n) - 1]; cw2, tot2 = look(lowern, pos8[m], y[m]); p = np.where(tot2 > 1, (cw2 - 1) / np.maximum(tot2 - 1, 1), 0.0)
    print('%d | %.3f (%d 位置) | %.3f / %.3f / %.3f | %.3f' % (n, m.mean(), m.sum(), (p >= 0.5).mean(), (p >= 0.2).mean(), (p >= 0.05).mean(), (p == 0).mean()), flush=True)
lg('終わり')
