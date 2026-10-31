"""
大規模 n-gram 集計エンジン (あふれる前に書き出して、 あとで併合する方式)。 勾配なし・数え上げのみ。
  語: ByteLevel BPE (サブワード, 語彙 32768)。 文字列・コメント・数値も全部 予測する。 1 語 = uint16。
  集計: 語列を chunk ごとに処理 -> 次数 n の キー = (文脈ハッシュ 48bit << 16 | 次の語) を 並べ替えて 数え -> 「ラン」としてディスクへ。
  併合: キー空間を R 個の範囲に分け、 範囲ごとに 全ランの該当部分だけ読み込んで 合計 -> しきい値未満を捨てる -> 最終表 (ソート済み)。
        (メモリは 1 範囲ぶんだけ。 範囲の数 R を増やせば どんな大きさでも同じ手順)
  確率: Witten-Bell 補間 (次数 1..N)。 文脈ごとの総数 N と 種類数 T は 最終表から作る。
  捨てたもの: 次数 n の出現が minc 未満の項目 (確率の質量は 低次に回る)。
"""
import os, sys, json, time, glob, hashlib
import numpy as np

VOC = 32768
M48 = np.uint64((1 << 48) - 1)
C1 = np.uint64(0x9E3779B97F4A7C15)


def ctx_hash(tok, idx, nctx, seed):
    """tok: uint16 配列, idx: 予測する位置の配列。 文脈 = 直前 nctx 語のハッシュ (48bit)"""
    h = np.full(len(idx), np.uint64(seed) * C1, np.uint64)
    for k in range(nctx, 0, -1):
        t = tok[idx - k].astype(np.uint64) + np.uint64(1)
        h = (h ^ t) * C1
        h ^= h >> np.uint64(29)
    return h & M48


def make_keys(tok, lo, hi, n):
    """位置 [lo,hi) を次語とする次数 n のキー。 tok は先頭に n-1 語以上の余白がある前提"""
    idx = np.arange(lo, hi)
    h = ctx_hash(tok, idx, n - 1, n)
    return (h << np.uint64(16)) | tok[idx].astype(np.uint64)


def spill(tok_path, ntok, run_dir, orders, chunk, log):
    """語列 (uint16 の memmap, 先頭に 8 語の余白(BOS)) を chunk ごとに数えて ラン を書く"""
    tok = np.memmap(tok_path, dtype=np.uint16, mode='r'); runs = {n: [] for n in orders}; os.makedirs(run_dir, exist_ok=True)
    pad = 8; r = 0; t0 = time.time()
    for lo in range(pad, pad + ntok, chunk):
        hi = min(lo + chunk, pad + ntok)
        for n in orders:
            k = make_keys(tok, lo, hi, n); k.sort()
            first = np.r_[True, k[1:] != k[:-1]]; u = k[first]; c = np.diff(np.r_[np.flatnonzero(first), len(k)]).astype(np.uint32)
            pk = '%s/o%d_r%d_k.npy' % (run_dir, n, r); pc = '%s/o%d_r%d_c.npy' % (run_dir, n, r)
            np.save(pk, u); np.save(pc, c); runs[n].append((pk, pc))
        r += 1
    log('ラン %d 本 (次数 %s) %.1f 秒' % (r, list(orders), time.time() - t0))
    return runs


def merge(runs_n, out_prefix, minc, max_range_entries, log):
    ks = [np.load(a, mmap_mode='r') for a, _ in runs_n]; cs = [np.load(b, mmap_mode='r') for _, b in runs_n]
    total = sum(len(k) for k in ks); R = max(1, int(np.ceil(total / max_range_entries)))
    # 範囲の境界: 各ランから等間隔に標本
    samp = np.sort(np.concatenate([np.asarray(k[::max(1, len(k) // 20000)]) for k in ks]))
    cuts = [np.uint64(0)] + [samp[int(len(samp) * i / R)] for i in range(1, R)] + [None]
    fk = open(out_prefix + '_k.bin', 'wb'); fc = open(out_prefix + '_c.bin', 'wb'); nkeep = 0
    for r in range(R):
        a, b = cuts[r], cuts[r + 1]; parts_k = []; parts_c = []
        for k, c in zip(ks, cs):
            i0 = np.searchsorted(k, a, side='left') if r > 0 else 0
            i1 = np.searchsorted(k, b, side='left') if b is not None else len(k)
            if i1 > i0:
                parts_k.append(np.asarray(k[i0:i1])); parts_c.append(np.asarray(c[i0:i1]))
        if not parts_k:
            continue
        kk = np.concatenate(parts_k); cc = np.concatenate(parts_c).astype(np.uint64)
        o = np.argsort(kk, kind='stable'); kk = kk[o]; cc = cc[o]
        first = np.r_[True, kk[1:] != kk[:-1]]; st = np.flatnonzero(first); u = kk[st]; s = np.add.reduceat(cc, st)
        keep = s >= minc; u = u[keep]; s = np.minimum(s[keep], 2 ** 32 - 1).astype(np.uint32)
        u.tofile(fk); s.tofile(fc); nkeep += len(u)
    fk.close(); fc.close(); log('併合 %d 項目 -> %d 項目 (範囲 %d, しきい値 %d)' % (total, nkeep, R, minc))
    return nkeep


class Model:
    def __init__(self, prefix_by_order, V=VOC):
        self.orders = sorted(prefix_by_order); self.V = V; self.t = {}
        for n, pf in prefix_by_order.items():
            k = np.fromfile(pf + '_k.bin', dtype=np.uint64); c = np.fromfile(pf + '_c.bin', dtype=np.uint32)
            ctx = k >> np.uint64(16); st = np.r_[0, np.flatnonzero(ctx[1:] != ctx[:-1]) + 1]
            Nc = np.add.reduceat(c.astype(np.uint64), st).astype(np.float64); T = np.diff(np.r_[st, len(k)]).astype(np.float64)
            self.t[n] = dict(k=k, c=c.astype(np.float64), uc=ctx[st], N=Nc, T=T)
        c1 = self.t[1]; self.uni = np.zeros(V); self.uni[(c1['k'] & np.uint64(0xFFFF)).astype(np.int64)] = c1['c']; self.uni_p = (self.uni + 1) / (self.uni.sum() + V)

    def logprob(self, tok_padded, lo, hi):
        """tok_padded: uint16 (先頭に 8 語の余白)。 位置 [lo,hi) の 各語の確率"""
        idx = np.arange(lo, hi); nxt = tok_padded[idx].astype(np.uint64)
        P = self.uni_p[tok_padded[idx].astype(np.int64)]
        for n in self.orders:
            if n == 1:
                continue
            h = ctx_hash(tok_padded, idx, n - 1, n); key = (h << np.uint64(16)) | nxt; t = self.t[n]
            i = np.minimum(np.searchsorted(t['k'], key), len(t['k']) - 1); cnt = np.where(t['k'][i] == key, t['c'][i], 0.0)
            j = np.minimum(np.searchsorted(t['uc'], h), len(t['uc']) - 1); seen = t['uc'][j] == h
            N = np.where(seen, t['N'][j], 0.0); T = np.where(seen, t['T'][j], 0.0)
            P = np.where(seen, (cnt + T * P) / np.maximum(N + T, 1e-9), P)
        return np.log2(P)
