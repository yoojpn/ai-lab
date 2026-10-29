"""
SIZE: 表(CB2)の「保存形式」と「枝刈り」。 実際に符号化したバイト数と、枝刈りしたときの bit/語 を出す。
保存形式 (列ごとの可変長整数 -> zlib):  疎な表1個を、文脈の順に
   s1 文脈番号の差分(可変長) / s2 文脈の合計(可変長) / s3 文脈ごとの項目数(可変長) / s4 語番号の差分(文脈内, 可変長) / s5 回数(可変長)
   可変長整数: 7ビットずつ(上位ビット=続き)。 sizeは zlib(6) で圧縮した各列の合計 + ブロック目次の見積もり(64KBごとに4バイト)。
枝刈り m: 回数が m 未満の項目を捨てる。 文脈の合計は元の値を残す(捨てた分は、見たことのない語に回る)。 g5は元から最小回数2。
その他: tri 表も同じ形式で符号化。 PMI(8 x V x V) は 1バイトに量子化した場合の大きさ。
使い方: LADDER_CORPUS=... python3 exp_size.py '{"N":103000000,"ms":[1,2,3,5]}'
"""
import os, sys, json, math, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
from exp_cb import V, SEQ, LN2, BUCKETS, log


def varint_bytes(a):
    a = a.astype(np.uint64)
    lens = 1 + (a >= 128).astype(np.int64) + (a >= 16384) + (a >= 2097152) + (a >= 268435456)
    buf = np.empty(int(lens.sum()), np.uint8); starts = np.cumsum(lens) - lens
    for j in range(5):
        m = lens > j
        if not m.any():
            break
        b = ((a[m] >> np.uint64(7 * j)) & np.uint64(127)).astype(np.uint8)
        buf[starts[m] + j] = b | ((lens[m] > j + 1).astype(np.uint8) << 7)
    return buf


def encode_table(uk, cn, tot):
    """uk: 昇順のキー(文脈*V+語), cn: 回数, tot: 項目ごとに繰り返した文脈の合計。 戻り値: (生バイト, zlibバイト, 文脈数)"""
    ctx = uk // V; w = (uk % V).astype(np.int64)
    first = np.r_[True, ctx[1:] != ctx[:-1]]; ci = np.flatnonzero(first)
    cid = ctx[ci]; n_e = np.diff(np.r_[ci, len(ctx)]); ctot = tot[ci]
    dw = np.where(first, w, w - np.r_[0, w[:-1]])
    streams = [np.diff(np.r_[0, cid]), np.round(ctot).astype(np.int64), n_e, dw, np.round(cn).astype(np.int64)]
    raw = comp = 0
    for s in streams:
        bb = varint_bytes(np.maximum(s, 0)); raw += len(bb); comp += len(zlib.compress(bb.tobytes(), 6))
    return raw, comp + 4 * (comp // 65536 + 1), len(cid)


def prune(T, m):
    uk, cn, tot = T
    keep = cn >= m
    return uk[keep], cn[keep], tot[keep]


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000); ms = cfg.get('ms', [1, 2, 3, 5])
    x, Xva, yva, Xte, yte = B.load(N, ntune, ntest)
    log('学習 %d 語, 調整窓 %d, テスト窓 %d' % (len(x), len(yva), len(yte)))
    G = B.Model(); cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = B.build_counts(x); G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float(); G.tri = B.build_tri(x)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, cfg.get('beta', 1280.0)) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, Sv, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :64] * Sv[:64].sqrt()[None, :]; E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    G.cls = B.kmeans(E, C2.K); del C, ppmi
    cls_np = G.cls.numpy(); base = {}
    for nm, spec in C2.SPECS.items():
        base[nm] = C2.build_sparse(x, cls_np, spec, C2.MINCNT.get(nm, 1))
    del x
    log('表の構築 完了。 項目数 %s' % {k: len(v[0]) for k, v in base.items()})
    # tri 表の大きさ (キー = a*V*V + b*V + c -> 文脈 (a,b))
    tk, tc = G.tri; ttot = np.repeat(np.add.reduceat(tc, np.r_[0, np.flatnonzero((tk // V)[1:] != (tk // V)[:-1]) + 1]),
                                      np.diff(np.r_[np.r_[0, np.flatnonzero((tk // V)[1:] != (tk // V)[:-1]) + 1], len(tk)])).astype(np.float32)
    r_tri = encode_table(tk, tc, ttot); log('tri 表: 項目 %d, 生 %.1f MB, 圧縮 %.1f MB (%.2f バイト/項目)' % (len(tk), r_tri[0] / 1e6, r_tri[1] / 1e6, r_tri[1] / len(tk)))
    pmi_mb = 8 * V * V / 1e6; log('PMI 8個 (1バイト量子化): %.0f MB' % pmi_mb)
    rows = []
    for m in ms:
        t = time.time(); sp = {}; tab = {}; ent = 0; comp_t = raw_t = 0
        for nm, T in base.items():
            Tm = prune(T, m) if m > 1 else T
            sp[nm] = Tm; r = encode_table(*Tm) if len(Tm[0]) else (0, 0, 0)
            tab[nm] = dict(entries=int(len(Tm[0])), raw=int(r[0]), comp=int(r[1]), ctx=int(r[2])); ent += len(Tm[0]); comp_t += r[1]; raw_t += r[0]
        log('m=%d: 疎な表の項目 %d, 生 %.1f MB, 圧縮 %.1f MB (%.2f バイト/項目), 符号化 %.0f 秒' % (m, ent, raw_t / 1e6, comp_t / 1e6, comp_t / max(ent, 1), time.time() - t))
        G.sp = sp; t = time.time(); Fv, gv = C2.collect(Xva, G)
        bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
        allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
        w, D, b = C2.fit(Fv, allx, yva, gate=gv, gated=True); del Fv
        tot = 0.0; n_te = len(yte)
        for a in range(0, n_te, 250):
            Fc, gc = C2.all_feats(Xte[a:a + 250], G); yc = yte[a:a + 250]
            with torch.no_grad():
                tot += Fn.cross_entropy(C2.logits(Fc, allx, w, gc, D), yc, reduction='sum').item()
        bits = tot / n_te / LN2
        total_mb = (comp_t + r_tri[1]) / 1e6 + pmi_mb
        rows.append(dict(m=m, entries=int(ent), comp_mb=comp_t / 1e6, raw_mb=raw_t / 1e6, total_mb=total_mb, bits=bits, tab=tab))
        log('m=%d: テスト %.4f bit/語, 合計 %.0f MB (疎な表 %.0f + tri %.0f + PMI %.0f), %.0f 秒' % (m, bits, total_mb, comp_t / 1e6, r_tri[1] / 1e6, pmi_mb, time.time() - t))
    log('--- 結果 ---')
    for r in rows:
        log('m=%d 項目 %.3e 合計 %.0f MB テスト %.4f bit/語' % (r['m'], r['entries'], r['total_mb'], r['bits']))
    log('RESULT_SIZE ' + json.dumps(dict(cfg=cfg, rows=rows, tri_mb=r_tri[1] / 1e6, pmi_mb=pmi_mb)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
