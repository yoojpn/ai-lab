"""100億語の索引の「誤りの地図」と 1 周目 (文書内の一致を 対数で足す)、 小型 LLM との比較。
入力: map.npz (job_map.py が作る; 分野ごとに tok, doc, pos, cand, mlen, base[n,12,3], cq[n,12,3]) , llm.json, bytes.json
分け方: 文書 60 以降で 調整 (lam と 重み)、 文書 0〜59 で 測る (LLM と同じ文書)。
使い方: python3 map_analyze.py <dir>"""
import sys, json, numpy as np
DIR = sys.argv[1]; Z = np.load(DIR + '/map.npz'); V = 32768; ORD = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
DOMS = sorted({k.rsplit('_', 1)[0] for k in Z.files if k.endswith('_tok')})
try: LLM = json.load(open(DIR + '/llm.json')); BY = json.load(open(DIR + '/bytes.json'))
except Exception: LLM, BY = {}, {}


def wb(out, lam):
    p = np.full(out.shape[0], 1.0 / V)
    for oi in range(len(ORD)):
        N = out[:, oi, 0].astype(float); T = out[:, oi, 1].astype(float); c = out[:, oi, 2].astype(float); m = N > 0
        p = np.where(m, (c + lam[oi] * T * p) / np.maximum(N + lam[oi] * T, 1e-9), p)
    return p


def tune(out):
    lam = [4.0] * len(ORD); grid = [0.1, 0.2, 0.3, 0.6, 1, 1.5, 2, 3, 4.5, 6, 9, 14, 22, 40]
    for it in range(2):
        for oi in range(len(ORD)):
            best = None
            for g in grid:
                l = list(lam); l[oi] = g; b = -np.log2(np.maximum(wb(out, l), 1e-12)).mean()
                if best is None or b < best[0]: best = (b, g)
            lam[oi] = best[1]
    return lam


def table(title, keys, names, bits, te):
    print('  ' + title); tot = bits[te].sum()
    for k, nm in enumerate(names):
        m = te & (keys == k)
        if m.sum(): print('    %-22s 位置の %5.1f%%  平均 %6.3f bit  損失全体の %5.1f%%' % (nm, 100 * m.sum() / te.sum(), bits[m].mean(), 100 * bits[m].sum() / tot))


ALL = {}
for dn in DOMS:
    tok, doc, pos, cand, mlen = (Z[dn + '_' + k] for k in ('tok', 'doc', 'pos', 'cand', 'mlen')); base, cq = Z[dn + '_base'], Z[dn + '_cq']
    tu = doc >= 60; te = doc < 60; lam = tune(base[tu]); P = wb(base, lam); Pc = wb(cq, lam); bits = -np.log2(np.maximum(P, 1e-12))
    hi = np.zeros(len(tok), int)                       # 文脈が見つかった 最高の次数 (の番号)
    for oi in range(len(ORD)): hi[base[:, oi, 0] > 0] = oi
    hit_hi = base[np.arange(len(tok)), hi, 2] > 0     # その次数で 正解の語が 続きにあったか
    seen = np.zeros(len(tok), bool); last = {}
    for i in range(len(tok)):
        if pos[i] == 0: last = {}
        seen[i] = tok[i] in last; last[int(tok[i])] = 1
    chit = (cand == tok) & (mlen > 0)
    uni = base[:, 0, 2] / np.maximum(base[:, 0, 0], 1); fb = np.digitize(uni, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    print('\n===== %s: 索引のみ %.4f bit/語 (文書 0〜59, %d 位置; 調整は 文書 60〜) lam=%s' % (dn, bits[te].mean(), te.sum(), lam))
    hb = np.minimum(np.searchsorted([2, 3, 4, 5, 6, 9, 13, 25], np.array(ORD)[hi], side='right'), 7)
    table('文脈が見つかった最高次数', hb, ['1 (文脈なし)', '2', '3', '4', '5', '6〜8', '10〜12', '16〜24'], bits, te)
    table('その次数で 正解が続きにあったか', hit_hi.astype(int), ['無かった', 'あった'], bits, te)
    table('正解の語が 文書の前に出ていたか', seen.astype(int), ['出ていない', '出ていた'], bits, te)
    table('文書内の最長一致の続き', np.where(mlen == 0, 0, np.where(chit, 2, 1)), ['一致なし', '続きが外れ', '続きが正解'], bits, te)
    table('正解の語の頻度', fb, ['<1e-7', '1e-7〜', '1e-6〜', '1e-5〜', '1e-4〜', '>=1e-3'], bits, te)
    table('文書内の位置', np.minimum(np.searchsorted([32, 100, 300], pos, side='right'), 3), ['0〜31', '32〜99', '100〜299', '300〜'], bits, te)
    # ---- 1 周目: 文書内の一致の続き を 足す。 (a) 切り替え (平均)  (b) 対数で足す ----
    mb = np.minimum(mlen, 8); hb2 = np.minimum(hi, 8); Pa = P.copy(); Pb = P.copy(); has = mlen > 0
    for a in range(1, 9):
        for b in range(9):
            g = has & (mb == a) & (hb2 == b); gt = g & tu
            if gt.sum() < 20: gt = has & (mb == a) & tu
            if gt.sum() < 20: continue
            p = (chit[gt].sum() + 0.5) / (gt.sum() + 2.0); Pa[g] = np.where(chit[g], p + (1 - p) * P[g], (1 - p) * P[g])
            best = None
            for l in np.linspace(0, 9, 46):              # 対数の重み: P'(w) = P(w) e^{l[w=続き]} / (1 + (e^l - 1) P(続き))
                e = np.exp(l); ll = (np.where(chit[gt], l, 0) - np.log(1 + (e - 1) * Pc[gt])).sum()
                if best is None or ll > best[0]: best = (ll, l)
            l = best[1]; e = np.exp(l); Pb[g] = P[g] * np.where(chit[g], e, 1.0) / (1 + (e - 1) * Pc[g])
    ba = -np.log2(np.maximum(Pa, 1e-12)); bb = -np.log2(np.maximum(Pb, 1e-12))
    print('  1 周目 (文書内の一致の続き): 索引 %.4f -> 切り替え %.4f (得 %.4f) / 対数で足す %.4f (得 %.4f)' % (bits[te].mean(), ba[te].mean(), bits[te].mean() - ba[te].mean(), bb[te].mean(), bits[te].mean() - bb[te].mean()))
    r = dict(index=float(bits[te].mean()), switch=float(ba[te].mean()), loglin=float(bb[te].mean()))
    if dn in BY:
        nb = sum(BY[dn]); r['bytes_per_tok'] = nb / te.sum()
        line = '  bit/byte (同じ 60 文書): 索引 %.4f, 索引+1周目 %.4f' % (bits[te].sum() / nb, bb[te].sum() / nb); r['index_bpb'] = float(bits[te].sum() / nb); r['round1_bpb'] = float(bb[te].sum() / nb)
        for mn, d in LLM.items():
            if dn in d: line += ' | %s (%.0fM) %.4f' % (mn.split('/')[-1], d['params'] / 1e6, d[dn]['bpb']); r[mn] = d[dn]['bpb']
        print(line)
    ALL[dn] = r
json.dump(ALL, open(DIR + '/map_summary.json', 'w'), ensure_ascii=False, indent=1)
