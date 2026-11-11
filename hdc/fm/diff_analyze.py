"""LLM と 100億語の索引の 語ごとの差。 LLM が索引に勝つのは どんな語・文脈か。
入力: map.npz (索引の次数ごとの数え), llm_tok.npz (LLM の語ごとの bit; 偶数番の文書のみ)。 使い方: python3 diff_analyze.py <dir> <モデル名の末尾 (gpt2 / SmolLM2-135M / Qwen2.5-0.5B)>"""
import sys, json, numpy as np
DIR, MN = sys.argv[1], sys.argv[2]; Z = np.load(DIR + '/map.npz'); L = np.load(DIR + '/llm_tok.npz'); V = 32768; ORD = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
from tokenizers import Tokenizer
OT = Tokenizer.from_file(DIR + '/tok.json')
def wb(out, lam):
    p = np.full(out.shape[0], 1.0 / V)
    for oi in range(len(ORD)):
        N = out[:, oi, 0].astype(float); T = out[:, oi, 1].astype(float); c = out[:, oi, 2].astype(float); m = N > 0
        p = np.where(m, (c + lam[oi] * T * p) / np.maximum(N + lam[oi] * T, 1e-9), p)
    return p
LAM = [40, 40, 6, 9, 9, 14, 9, 9, 9, 2, 3, 3]
for dn in sorted({k.split('__')[0] for k in L.files if k.endswith('__' + MN)}):
    tok, doc, pos, base = Z[dn + '_tok'], Z[dn + '_doc'], Z[dn + '_pos'], Z[dn + '_base']; lb = L[dn + '__' + MN]; ok = ~np.isnan(lb) & (pos > 0)
    ib = -np.log2(np.maximum(wb(base, LAM), 1e-12)); d = ib - lb                      # 正 = LLM の方が良い
    hi = np.zeros(len(tok), int)
    for oi in range(len(ORD)): hi[base[:, oi, 0] > 0] = oi
    print('\n===== %s (%s): 索引 %.3f bit/語, LLM %.3f bit/語, 差 %.3f (LLM の得)' % (dn, MN, ib[ok].mean(), lb[ok].mean(), d[ok].mean()))
    def tab(title, keys, names):
        print('  ' + title)
        for k, nm in enumerate(names):
            m = ok & (keys == k)
            if m.sum() > 30: print('    %-18s %5.1f%% 位置  索引 %6.3f  LLM %6.3f  LLM の得 %6.3f  (全体の差の %5.1f%%)' % (nm, 100 * m.sum() / ok.sum(), ib[m].mean(), lb[m].mean(), d[m].mean(), 100 * d[m].sum() / d[ok].sum()))
    hb = np.minimum(np.searchsorted([2, 3, 4, 5, 6, 9, 13, 25], np.array(ORD)[hi], side='right'), 7); tab('索引が見つけた最高次数', hb, ['1', '2', '3', '4', '5', '6〜8', '10〜12', '16〜24'])
    uni = base[:, 0, 2] / np.maximum(base[:, 0, 0], 1); fb = np.digitize(uni, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]); tab('正解の語の頻度', fb, ['<1e-7', '1e-7〜', '1e-6〜', '1e-5〜', '1e-4〜', '>=1e-3'])
    txt = [OT.decode([int(t)]) for t in tok]; kind = np.array([0 if t.strip() == '' else 1 if any(c.isdigit() for c in t) else 2 if t.strip()[:1].isupper() else 3 if t.strip()[:1].isalpha() else 4 for t in txt])
    tab('語の種類', kind, ['空白', '数字を含む', '大文字で始まる', '小文字の語', 'それ以外(記号)'])
    tab('文書内の位置', np.minimum(np.searchsorted([8, 32, 100, 300], pos, side='right'), 4), ['1〜7', '8〜31', '32〜99', '100〜299', '300〜'])
    tab('LLM の損失の大きさ', np.digitize(lb, [0.5, 1, 2, 4, 7]), ['<0.5', '0.5〜1', '1〜2', '2〜4', '4〜7', '>=7'])
    big = np.flatnonzero(ok & (d > 6)); rng = np.random.default_rng(0); print('  LLM だけが当てた例 (差 > 6 bit): %d 位置 (%.1f%%)' % (len(big), 100 * len(big) / ok.sum()))
    for i in rng.choice(big, min(12, len(big)), replace=False):
        s = max(0, i - 12); ctx = ''.join(txt[s:i]).replace('\n', ' ')[-70:]; print('    ...%s [[%s]]  索引 %.1f / LLM %.1f' % (ctx, txt[i].replace('\n', ' '), ib[i], lb[i]))
