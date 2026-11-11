"""統合モデルと GPT-2 の 差を 位置の種類別に。 使い方: python3 gap_analyze.py"""
import numpy as np
from tokenizers import Tokenizer
W = '/tmp/tokhf/tok'; val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); docs = np.load(W + '/val_docs.npy') + 64
idx = np.random.default_rng(777).integers(64, len(val) - 1, 4000) + 64; OT = Tokenizer.from_file(W + '/tok.json')
cb = np.load('/tmp/combo_te_bits.npy'); nb_ = np.load('/tmp/ngram_te_bits.npy'); gb = np.load('/tmp/gpt2_local.npy'); hi = np.load('/tmp/te_hi.npy'); ok = ~np.isnan(gb)
print('平均 (GPT-2 が取れた %d 位置): n-gram %.3f / 統合 %.3f / GPT-2 %.3f bit/語' % (ok.sum(), nb_[ok].mean(), cb[ok].mean(), gb[ok].mean()))
tr = pad[idx]; ds = docs[np.searchsorted(docs, idx, side='right') - 1]; dpos = idx - ds
tr_text = [OT.decode([int(t)]) for t in tr]
kind = np.array([0 if t.strip() == '' else 1 if any(c.isdigit() for c in t) else 2 if t.strip()[:1].isupper() else 3 if t.strip()[:1].isalpha() else 4 for t in tr_text])
uni = np.bincount(np.fromfile('/tmp/bo24b/sub.bin', dtype=np.uint16)[8:], minlength=32768).astype(float); uni /= uni.sum(); fb = np.digitize(uni[tr], [1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
d = cb - gb                                                              # 正 = GPT-2 の方が良い
def tab(title, keys, names):
    print('  ' + title)
    for k, nm in enumerate(names):
        m = ok & (keys == k)
        if m.sum() > 25: print('    %-14s %5.1f%%  n-gram %6.3f  統合 %6.3f  GPT-2 %6.3f  GPT-2 の得(対 統合) %6.3f  (差の合計の %5.1f%%)' % (nm, 100 * m.sum() / ok.sum(), nb_[m].mean(), cb[m].mean(), gb[m].mean(), d[m].mean(), 100 * d[m].sum() / d[ok].sum()))
tab('文書内の位置', np.minimum(np.searchsorted([8, 32, 100, 300], dpos, side='right'), 4), ['1〜7', '8〜31', '32〜99', '100〜299', '300〜'])
tab('n-gram の最高次数', np.minimum(hi, 5) - 1, ['1', '2', '3', '4', '5'])
tab('正解の語の頻度', fb, ['<1e-7', '1e-7〜', '1e-6〜', '1e-5〜', '1e-4〜', '>=1e-3'])
tab('語の種類', kind, ['空白', '数字', '大文字始まり', '小文字の語', '記号'])
tab('GPT-2 の損失', np.digitize(gb, [0.5, 1, 2, 4, 7]), ['<0.5', '0.5〜1', '1〜2', '2〜4', '4〜7', '>=7'])
tab('統合の損失', np.digitize(cb, [0.5, 1, 2, 4, 7]), ['<0.5', '0.5〜1', '1〜2', '2〜4', '4〜7', '>=7'])
big = np.flatnonzero(ok & (d > 5)); print('  GPT-2 が 5 bit 以上 勝つ位置: %d (%.1f%%) / 統合が 5 bit 以上 勝つ位置: %d (%.1f%%)' % (len(big), 100 * len(big) / ok.sum(), (ok & (d < -5)).sum(), 100 * (ok & (d < -5)).sum() / ok.sum()))
rng = np.random.default_rng(1)
for i in rng.choice(big, 16, replace=False):
    s = max(ds[i], idx[i] - 14); ctx = ''.join(OT.decode([int(t)]) for t in pad[s:idx[i]]).replace('\n', ' ')[-80:]; print('    ...%s [[%s]]  統合 %.1f / GPT-2 %.1f (文書内 %d 語目)' % (ctx, tr_text[i].replace('\n', ' '), cb[i], gb[i], dpos[i]))
