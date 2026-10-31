"""
MIX: 文章(Web)を足しても会話を悪化させない組み込み方。 新エンジンで測る。 英語のみ、 きれいに分けた DailyDialog (重複を除く) を会話の評価に使う。
  D = 会話だけの表 / W = Web 文章だけの表 / P = 会話+Web を単純に合算した表 / P5 = 会話を5回数えて合算
  静的補間 = λ D + (1-λ) W (λ は会話の検証窓で決める)
  ベイズ混合 = 専門家 D, W の確率を 語ごとに  w_i ∝ w_i * p_i(x)  で更新し、 毎語 α だけ一様へ戻す (fixed-share)。 調整するのは α のみ。 文書(会話)の先頭で w を戻す。
      保証: 専門家 K 個の最良 1 つに対し、 合計の損失は  最良の損失 + log2 K ビット (+ 切替ごとに log2(1/α) 程度)。
  評価: 会話のテスト(きれいな分割)の bit/語、 Web のテストの bit/語 (Web 側が悪化しないか)
使い方: python3 exp_mix.py '{"web_mb":300}'
"""
import sys, os, re, io, json, time, lzma, shutil, hashlib
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import engine as E
import exp_dlg as D

TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")


def norm(s):
    return ' '.join(TOK.findall(s.lower()))


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    W = '/tmp/mix'; shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    dl = D.load_dialogs()
    def turns(d):
        out = []; cur = []
        for t in d:
            if t == D.SEP:
                out.append(tuple(cur)); cur = []
            else:
                cur.append(t)
        if cur: out.append(tuple(cur))
        return out
    tr = {}; seen_long = set(); key3 = set()
    for d in dl['train']:
        T = turns(d); k = tuple(T[:3])
        if k in key3: continue
        key3.add(k); tr[len(tr)] = T; seen_long.update(t for t in T if len(t) >= 5)
    def clean(ds):
        out = []; ks = set()
        for d in ds:
            T = turns(d)
            if any(t in seen_long for t in T if len(t) >= 5) or tuple(T[:3]) in key3: continue
            k = tuple(T[:3])
            if k in ks: continue
            ks.add(k); out.append(T)
        return out
    val = clean(dl['validation']); test = clean(dl['test']); train = list(tr.values())
    log('会話: 学習 %d (重複除去後), 検証 %d, テスト %d (きれい)' % (len(train), len(val), len(test)))
    dtxt = lambda ds: [' \n '.join(' '.join(t) for t in T) for T in ds]
    # Web: CC-100 英語の先頭 (xz の途中までを読む)
    web = []; nb = 0
    dec = lzma.LZMADecompressor(); buf = b''
    with open('/tmp/data/cc100_en.xz', 'rb') as f:
        while nb < cfg['web_mb'] * 1e6:
            c = f.read(1 << 22)
            if not c: break
            try:
                out = dec.decompress(c)
            except Exception:
                break
            buf += out; nb += len(out)
            *lines, buf = buf.split(b'\n')
            web += [l.decode('utf8', 'ignore') for l in lines if l.strip()]
    web = [norm(l) for l in web]; web = [l for l in web if len(l) > 20]
    nte = len(web) // 100; web_test = web[:nte]; web_train = web[nte:]
    log('Web: 学習 %d 行 (%.0f MB), テスト %d 行' % (len(web_train), sum(map(len, web_train)) / 1e6, len(web_test)))
    tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tr_ = trainers.BpeTrainer(vocab_size=E.VOC, special_tokens=['<s>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train_from_iterator(dtxt(train)[:20000] + web_train[:200000], tr_)
    def enc(texts, per_doc=True):
        out = [np.zeros(8, np.uint16)]
        for a in range(0, len(texts), 500):
            for e in tk.encode_batch(texts[a:a + 500]):
                out.append(np.array(e.ids, np.uint16)); out.append(np.zeros(1, np.uint16))
        return np.concatenate(out)
    Dtr, Dva, Dte = enc(dtxt(train)), enc(dtxt(val)), enc(dtxt(test)); Wtr = enc(web_train); Wte = enc(web_test[:20000])
    log('語数: 会話学習 %d, 会話検証 %d, 会話テスト %d, Web学習 %d, Webテスト %d' % (len(Dtr), len(Dva), len(Dte), len(Wtr), len(Wte)))
    ORD = [1, 2, 3, 4]
    def build(tokens, tag, minc4=1):
        tokens.tofile(W + '/%s.tok' % tag); runs = E.spill(W + '/%s.tok' % tag, len(tokens) - 8, W + '/r_' + tag, ORD, 8_000_000, lambda s: None); pf = {}
        for n in ORD:
            pf[n] = W + '/f_%s_o%d' % (tag, n); E.merge(runs[n], pf[n], minc4 if n == 4 else 1, 20_000_000, lambda s: None)
        shutil.rmtree(W + '/r_' + tag); return E.Model(pf)
    mD = build(Dtr, 'D'); mW = build(Wtr, 'W', 2)
    mP = build(np.concatenate([Dtr, Wtr[8:]]), 'P', 2); mP5 = build(np.concatenate([Dtr] + [Dtr[8:]] * 4 + [Wtr[8:]]), 'P5', 2)
    log('表を作成')
    def lp(m, t): return m.logprob(t, 8, len(t))
    res = {}
    sets = {'会話(検証)': Dva, '会話(テスト)': Dte, 'Web(テスト)': Wte}
    L = {k: {n: lp(m, t) for n, m in (('D', mD), ('W', mW), ('P', mP), ('P5', mP5))} for k, t in sets.items()}
    # 静的補間 (λ を会話の検証で選ぶ)
    best = None
    for lam in np.linspace(0.05, 0.95, 19):
        v = L['会話(検証)']; b = -np.log2(lam * 2 ** v['D'] + (1 - lam) * 2 ** v['W']).mean()
        if best is None or b < best[0]: best = (b, lam)
    lam = best[1]; log('静的補間 λ=%.2f (検証 %.4f)' % (lam, best[0]))
    def bayes(lD, lW, t, alpha):
        isb = (t[8:] == 0)                                     # 文書の区切り
        w = np.array([0.5, 0.5]); tot = 0.0
        pD = 2 ** lD; pW = 2 ** lW
        for i in range(len(pD)):
            p = w[0] * pD[i] + w[1] * pW[i]; tot -= np.log2(p)
            w = w * np.array([pD[i], pW[i]]) / p
            w = (1 - alpha) * w + alpha / 2
            if isb[i]: w = np.array([0.5, 0.5])
        return tot / len(pD)
    # α を会話の検証で選ぶ
    ba = min(((bayes(L['会話(検証)']['D'], L['会話(検証)']['W'], Dva, a), a) for a in (0.001, 0.003, 0.01, 0.03, 0.1)))
    alpha = ba[1]; log('ベイズ混合 α=%.3f (検証 %.4f)' % (alpha, ba[0]))
    for k in ('会話(テスト)', 'Web(テスト)'):
        v = L[k]; t = sets[k]; r = {n: float(-v[n].mean()) for n in v}
        r['静的補間'] = float(-np.log2(lam * 2 ** v['D'] + (1 - lam) * 2 ** v['W']).mean()); r['ベイズ混合'] = float(bayes(v['D'], v['W'], t, alpha))
        res[k] = r; log('%s: %s' % (k, {a: round(b, 4) for a, b in r.items()}))
    log('RESULT_MIX ' + json.dumps(dict(cfg=cfg, res=res, lam=float(lam), alpha=alpha, sizes=dict(Dtr=len(Dtr), Wtr=len(Wtr)))))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
