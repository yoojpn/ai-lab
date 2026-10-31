"""会話の返答検索 (1-of-100): 文脈に対する本当の返答を、 99 個のにせ物 (ほかの会話の最後の発言) の中から当てる。
  採点 = log2 P(返答 | 文脈) - log2 P(返答 | 空の文脈)   (空の文脈 = <bos><turn>。 返答の言い回しの出やすさを引く)
  比べるもの: LM(1..5) だけ / +規則+コピー  x  文脈 = 全部 / 直前の発言だけ
  文脈を全部使って 直前だけより良くなるか = 遠い文脈を使えているか。
使い方: python3 dlg_retr.py '{"dir":"/tmp/tok","n":300,"rules":"D"}'
"""
import sys, json, time, pickle
import numpy as np
import engine as E, core as C
import dlg_eval as D


def build_items(Tte, docs, lang, n, rng, maxlen=300):
    ends = np.r_[docs[1:], len(Tte)]; items = []
    for d in rng.permutation(len(docs)):
        a, b = docs[d], ends[d]
        if b - a > maxlen or b - a < 20: continue
        x = np.asarray(Tte[a:b]); seps = np.flatnonzero(x == 2)
        if len(seps) < 2: continue
        cut = seps[-1]; resp = x[cut + 1:]
        if len(resp) < 3 or resp[-1] != 1: continue
        items.append(dict(ctx=x[:cut + 1], last=np.r_[0, x[seps[-2] + 1:cut + 1]], resp=resp, lang=int(lang[d])))
        if len(items) >= n: break
    return items


def score(parts, resp_lens, model, fm, cls, rcset, SEPS=(1, 2)):
    """parts: [(文脈の語列, 返答の語列)]  -> 返答の各候補の log2 確率の合計 (LM のみ / 規則+コピーつき)"""
    arrs = [np.zeros(8, np.uint16)]; starts = []; pos = []; off = 8; seg = []
    for c, r in parts:
        starts.append(off); a = np.r_[c, r].astype(np.uint16); arrs.append(a)
        pos.append(np.arange(off + len(c), off + len(a))); seg.append(len(r)); off += len(a)
    T = np.concatenate(arrs); P = np.concatenate(pos); seg = np.array(seg); idx = np.r_[0, np.cumsum(seg)[:-1]]
    lp = model.logprob_at(T, P); out = {'lm': np.add.reduceat(lp, idx)}
    if rcset is not None:
        st = C.Struct(T, np.array(starts), list(SEPS)); F, names, cp = C.features(T, P, st, fm, cls); y = T[P].astype(np.int64)
        Pm, _, _ = D.mix(2.0 ** lp, y, rcset['rc'], rcset['prec'], rcset['rel'], F, cp, fm, True, True)
        out['mix'] = np.add.reduceat(np.log2(Pm), idx)
    return out


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    W = cfg.get('dir', '/tmp/tok'); rng = np.random.default_rng(cfg.get('seed', 1)); aux = np.load(W + '/aux.npz'); fm = aux['fm']; cls = aux['cls']
    Tte = np.fromfile(W + '/test.bin', dtype=np.uint16); docs = np.load(W + '/test_docs.npy'); lang = np.load(W + '/test_lang.npy')
    model = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}); rcset = pickle.load(open('%s/rules_%s.pkl' % (W, cfg.get('rules', 'D')), 'rb'))
    items = build_items(Tte, docs, lang, cfg.get('n', 300), rng); N = len(items); log('評価の会話 %d (日本語 %d)' % (N, sum(i['lang'] for i in items)))
    byl = {l: [k for k, i in enumerate(items) if i['lang'] == l] for l in (0, 1)}
    cand = []                                                                                    # 各 item の 候補 (真 + 99)
    for k, it in enumerate(items):
        pool = [j for j in byl[it['lang']] if j != k]; neg = list(rng.choice(pool, size=min(99, len(pool)), replace=False)); cand.append([k] + neg)
    res = {}
    null = score([(np.array([0, 2]), items[j]['resp']) for j in range(N)], None, model, fm, cls, rcset)
    for mode in ('full', 'last'):
        parts = []; keys = []
        for k, it in enumerate(items):
            c = it['ctx'] if mode == 'full' else it['last']
            for j in cand[k]: parts.append((c, items[j]['resp'])); keys.append((k, j))
        sc = score(parts, None, model, fm, cls, rcset)
        for meth in ('lm', 'mix'):
            S = sc[meth].reshape(N, -1) - np.array([[null[meth][j] for j in cand[k]] for k in range(N)])
            rank = (S > S[:, :1]).sum(1) + 1                                                      # 真の返答の順位
            res['%s/%s' % (meth, mode)] = dict(r1=float((rank == 1).mean()), r5=float((rank <= 5).mean()), mrr=float((1.0 / rank).mean()),
                                              r1_en=float((rank[[k for k in byl[0]]] == 1).mean()) if byl[0] else None, r1_ja=float((rank[[k for k in byl[1]]] == 1).mean()) if byl[1] else None)
            log('%s 文脈=%s: R@1 %.3f R@5 %.3f MRR %.3f (en %s, ja %s)  [偶然なら R@1 0.010]' % (meth, mode, res['%s/%s' % (meth, mode)]['r1'], res['%s/%s' % (meth, mode)]['r5'], res['%s/%s' % (meth, mode)]['mrr'], res['%s/%s' % (meth, mode)]['r1_en'], res['%s/%s' % (meth, mode)]['r1_ja']))
    json.dump(res, open(W + '/dlg_retr.json', 'w'), indent=1); log('RESULT_RETR ' + json.dumps(res))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
