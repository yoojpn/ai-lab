"""段階 B: 会話での測定。
  1. 特徴を標本位置で作り、 規則被覆で規則を集める (特徴の組ごとに 4 通り: 連続のみ / +遠い・構造 / +クラス / 全部)
  2. 検証の標本で 規則ごとの適合率とコピーの信頼度を数える
  3. テストの標本で bit/token:  LM(1..5) / LM(1..3) / +規則 / +コピー  (言語別も)
  4. 見つかった規則の実物 (復号して表示)
使い方: python3 dlg_eval.py '{"dir":"/tmp/tok","ntrain":2000000,"nval":400000,"ntest":1000000}'
"""
import sys, json, os, time, pickle
import numpy as np
import engine as E, core as C
from rulecover import RuleCover
from tokenizers import Tokenizer


def sample_pos(T, lo, hi, n, rng):
    p = rng.integers(lo, hi, int(n * 1.02)); p = p[np.asarray(T[p]) != 0]
    return np.unique(p)[:n] if len(p) >= n else np.unique(p)


def lang_of(docs, lang, pos):
    d = np.searchsorted(docs, pos, side='right') - 1
    return lang[np.maximum(d, 0)]


def calibrate(rc, F, cp, y, fm):
    """検証の標本で、 規則ごとの適合率 と コピーの信頼度 (種類 x 語の種類) を数える"""
    pred, rid = rc.predict(F) if rc is not None else (np.full(len(y), -1), np.full(len(y), -1))
    nr = len(rc.rules) if rc is not None else 0
    prec = np.zeros(max(nr, 1)); fires = np.zeros(max(nr, 1), np.int64)
    for r in range(nr):
        m = rid == r; fires[r] = m.sum(); prec[r] = ((pred[m] == y[m]).sum() + 1) / (m.sum() + 2)
    fired = pred >= 0
    cpred = np.where(cp[:, 1] >= 0, cp[:, 1], cp[:, 0]); mt = np.where(cp[:, 1] >= 0, 2, np.where(cp[:, 0] >= 0, 1, 0))
    kind = np.where(cpred < 0, 0, np.where(fm[np.clip(cpred, 0, len(fm) - 1)] <= 240, 1, 2))          # 1=頻出語, 2=珍しい語
    rel = np.zeros((3, 3))
    for m_ in (1, 2):
        for k_ in (1, 2):
            s = (~fired) & (mt == m_) & (kind == k_)
            rel[m_, k_] = ((cpred[s] == y[s]).sum() + 1) / (s.sum() + 2) if s.sum() > 50 else 0.0
    return prec, rel


def mix(plm, y, rc, prec, rel, F, cp, fm, use_rules=True, use_copy=True):
    P = plm.copy(); nfire = 0; ncp = 0
    if use_rules and rc is not None and len(rc.rules):
        pred, rid = rc.predict(F); f = pred >= 0; r = np.where(f, prec[np.maximum(rid, 0)], 0.0)
        P = np.where(f, np.where(pred == y, r + (1 - r) * plm, (1 - r) * plm), plm); nfire = int(f.sum())
    else:
        f = np.zeros(len(y), bool)
    if use_copy:
        cpred = np.where(cp[:, 1] >= 0, cp[:, 1], cp[:, 0]); mt = np.where(cp[:, 1] >= 0, 2, np.where(cp[:, 0] >= 0, 1, 0))
        kind = np.where(cpred < 0, 0, np.where(fm[np.clip(cpred, 0, len(fm) - 1)] <= 240, 1, 2)); r = rel[mt, kind]
        u = (~f) & (cpred >= 0) & (r > 0); ncp = int(u.sum())
        P = np.where(u, np.where(cpred == y, r + (1 - r) * plm, (1 - r) * plm), P)
    return P, nfire, ncp


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    W = cfg.get('dir', '/tmp/tok'); rng = np.random.default_rng(cfg.get('seed', 0)); meta = json.load(open(W + '/meta.json'))
    aux = np.load(W + '/aux.npz'); fm = aux['fm']; cls = aux['cls']
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); Tva = np.fromfile(W + '/val.bin', dtype=np.uint16); Tte = np.fromfile(W + '/test.bin', dtype=np.uint16)
    docs = {s: np.load('%s/%s_docs.npy' % (W, s)) for s in ('train', 'val', 'test')}; lang = {s: np.load('%s/%s_lang.npy' % (W, s)) for s in ('train', 'val', 'test')}
    SEPS = [1, 2]
    st = {'train': C.Struct(Ttr, docs['train'], SEPS), 'val': C.Struct(Tva, docs['val'], SEPS), 'test': C.Struct(Tte, docs['test'], SEPS)}; log('構造 (区切り位置) 済み')
    model5 = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3, 4, 5)}); model3 = E.Model({n: '%s/final_o%d' % (W, n) for n in (1, 2, 3)}); log('LM 読み込み済み')
    ptr = sample_pos(Ttr, 8, len(Ttr), cfg.get('ntrain', 2_000_000), rng); pva = sample_pos(Tva, 8, len(Tva), cfg.get('nval', 400_000), rng); pte = sample_pos(Tte, 8, len(Tte), cfg.get('ntest', 1_000_000), rng)
    Ftr, names, cptr = C.features(Ttr, ptr, st['train'], fm, cls); ytr = np.asarray(Ttr[ptr]).astype(np.int64)
    Fva, _, cpva = C.features(Tva, pva, st['val'], fm, cls); yva = Tva[pva].astype(np.int64)
    Fte, _, cpte = C.features(Tte, pte, st['test'], fm, cls); yte = Tte[pte].astype(np.int64); log('特徴 %d 個: %s' % (len(names), names))
    ix = {n: i for i, n in enumerate(names)}
    sets = {
        'A連続のみ': [ix['off%d' % o] for o in range(1, 7)],
        'B+遠い・構造': [i for n, i in ix.items() if not n.startswith('c') or n in ('cur_first',)] ,
        'C+クラス': [i for n, i in ix.items() if n not in ('ind1', 'ind2', 'topic')],
        'D全部': list(range(len(names)))}
    sets['B+遠い・構造'] = [i for n, i in ix.items() if not (n.startswith('c1') or n.startswith('c2') or n.startswith('c3') or n.startswith('c_'))]
    lp5 = model5.logprob_at(Tte, pte); lp3 = model3.logprob_at(Tte, pte); plm5 = 2.0 ** lp5; plm3 = 2.0 ** lp3
    plm5v = 2.0 ** model5.logprob_at(Tva, pva)
    ltt = lang_of(docs['test'], lang['test'], pte)
    def bits(P, m=None): return float(-np.log2(P if m is None else P[m]).mean())
    res = {'n_test': int(len(pte)), 'lm5': bits(plm5), 'lm3': bits(plm3), 'lm5_en': bits(plm5, ltt == 0), 'lm5_ja': bits(plm5, ltt == 1), 'n_ja': int((ltt == 1).sum())}
    log('LM のみ: 5-gram %.4f / 3-gram %.4f bit/token (en %.4f, ja %.4f; ja 件数 %d)' % (res['lm5'], res['lm3'], res['lm5_en'], res['lm5_ja'], res['n_ja']))
    Pc, _, ncp = mix(plm5, yte, None, None, calibrate(None, Fva, cpva, yva, fm)[1], Fte, cpte, fm, use_rules=False, use_copy=True)
    res['lm5+copy'] = bits(Pc); log('LM5 + コピー: %.4f (コピー使用 %d)' % (res['lm5+copy'], ncp))
    tk = Tokenizer.from_file(W + '/tok.json'); rules_out = {}
    for nm, allowed in sets.items():
        rc = RuleCover(maxk=cfg.get('maxk', 4), beam=cfg.get('beam', 6), purity=cfg.get('purity', 0.98), minn=5, sample=cfg.get('sample', 200_000), rounds=cfg.get('rounds', 20), ny=65536)
        rc.fit(Ftr, ytr, names, log=lambda s: None, allowed=allowed)
        prec, rel = calibrate(rc, Fva, cpva, yva, fm)
        P1, nf, _ = mix(plm5, yte, rc, prec, rel, Fte, cpte, fm, use_rules=True, use_copy=False)
        P2, nf2, ncp2 = mix(plm5, yte, rc, prec, rel, Fte, cpte, fm, use_rules=True, use_copy=True)
        res[nm] = dict(rules=len(rc.rules), fire=nf / len(yte), lm5_rules=bits(P1), lm5_rules_copy=bits(P2), en=bits(P2, ltt == 0), ja=bits(P2, ltt == 1))
        log('%s: 規則 %d 本 発火 %.3f | LM5+規則 %.4f | +コピー %.4f (en %.4f, ja %.4f)' % (nm, len(rc.rules), nf / len(yte), res[nm]['lm5_rules'], res[nm]['lm5_rules_copy'], res[nm]['en'], res[nm]['ja']))
        ex = []
        pred, rid = rc.predict(Fva)
        for r, (sub, _) in enumerate(rc.rules):
            m = np.flatnonzero((rid == r) & (pred == yva))[:2]
            for i in m:
                p = pva[i]; ctx = Tva[max(0, p - 14):p].tolist(); ex.append(dict(rule=r, feats=[names[f] for f in sub], prec=float(prec[r]), ctx=tk.decode(ctx, skip_special_tokens=False), pred=tk.decode([int(pred[i])], skip_special_tokens=False)))
        rules_out[nm] = ex[:40]
        pickle.dump(dict(rc=rc, prec=prec, rel=rel), open('%s/rules_%s.pkl' % (W, nm[0]), 'wb'))
    json.dump(dict(res=res, rules=rules_out), open(W + '/dlg_eval.json', 'w'), ensure_ascii=False, indent=1)
    log('RESULT_DLG ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
