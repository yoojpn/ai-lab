"""
G6: 実際の Python コード (標準ライブラリ 約 590 ファイル) で、 数え上げの言語モデルが どこまで予測できるか。
  語: tokenize で分けた語。 NAME/OP はそのまま (頻出 8000 まで, 残りは UNK_NAME), NUMBER -> <num>, STRING -> <str>, COMMENT -> <cmt>,
      NEWLINE/NL/INDENT/DEDENT も語。 (文字列・数値の中身は 予測しない = 構造と識別子の予測)
  部品 (全部 数え上げ):
    LM    : 語単位の 3-gram (Witten-Bell 補間)
    規則  : 規則被覆。 特徴 = off1..8, 括弧スタック (stk1..3, depth0), インデント段数, 行内の位置, 直前語の前回出現の次語 (lo)。
    コピー: 直前2語 / 直前1語と同じ並びの前回出現(同じファイル内)の次の語を予測。 信頼度 r は 検証ファイルでの当たり率 (一致の長さ x 語の種類) の表。
  合流: 規則が発火 -> r_rule, なければコピー -> r_copy, なければ LM。  P = r*[x==w] + (1-r) P_lm
  評価: 保持したファイル (学習に出ていない) での bit/語。 gzip などとは 対象が違う (中身を除く) ので比べない。
使い方: python3 exp_g6.py '{"maxfiles":600}'
"""
import sys, json, io, glob, os, time, sysconfig, tokenize, hashlib
import numpy as np
from collections import Counter
from exp_g5 import NGram
from rulecover import RuleCover

KW_SPECIAL = {tokenize.NUMBER: '<num>', tokenize.STRING: '<str>', tokenize.COMMENT: '<cmt>', tokenize.NEWLINE: 'NEWLINE', tokenize.NL: 'NL', tokenize.INDENT: 'INDENT', tokenize.DEDENT: 'DEDENT'}


def toks_of(path):
    try:
        src = open(path, encoding='utf-8').read()
        out = []
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.ENDMARKER:
                continue
            if t.type in KW_SPECIAL:
                out.append(KW_SPECIAL[t.type])
            else:
                out.append(t.string)
        return out
    except Exception:
        return None


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    p = sysconfig.get_paths()['stdlib']; files = sorted(f for f in glob.glob(p + '/**/*.py', recursive=True) if 'site-packages' not in f and 'lib2to3/tests/data' not in f)[:cfg.get('maxfiles', 600)]
    split = {'train': [], 'val': [], 'test': []}; docs = {}
    for f in files:
        tk = toks_of(f)
        if not tk or len(tk) < 50:
            continue
        h = int(hashlib.md5(f.encode()).hexdigest(), 16) % 10
        split['test' if h == 0 else 'val' if h == 1 else 'train'].append(tk)
    for k, v in split.items():
        log('%s: %d ファイル, %d 語' % (k, len(v), sum(map(len, v))))
    cnt = Counter(t for d in split['train'] for t in d)
    NVOC = 8000; vocab = {t: k + 1 for k, (t, _) in enumerate([x for x in cnt.most_common(NVOC) if x[1] >= 2])}; UNK = 0              # 0 = UNK_NAME
    inv = {k: t for t, k in vocab.items()}; V = NVOC + 2
    def enc(ds):
        T = []; S = []; pos = 0
        for d in ds:
            T += [vocab.get(t, UNK) for t in d]; S += [pos] * len(d); pos += len(d)
        return np.array(T, np.int64), np.array(S, np.int64)
    D = {k: enc(v) for k, v in split.items()}
    OPEN = {'(': 1, '[': 2, '{': 3}; CLOSE = {')', ']', '}'}
    def fid(i):                                                       # 特徴用の id (255 未満): 頻出 248 は そのまま, 他は 250 (他の語)
        return i if 0 < i < 249 else (0 if i == 0 else 250)
    def build_feats(T, S, ds_lens):
        N = len(T); F = np.zeros((N, 8 + 3 + 3 + 3), np.int16)
        ft = np.array([fid(x) for x in T.tolist()], np.int16)
        ar = np.arange(N)
        for o in range(1, 9):
            j = ar - o; F[:, o - 1] = np.where(j >= S, ft[np.clip(j, 0, N - 1)], 251)
        # 括弧スタック / インデント / 行内の位置
        pos = 0
        for dl in ds_lens:
            st = []; ind = 0; since = 0
            for k in range(pos, pos + dl):
                s3 = [x for x in st[::-1][:3]] + [0] * (3 - min(3, len(st)))
                F[k, 8:11] = s3; F[k, 11] = int(len(st) == 0); F[k, 12] = min(ind, 8); F[k, 13] = min(since, 20)
                w = inv.get(int(T[k]), 'UNK')
                if w in OPEN:
                    st.append(OPEN[w])
                elif w in CLOSE and st:
                    st.pop()
                if w == 'INDENT':
                    ind += 1
                elif w == 'DEDENT':
                    ind = max(0, ind - 1)
                since = 0 if w in ('NEWLINE', 'NL', 'INDENT', 'DEDENT') else since + 1
            pos += dl
        # 直前語の前回出現の次の語 (lo1_1, lo1_2) と 直前2語の前回出現の次の語 (lo2_1)
        tt = T.astype(np.int64)
        def prevkey(key):
            idx = np.lexsort((ar, key)); ks = key[idx]; same = np.r_[False, ks[1:] == ks[:-1]]
            prv = np.full(N, -1, np.int64); prv[idx[same]] = idx[np.flatnonzero(same) - 1]; ok = prv >= 0; ok[ok] = S[prv[ok]] == S[ok]; prv[~ok] = -1
            return prv
        key1 = S * 20011 + tt; pv1 = prevkey(key1)                       # j の語と同じ語の前回の位置
        for c, v in ((14, 1), (15, 2)):
            j = ar - 1; p_ = np.where(j >= 0, pv1[np.clip(j, 0, N - 1)], -1); q = p_ + v
            F[:, c] = np.where((p_ >= 0) & (q <= ar - 1), ft[np.clip(q, 0, N - 1)], 251)
        names = ['off%d' % o for o in range(1, 9)] + ['stk1', 'stk2', 'stk3', 'depth0', 'indent', 'since_line', 'lo1_1', 'lo1_2']
        F = F[:, :16]
        # コピー予測 (全語): 直前2語 / 1語の前回出現の「次」の語
        prev_t = np.where(ar - 1 >= S, np.roll(tt, 1), -1); prev2 = np.where(ar - 2 >= S, np.roll(tt, 2), -1)
        k2 = prev2 * 20011 + prev_t; pk2 = prevkey(np.where(prev2 >= 0, k2, -ar - 10))
        k1 = prev_t; pk1 = prevkey(np.where(prev_t >= 0, k1 + 10 ** 9 * 0 + 0, -ar - 10))
        # pk の意味: 位置 i の「同じ並びの前回の位置」 p -> 予測 = tt[p]
        cp2 = np.where(pk2 >= 0, tt[np.clip(pk2, 0, N - 1)], -1); cp1 = np.where(pk1 >= 0, tt[np.clip(pk1, 0, N - 1)], -1)
        cp = np.where(cp2 >= 0, cp2, cp1); m = np.where(cp2 >= 0, 2, np.where(cp1 >= 0, 1, 0))
        return F, names, cp, m
    G = {}
    for k in D:
        G[k] = build_feats(D[k][0], D[k][1], [len(d) for d in split[k]])
    names = G['train'][1]; log('特徴 %d 個' % len(names))
    lm = NGram(V + 1).fit(D['train'][0], D['train'][1])
    rc = RuleCover(maxk=cfg.get('maxk', 4), beam=cfg.get('beam', 8), purity=cfg.get('purity', 0.99), minn=cfg.get('minn', 5), sample=cfg.get('sample', 100000), rounds=cfg.get('rounds', 25))
    rc.fit(G['train'][0], D['train'][0], names, log=log)
    def apply_rules(k):
        pred, rid = rc.predict(G[k][0]); return pred
    # 信頼度 (検証ファイルで数える)
    Tv = D['val'][0]; pv = apply_rules('val'); fv = pv >= 0
    r_rule = float((pv[fv] == Tv[fv]).mean()) if fv.any() else 0.0
    cpv, mv = G['val'][2], G['val'][3]
    rel = {}
    for m in (1, 2):
        for cls in range(0, 3):
            sel = (~fv) & (mv == m) & (cpv >= 0) & ((np.array([0 if x == 0 else (1 if x <= 248 else 2) for x in cpv.tolist()]) == cls))
            rel[(m, cls)] = float((cpv[sel] == Tv[sel]).mean()) if sel.sum() > 30 else 0.0
    log('規則 r=%.4f (検証で発火 %.3f) / コピーの信頼度 %s' % (r_rule, fv.mean(), {k: round(v, 3) for k, v in rel.items()}))
    # テスト
    Tt, St = D['test']; Plm = lm.prob(Tt, St)[0]; pt = apply_rules('test'); ft = pt >= 0; cpt, mt = G['test'][2], G['test'][3]
    cls_t = np.array([0 if x == 0 else (1 if 0 < x <= 248 else 2) for x in np.maximum(cpt, 0).tolist()])
    rc_arr = np.array([[rel.get((m, c), 0.0) for c in range(3)] for m in (0, 1, 2)]) if False else None
    r_copy = np.array([rel.get((int(m), int(c)), 0.0) if m > 0 else 0.0 for m, c in zip(mt.tolist(), cls_t.tolist())])
    def bits(P):
        return float(-np.log2(P).mean())
    P_rule = np.where(ft, np.where(pt == Tt, r_rule + (1 - r_rule) * Plm, (1 - r_rule) * Plm), Plm)
    use_c = (~ft) & (cpt >= 0)
    P_copy = np.where(use_c, np.where(cpt == Tt, r_copy + (1 - r_copy) * Plm, (1 - r_copy) * Plm), Plm)
    P_all = np.where(ft, P_rule, P_copy)
    res = dict(n_test=int(len(Tt)), lm=bits(Plm), lm_rules=bits(P_rule), lm_copy=bits(P_copy), lm_rules_copy=bits(P_all),
               rule_fire=float(ft.mean()), rule_acc=float((pt[ft] == Tt[ft]).mean()), copy_use=float(use_c.mean()), copy_acc=float((cpt[use_c] == Tt[use_c]).mean()),
               unk_rate=float((Tt == 0).mean()))
    log('テスト %d 語: LM %.4f / +規則 %.4f / +コピー %.4f / +規則+コピー %.4f bit/語' % (res['n_test'], res['lm'], res['lm_rules'], res['lm_copy'], res['lm_rules_copy']))
    log('規則: 発火 %.3f 正解 %.4f | コピー: 使用 %.3f 正解 %.4f | UNK 率 %.3f' % (res['rule_fire'], res['rule_acc'], res['copy_use'], res['copy_acc'], res['unk_rate']))
    # 語の種類別 (合流後)
    cat = {}
    for nm, sel in (('NEWLINE/NL/INDENT/DEDENT', np.isin(Tt, [vocab.get(x, -9) for x in ('NEWLINE', 'NL', 'INDENT', 'DEDENT')])), ('OP', np.array([inv.get(int(x), '') in set('()[]{}:,.=+-*/<>%') or inv.get(int(x), '') in ('==', '!=', '<=', '>=', '->', '+=') for x in Tt.tolist()])), ('UNK', Tt == 0)):
        if sel.any():
            cat[nm] = dict(n=int(sel.sum()), lm=float(-np.log2(Plm[sel]).mean()), mix=float(-np.log2(P_all[sel]).mean())); log('  %s: n=%d  LM %.3f -> 合流 %.3f' % (nm, sel.sum(), cat[nm]['lm'], cat[nm]['mix']))
    log('RESULT_G6 ' + json.dumps(dict(cfg=cfg, res=res, cat=cat, rules=[[names[f] for f in s] for s, _ in rc.rules])))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
