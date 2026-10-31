"""
G5: 確率の表 (n-gram) と 決定の規則 (規則被覆) の合流。 文章 (pydoc の英文) と Python 実行記録 (G3) を混ぜた列で測る。
  LM   : 語単位の 3-gram (Witten-Bell 補間)。 全部の語を対象にする。
  規則 : 規則被覆。 文章の語は全部 OTHER に潰して特徴を作る。 目的も潰す。 規則が OTHER を予測したら「意見なし」。
         役割・領域の指定はしない (どこで規則が効くかは、 文脈が合うかどうかだけで決まる)。
  合流 : 規則が語 w を予測するとき  P(x) = r*[x==w] + (1-r) P_lm(x)。  r は 学習用の別集合での規則の適合率。
  評価 : 保持した文書での bit/語。  記録部分 / 文章部分 / ソース部分 (ランダムなプログラム文) に分けて。  文章を悪化させないか。
使い方: python3 exp_g5.py '{"ntrain":8000,"ntest":800}'
"""
import sys, json, re, time
import numpy as np
import pydoc_data.topics as PT
import exp_g3 as G
from rulecover import RuleCover

NV = G.NV; OTHER = NV; NW = 4000


class NGram:
    def __init__(self, V):
        self.V = V

    def fit(self, toks, starts):
        V = self.V; t = toks.astype(np.int64); N = len(t); ar = np.arange(N)
        p1 = np.where(ar - 1 >= starts, np.roll(t, 1), V); p2 = np.where(ar - 2 >= starts, np.roll(t, 2), V)       # V = 文頭
        self.c1 = np.bincount(t, minlength=V + 1).astype(np.float64); self.N = N
        V1 = V + 1
        def tab(ctx, nxt):
            kl = ctx * V1 + nxt; u, c = np.unique(kl, return_counts=True); cx = u // V1
            ucx, inv = np.unique(cx, return_inverse=True); n = np.bincount(inv, weights=c); T = np.bincount(inv)
            return u, c.astype(np.float64), ucx, n, T.astype(np.float64)
        self.t2 = tab(p1, t); self.t3 = tab(p2 * V1 + p1, t)
        return self

    def prob(self, toks, starts, which=None):
        """全語彙の確率ではなく、 実際の語の確率を返す (評価用) と、 必要なら全語彙"""
        V = self.V; V1 = V + 1; t = toks.astype(np.int64); N = len(t); ar = np.arange(N)
        p1 = np.where(ar - 1 >= starts, np.roll(t, 1), V); p2 = np.where(ar - 2 >= starts, np.roll(t, 2), V)
        P1 = (self.c1[t] + 1) / (self.N + V)
        def wb(tb, ctx, nxt, lower):
            u, c, ucx, n, T = tb; kl = ctx * V1 + nxt
            i = np.minimum(np.searchsorted(u, kl), len(u) - 1); cnt = np.where(u[i] == kl, c[i], 0.0)
            j = np.minimum(np.searchsorted(ucx, ctx), len(ucx) - 1); seen = ucx[j] == ctx
            nn = np.where(seen, n[j], 0.0); TT = np.where(seen, T[j], 0.0)
            return np.where(seen, (cnt + TT * lower) / (nn + TT), lower)
        P2 = wb(self.t2, p1, t, P1); P3 = wb(self.t3, p2 * V1 + p1, t, P2)
        return P3, (p1, p2)


def lm_prob_of(lm, toks, starts, x_override=None):
    return lm.prob(toks if x_override is None else x_override, starts)[0]


def text_docs(rng):
    words = re.findall(r"[a-z]+|\d+|[^\sa-z\d]", ' '.join(PT.topics.values()).lower())
    from collections import Counter
    cnt = Counter(words); vocab = {w: k for k, (w, _) in enumerate(cnt.most_common(NW - 1), 1)}      # 0 = unk
    ids = np.array([vocab.get(w, 0) for w in words], np.int64) + NV + 1
    docs = [ids[i:i + 150] for i in range(0, len(ids) - 150, 150)]
    rng.shuffle(docs); return docs


def assemble(trace_set, tdocs):
    """トレース集合 (G3 の build_set) と文章の文書を並べる。 戻り値: 語列, 文書の開始, 記録の開始 (文章は無限大), 種類 (0=ソース,1=記録,2=文章)"""
    T, S, TS, _ = trace_set; toks = [T.astype(np.int64)]; st = [S]; ts = [TS]; kind = []
    ar = np.arange(len(T)); kind.append(np.where(ar > TS, 1, 0))
    off = len(T)
    for d in tdocs:
        toks.append(d); st.append(np.full(len(d), off)); ts.append(np.full(len(d), 10 ** 12)); kind.append(np.full(len(d), 2)); off += len(d)
    return np.concatenate(toks), np.concatenate(st), np.concatenate(ts), np.concatenate(kind)


def main(cfg):
    rng = np.random.default_rng(cfg.get('seed', 0)); t0 = time.time()
    log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    docs = text_docs(rng); nte = max(20, len(docs) // 10); dte, dtr = docs[:nte], docs[nte:]
    ntr_t = len(dtr)
    tr = G.build_set(rng, cfg['ntrain'], 4, 10); te = G.build_set(rng, cfg['ntest'], 4, 10)
    # 規則の適合率を測る別集合
    rcal = G.build_set(rng, cfg['ntest'], 4, 10)
    Ttr, Str, TStr, Ktr = assemble(tr, dtr); Tte, Ste, TSte, Kte = assemble(te, dte); Tca, Sca, TSca, Kca = assemble(rcal, dte)
    log('学習 %d 語 (文章 %d, 記録 %d), テスト %d 語 (文章 %d, 記録 %d)' % (len(Ttr), (Ktr == 2).sum(), (Ktr == 1).sum(), len(Tte), (Kte == 2).sum(), (Kte == 1).sum()))
    V = NV + 1 + NW
    lm = NGram(V).fit(Ttr, Str)
    # 規則 (文章を OTHER に潰す)
    col = lambda T_: np.where(T_ >= NV, OTHER, T_).astype(np.int16)
    Ftr, names = G.features(col(Ttr), Str, TStr); ytr = col(Ttr).astype(np.int64)
    tgt = np.flatnonzero((np.arange(len(Ttr)) > np.r_[0, np.where(Ktr[:-1] >= 0, TStr[:-1], 0)]) | (Ktr == 2))
    tgt = np.arange(len(Ttr))                                                 # 全位置を対象 (ソース部分のランダムな語も含める: 純度でふるい落とす)
    rc = RuleCover(maxk=cfg.get('maxk', 5), beam=cfg.get('beam', 8), purity=0.995, minn=3, sample=cfg.get('sample', 80000), rounds=cfg.get('rounds', 30))
    rc.fit(Ftr[tgt], ytr[tgt], names, log=log)
    def rules_on(T_, S_, TS_):
        F_, _ = G.features(col(T_), S_, TS_); pred, rid = rc.predict(F_)
        pred = np.where(pred == OTHER, -1, pred)                              # OTHER 予測 = 意見なし
        return pred
    pca = rules_on(Tca, Sca, TSca); fire = pca >= 0
    # 適合率 r は 領域ごとではなく 全体で 1 つ (領域を教えない)
    r = float((pca[fire] == Tca[fire]).mean()); log('規則の適合率 r=%.4f (別集合で %d 語に発火)' % (r, fire.sum()))
    pte = rules_on(Tte, Ste, TSte); firet = pte >= 0
    P_lm = lm.prob(Tte.astype(np.int64), Ste)[0]
    P_mix = np.where(firet, np.where(pte == Tte, r + (1 - r) * P_lm, (1 - r) * P_lm), P_lm)
    res = {}
    for nm, k in (('ソース(ランダムな文)', 0), ('記録(決定的)', 1), ('文章', 2)):
        m = Kte == k
        b_lm = float(-np.log2(P_lm[m]).mean()); b_mix = float(-np.log2(P_mix[m]).mean())
        res[nm] = dict(n=int(m.sum()), lm=b_lm, mix=b_mix, fired=float(firet[m].mean()), fire_acc=float((pte[m & firet] == Tte[m & firet]).mean()) if (m & firet).any() else None)
        log('%s: LM のみ %.4f bit/語, 合流 %.4f bit/語 (規則が発火 %.4f, 発火時の正解 %s)' % (nm, b_lm, b_mix, res[nm]['fired'], res[nm]['fire_acc']))
    ball = {'lm': float(-np.log2(P_lm).mean()), 'mix': float(-np.log2(P_mix).mean())}; log('全体: LM のみ %.4f, 合流 %.4f' % (ball['lm'], ball['mix']))
    log('RESULT_G5 ' + json.dumps(dict(cfg=cfg, r=r, res=res, all=ball, rules=[[names[f] for f in s] for s, _ in rc.rules])))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
