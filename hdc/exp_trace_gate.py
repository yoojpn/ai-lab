"""
TRACE-GATE: 役割を教えずに、 モデル自身が「どの規則を使うか」を選べるか (関門1)。
  1. 役割ごとの規則 (exp_trace の自動探索) を 6 本得る。
  2. 学習位置ごとに 各規則の予測が当たったかを記録し、 直前3語 (off1..off3) の組ごとに「最も当たる規則」を数えて選ぶ (数え上げのゲート)。
  3. 評価: 役割を使わずに、 当てる対象 (ランダム語を除く決定的な語) の正解率、 行の完全一致率、 プログラム全体の完全一致率 (通常 / 長い)。
     ランダムな語 (行頭の変数名, 演算の変数名, const の数字) は 与えられるものとして扱う。
使い方: python3 exp_trace_gate.py '{"ntrain":30000,"nval":3000,"ntest":3000}'
"""
import sys, json, time
import numpy as np
import exp_trace as T


def since_feats(tok, start):
    """各語種 t について、 直前で t が最後に出てから何語たったか (上限40)。 語種に依らない一般の特徴。"""
    N = len(tok); t = tok.astype(np.int64); ar = np.arange(N); out = []
    for k in range(T.NT):
        pos = np.where(t == k, ar, -1); last = np.maximum.accumulate(pos)
        lp = np.full(N, -1, np.int64); lp[1:] = last[:-1]                      # i より前で最後の t の位置
        d = np.where((lp >= 0) & (lp >= start), ar - lp, 41)
        out.append(np.minimum(d, 41))
    return np.stack(out, 1).astype(np.int8)


def main(cfg):
    rng = np.random.default_rng(1); t0 = time.time()
    log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    train = T.make_set(rng, cfg['ntrain'], 4, 8, None); val = T.make_set(rng, cfg['nval'], 4, 8, None)
    test = T.make_set(rng, cfg['ntest'], 4, 8, None); long_ = T.make_set(rng, cfg['ntest'], 10, 14, None)
    F = {};
    for k, v in (('train', train), ('val', val), ('test', test), ('long', long_)):
        F[k], names = T.features(v[0], v[3])
    nf = len(names); rules = {}
    S = {k: since_feats(v[0], v[3]) for k, v in (('train', train), ('val', val), ('test', test), ('long', long_))}
    G = {k: np.concatenate([F[k][:, :4], S[k]], 1) for k in F}                  # ゲート候補: off1..off4 + since_t (語種ごと)
    gnames = names[:4] + ['since_' + T.TOKS[k] for k in range(T.NT)]
    for role in ['syntax', 'copy_name', 'lookup', 'copy_const', 'sum_ones', 'sum_tens']:
        rid = T.R[role]; ptr = np.flatnonzero(train[1] == rid)[:60000]; vpt = np.flatnonzero(val[1] == rid)
        acc, sub = T.search_role(F['train'][ptr], train[0][ptr].astype(np.int64), F['val'][vpt], val[0][vpt].astype(np.int64), nf, log=lambda s: None)
        allp = np.flatnonzero(train[1] == rid)
        tab = T.fit_table(T.encode(F['train'][allp], sub), train[0][allp].astype(np.int64)); dflt = int(np.bincount(train[0][allp].astype(np.int64)).argmax())
        rules[role] = (sub, tab, dflt); log('規則 %s: %s (検証 %.4f)' % (role, [names[s] for s in sub], acc))
    rl = list(rules)
    def preds(setname, pos):
        return np.stack([T.predict(rules[r][1], T.encode(F[setname][pos], rules[r][0]), rules[r][2])[0] for r in rl], 1)   # n x 6
    # --- ゲート: 直前3語の組 -> 最も当たる規則
    tp = np.flatnonzero(train[1] >= 0); Pt = preds('train', tp); yt = train[0][tp].astype(np.int64)
    ok = (Pt == yt[:, None])
    import itertools
    sub_tr = np.random.default_rng(0).choice(len(tp), min(len(tp), 300000), replace=False)
    vp = np.flatnonzero(val[1] >= 0)[:150000]; Pv = preds('val', vp); yv = val[0][vp].astype(np.int64); okv = (Pv == yv[:, None])
    NG = G['train'].shape[1]; gbest = None
    def route(subset):
        kt = np.zeros(len(sub_tr), np.int64)
        for q in subset:
            kt = kt * 64 + G['train'][tp[sub_tr], q].astype(np.int64)
        uk_, inv_ = np.unique(kt, return_inverse=True)
        sc = np.stack([np.bincount(inv_, weights=ok[sub_tr, r].astype(np.float64), minlength=len(uk_)) for r in range(len(rl))], 1)
        b_ = sc.argmax(1)
        kv = np.zeros(len(vp), np.int64)
        for q in subset:
            kv = kv * 64 + G['val'][vp, q].astype(np.int64)
        i_ = np.minimum(np.searchsorted(uk_, kv), len(uk_) - 1); h_ = uk_[i_] == kv
        ch = np.where(h_, b_[i_], 0)
        return float(okv[np.arange(len(vp)), ch].mean())
    for k in (1, 2, 3):
        bk = None
        for subset in itertools.combinations(range(NG), k):
            a_ = route(subset)
            if bk is None or a_ > bk[0]:
                bk = (a_, subset)
        log('ゲート 大きさ%d の最良 検証 %.4f %s' % (k, bk[0], [gnames[q] for q in bk[1]]))
        if gbest is None or bk[0] > gbest[0] + 1e-9:
            gbest = bk
        if bk[0] >= 0.9995:
            break
    gsub = gbest[1]
    def gkey(setname, pos):
        kk = np.zeros(len(pos), np.int64)
        for q in gsub:
            kk = kk * 64 + G[setname][pos, q].astype(np.int64)
        return kk
    kt = np.zeros(len(tp), np.int64)
    for q in gsub:
        kt = kt * 64 + G['train'][tp, q].astype(np.int64)
    uk, inv = np.unique(kt, return_inverse=True)
    score = np.zeros((len(uk), len(rl)))
    for r in range(len(rl)):
        score[:, r] = np.bincount(inv, weights=ok[:, r].astype(np.float64), minlength=len(uk))
    best = score.argmax(1)
    log('ゲートの文脈 %d 通り' % len(uk))
    def run(setname, dat):
        pos = np.flatnonzero(dat[1] >= 0); P = preds(setname, pos); k = gkey(setname, pos)
        i = np.minimum(np.searchsorted(uk, k), len(uk) - 1); hit = uk[i] == k
        choice = np.where(hit, best[i], 0)
        pred = P[np.arange(len(pos)), choice]; y = dat[0][pos].astype(np.int64)
        tok_ok = pred == y
        # 行・プログラムの完全一致: 当てる対象のあるトークンを、 行 (';' で区切る) とプログラム (start) ごとに集計
        start = dat[3][pos]; line_id = np.cumsum(dat[0].astype(np.int64) == T.ID[':='])[pos]   # 行の番号 (':=' ごと)
        bad_line = np.bincount(line_id, weights=(~tok_ok).astype(np.float64)); n_line = np.bincount(line_id)
        lines_ok = float(((bad_line == 0) & (n_line > 0)).sum() / (n_line > 0).sum())
        us, sinv = np.unique(start, return_inverse=True); bad_prog = np.bincount(sinv, weights=(~tok_ok).astype(np.float64))
        return float(tok_ok.mean()), lines_ok, float((bad_prog == 0).mean()), float(hit.mean())
    res = {}
    for nm, dat in (('通常', test), ('長い', long_)):
        a = run('test' if nm == '通常' else 'long', dat); res[nm] = dict(token=a[0], line=a[1], program=a[2], gate_hit=a[3])
        log('%s: 語の正解率 %.4f, 行の完全一致 %.4f, プログラム完全一致 %.4f (ゲートの文脈が既知 %.4f)' % (nm, *a))
    log('RESULT_GATE ' + json.dumps(dict(cfg=cfg, res=res, rules={r: [names[s] for s in rules[r][0]] for r in rules}, gate=[gnames[q] for q in gsub])))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
