"""
DLG2: 会話をつなげるための、勾配を使わない部品3つ。 対話データ (DailyDialog) だけで数える。
  A  ターンをまたぐ共起表 : 「直前のターンの語 -> 今のターンの語」(t-1 -> t) と (t-2 -> t) の PMI を別の表に数える。 窓の中の語を距離でなく「どのターンか」で分ける
  K  ターン単位の検索     : 直前のターン全体 (idf重みの座標の平均) が似ている過去のターンを引き、 その返事の語の分布を足す
                            knn_bag = 返事全体の語の分布, knn_pos = 返事の同じ位置の語の分布
  T  決定木の森           : 文脈への問い (直前のターンに語wがあるか, 直前が疑問文か, 3語前のクラスがcか, ...) を、 次の語のクラスの情報量が最大になるように貪欲に選ぶ。
                            葉と祖先の回数表を、 親へ向けて補間して確率にする。 勾配なし, 逆伝播なし。
  比べるもの : base(CB2全部入り) / +A / +K / +T / +A+K+T 、 対話のテスト窓の bit/語 (全体, ターン先頭, ターン内) と、 同じ問いかけへの生成例
使い方: python3 exp_dlg2.py '{"ntune":4000,"ntest":20000,"trees":8,"ntree":600000,"depth":14}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
import numpy as np, torch
import scipy.sparse as sp
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
import exp_dlg as D
from exp_cb import V, SEQ, LN2, log

SEPID = 1
KNN_K = 16; KNN_BETA = 20.0; LR = 24


# ---------------------------------------------------------------- ターンの分解 (窓 n x SEQ)
def turn_masks(X):
    isS = (X == SEPID); si = isS.long()
    after = torch.flip(torch.cumsum(torch.flip(si, [1]), 1), [1]) - si      # その位置より後ろにある区切りの数
    lead = torch.cumprod((X == 0).long(), 1).bool()                         # 左詰めの 0 (対話の前)
    content = (~isS) & (X != 0) & (~lead)
    cur_all = (after == 0) & (~isS) & (~lead)
    return dict(cur=content & (after == 0), p1=content & (after == 1), p2=content & (after == 2), curlen=cur_all.sum(1),
                prevlen=(((after == 1) & (~isS) & (~lead)).sum(1)), hasprev=((after == 1) & (~isS) & (~lead)).any(1), lead=lead, content=content)


def bag(X, mask, n):
    w = mask.float(); w = w / w.sum(1, keepdim=True).clamp(min=1)
    W = torch.zeros(n, V); W.scatter_add_(1, X, w); return W


# ---------------------------------------------------------------- A: ターンをまたぐ共起
def split_turns(ids):
    turns, cur = [], []
    for t in ids:
        cur.append(int(t))
        if t == SEPID:
            turns.append(cur); cur = []
    if cur:
        turns.append(cur)
    return turns


def build_cross(dialogs, back):
    keys, ws = [], []
    for d in dialogs:
        T = split_turns(d)
        for t in range(back, len(T)):
            a = np.array([w for w in T[t - back] if w > 1], np.int64); b = np.array(T[t], np.int64)
            if len(a) == 0:
                continue
            keys.append((a[:, None] * V + b[None, :]).ravel()); ws.append(np.full(len(a) * len(b), 1.0 / len(a), np.float32))
    k = np.concatenate(keys); w = np.concatenate(ws)
    return torch.from_numpy(np.bincount(k, weights=w, minlength=V * V).astype(np.float32)).view(V, V)


# ---------------------------------------------------------------- K: ターン単位の検索
def turn_vec(X, mask, E, idf):
    w = mask.float() * idf[X]
    q = (w[:, :, None] * E[X]).sum(1)
    return q / (q.norm(dim=1, keepdim=True) + 1e-9)


def build_knn(dialogs, E, idf):
    keys, reps = [], []
    for d in dialogs:
        T = split_turns(d)
        for t in range(1, len(T)):
            a = np.array([w for w in T[t - 1] if w > 1], np.int64)
            if len(a) == 0:
                continue
            v = (idf[torch.from_numpy(a)][:, None] * E[torch.from_numpy(a)]).sum(0)
            keys.append(v / (v.norm() + 1e-9))
            r = T[t][:LR]; reps.append(r + [V] * (LR - len(r)))
    return torch.stack(keys), torch.tensor(reps, dtype=torch.long)


def knn_feats(X, M, G, pu):
    n = len(X); q = turn_vec(X, M['p1'], G.E, G.idf)
    sim = q @ G.KK.T; sv, si = torch.topk(sim, KNN_K, dim=1); w = torch.softmax(KNN_BETA * sv, 1)
    rep = G.RR[si]                                                      # n x K x LR
    rl = (rep != V).sum(2).clamp(min=1).float()
    dist = torch.zeros(n, V + 1)
    dist.scatter_add_(1, rep.reshape(n, -1), (w / rl)[:, :, None].expand(-1, -1, LR).reshape(n, -1))
    kpos = M['curlen'].clamp(max=LR - 1)
    tok = rep[torch.arange(n)[:, None], torch.arange(KNN_K)[None, :], kpos[:, None]]
    dpos = torch.zeros(n, V + 1); dpos.scatter_add_(1, tok, w)
    hp = M['hasprev'][:, None]
    return (torch.log(torch.where(hp, dist[:, :V], pu[None, :]) + 1e-4), torch.log(torch.where(hp, dpos[:, :V], pu[None, :]) + 1e-4))


# ---------------------------------------------------------------- T: 決定木の森
class Quest:
    """文脈への問い (2値) の一覧。 列の並び: [cls(x-i)==c, i=1..3] [x-i==w(上位200), i=1..2] [前ターンにw(300)] [今ターンにw(200)] [直近32語にw(300)] [状態8個] [前ターン先頭語(60)] [前々ターンにw(200)]"""

    def __init__(self, rank_src, cls, qmark):
        c = Counter(int(t) for t in rank_src if t > 1); order = [w for w, _ in c.most_common(300)]
        self.cls = cls; self.qmark = qmark
        self.rk = {}
        for m in (60, 200, 300):
            r = torch.full((V,), -1, dtype=torch.long); r[torch.tensor(order[:m])] = torch.arange(m); self.rk[m] = r
        K = C2.K; self.off = {}; o = 0
        for nm, sz in (('cls', 3 * K), ('ex', 2 * 200), ('p1', 300), ('cur', 200), ('win', 300), ('st', 8), ('first', 60), ('p2', 200)):
            self.off[nm] = o; o += sz
        self.Q = o

    def dense(self, X, M):
        n = len(X); Qd = torch.zeros(n, self.Q, dtype=torch.bool); ar = torch.arange(n); K = C2.K
        for i in (1, 2, 3):
            Qd[ar, self.off['cls'] + (i - 1) * K + self.cls[X[:, SEQ - i]]] = True
        for i in (1, 2):
            r = self.rk[200][X[:, SEQ - i]]; ok = r >= 0; Qd[ar[ok], self.off['ex'] + (i - 1) * 200 + r[ok]] = True

        def setbag(mask, m, nm):
            r = self.rk[m][X]; ok = mask & (r >= 0); rr, cc = torch.nonzero(ok, as_tuple=True); Qd[rr, self.off[nm] + r[rr, cc]] = True
        setbag(M['p1'], 300, 'p1'); setbag(M['cur'], 200, 'cur'); setbag(M['p2'], 200, 'p2')
        w32 = M['content'].clone(); w32[:, :SEQ - 32] = False; setbag(w32, 300, 'win')
        k = M['curlen']; pl = M['prevlen']; o = self.off['st']
        st = torch.stack([k == 0, k <= 1, k <= 3, k <= 7, (X == self.qmark).__and__(M['p1']).any(1), (pl <= 3) & M['hasprev'], (pl <= 8) & M['hasprev'], M['hasprev']], 1)
        Qd[:, o:o + 8] = st
        has = M['p1'].any(1); fpos = M['p1'].float().argmax(1); fw = X[ar, fpos]; r = self.rk[60][fw]; ok = has & (r >= 0)
        Qd[ar[ok], self.off['first'] + r[ok]] = True
        return Qd


def ent(H, n):
    return np.where(H > 0, H * np.log(np.maximum(H, 1e-9) / np.maximum(n, 1e-9)[:, None]), 0.0).sum(1)


class Tree:
    def __init__(self):
        self.feat, self.left, self.right, self.n, self.cw = [], [], [], [], []

    def grow(self, Xs, cols, Ysp, yw, idx0, depth, min_leaf, K):
        stack = [(idx0, 0, self._new())]
        while stack:
            idx, d, nd = stack.pop(); self.n[nd] = len(idx)
            wu, cu = np.unique(yw[idx], return_counts=True); self.cw[nd] = (wu, cu.astype(np.float32))
            if d >= depth or len(idx) < 2 * min_leaf:
                continue
            Xn = Xs[idx]; Yn = Ysp[idx]
            H1 = (Xn.T @ Yn).toarray().astype(np.float64); tot = np.asarray(Yn.sum(0)).ravel().astype(np.float64)
            n1 = H1.sum(1); n0 = len(idx) - n1; H0 = tot[None, :] - H1
            gain = ent(H1, n1) + ent(H0, n0) - ent(tot[None, :], np.array([float(len(idx))]))[0]
            gain[(n1 < min_leaf) | (n0 < min_leaf)] = -1.0
            b = int(np.argmax(gain))
            if gain[b] <= 1e-6 * len(idx):
                continue
            col = np.asarray(Xn[:, b].todense()).ravel() > 0
            l, r = self._new(), self._new()
            self.feat[nd] = int(cols[b]); self.left[nd] = l; self.right[nd] = r
            stack.append((idx[~col], d + 1, l)); stack.append((idx[col], d + 1, r))

    def _new(self):
        self.feat.append(-1); self.left.append(-1); self.right.append(-1); self.n.append(0); self.cw.append(None); return len(self.feat) - 1

    def freeze(self):
        self.feat = np.array(self.feat); self.left = np.array(self.left); self.right = np.array(self.right); self.n = np.array(self.n, np.float32)
        self.cache = {}

    def vec(self, nd):
        v = self.cache.get(nd)
        if v is None:
            v = torch.zeros(V); wu, cu = self.cw[nd]; v[torch.from_numpy(wu)] = torch.from_numpy(cu); self.cache[nd] = v
        return v

    def predict(self, Qd, pu, lam, maxd):
        n = len(Qd); Qn = Qd.numpy(); cur = np.zeros(n, np.int64); path = [cur.copy()]
        for _ in range(maxd):
            f = self.feat[cur]; act = f >= 0
            if not act.any():
                break
            go = Qn[np.arange(n), np.where(act, f, 0)]
            cur = np.where(act, np.where(go, self.right[cur], self.left[cur]), cur); path.append(cur.copy())
        p = pu[None, :].repeat(n, 1)
        for lv, nodes in enumerate(path):
            act = np.ones(n, bool) if lv == 0 else nodes != path[lv - 1]
            for u in np.unique(nodes[act]):
                rows = torch.from_numpy(np.flatnonzero(act & (nodes == u)))
                p[rows] = (self.vec(int(u))[None, :] + lam * p[rows]) / (float(self.n[u]) + lam)
        return p


def build_forest(Qm, ycls, yw, ntrees, depth, min_leaf, rng, frac_cols=0.3):
    N, Qn = Qm.shape; K = C2.K
    Ysp = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), ycls)), shape=(N, K))
    forest = []
    for t in range(ntrees):
        t0 = time.time(); cols = np.sort(rng.choice(Qn, int(Qn * frac_cols), replace=False))
        Xs = Qm[:, cols].tocsr(); idx0 = np.sort(rng.choice(N, N // 2, replace=False))
        tr = Tree(); tr.grow(Xs, cols, Ysp, yw, idx0, depth, min_leaf, K); tr.freeze(); forest.append(tr)
        log('木 %d: 節 %d, 葉 %d, %.0f 秒' % (t, len(tr.feat), int((tr.feat < 0).sum()), time.time() - t0))
    return forest


# ---------------------------------------------------------------- 全部の特徴
def new_feats(X, G, bs=250):
    outs = {}
    for a in range(0, len(X), bs):
        Xc = X[a:a + bs]; n = len(Xc); M = turn_masks(Xc)
        outs.setdefault('xt1', []).append(bag(Xc, M['p1'], n) @ G.px1); outs.setdefault('xt2', []).append(bag(Xc, M['p2'], n) @ G.px2)
        kb, kp = knn_feats(Xc, M, G, G.pu); outs.setdefault('knn_bag', []).append(kb); outs.setdefault('knn_pos', []).append(kp)
        Qd = G.quest.dense(Xc, M)
        pm = sum(tr.predict(Qd, G.pu, G.lam, G.depth) for tr in G.forest) / len(G.forest)
        outs.setdefault('tree', []).append(torch.log(pm + 1e-9))
    return {k: torch.cat(v) for k, v in outs.items()}


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0); rng = np.random.default_rng(0)
    ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000); ntrees = cfg.get('trees', 8); ntree = cfg.get('ntree', 600000)
    depth = cfg.get('depth', 14); min_leaf = cfg.get('min_leaf', 40)
    dl = D.load_dialogs(); log('対話 学習 %d / 検証 %d / テスト %d 件' % (len(dl['train']), len(dl['validation']), len(dl['test'])))
    cnt = Counter(t for d in dl['train'] for t in d); cnt.pop(D.SEP, None)
    itos = ['<unk>', D.SEP] + [w for w, _ in cnt.most_common(V - 2)]; STOI = {w: i for i, w in enumerate(itos)}
    ids = {sp_: [np.array([STOI.get(t, 0) for t in d], np.int64) for d in dl[sp_]] for sp_ in dl}
    xd = np.concatenate(ids['train']); log('対話 %d 語, 未知語率 %.3f' % (len(xd), float((xd == 0).mean())))
    Xva, yva = D.windows(dl['validation'], STOI, ntune, 1); Xte, yte = D.windows(dl['test'], STOI, ntest, 2)
    G = D.build_model(xd); log('基本の表 完了')
    # 座標と idf
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0); U, S, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :64] * S[:64].sqrt()[None, :]; G.E = E / (E.norm(dim=1, keepdim=True) + 1e-9); del ppmi
    nturn = 0; df = np.zeros(V)
    for d in ids['train']:
        for T in split_turns(d):
            nturn += 1; df[np.unique(np.array(T))] += 1
    G.idf = torch.from_numpy(np.log(nturn / (1 + df)).astype(np.float32)).clamp(min=0)
    # A
    G.px1 = B.pmi_from_counts(build_cross(ids['train'], 1), G.pu, cfg.get('betax', 80.0)); G.px2 = B.pmi_from_counts(build_cross(ids['train'], 2), G.pu, cfg.get('betax', 80.0))
    # K
    G.KK, G.RR = build_knn(ids['train'], G.E, G.idf); log('ターン対 %d 個を索引化' % len(G.KK))
    # T
    G.quest = Quest(xd, G.cls, STOI.get('?', 0)); G.lam = cfg.get('lam', 8.0); G.depth = depth
    Xtr, ytr = D.windows(dl['train'], STOI, ntree, 3); t0 = time.time(); mats = []
    for a in range(0, len(Xtr), 50000):
        Xc = Xtr[a:a + 50000]; mats.append(sp.csr_matrix(G.quest.dense(Xc, turn_masks(Xc)).numpy().astype(np.uint8)))
    Qm = sp.vstack(mats).tocsr(); log('問い %d 個 x 学習位置 %d, 非零 %d, %.0f 秒' % (Qm.shape[1], Qm.shape[0], Qm.nnz, time.time() - t0))
    yw = ytr.numpy(); ycls = G.cls.numpy()[yw]
    G.forest = build_forest(Qm, ycls, yw, ntrees, depth, min_leaf, rng); del Qm, mats
    # 混合器
    t0 = time.time(); Fv, gv = C2.collect(Xva, G); Fv.update(new_feats(Xva, G)); log('調整窓の特徴 %.0f 秒, 専門家 %d 個' % (time.time() - t0, len(Fv)))
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    A_ = ['xt1', 'xt2']; K_ = ['knn_bag', 'knn_pos']; T_ = ['tree']
    variants = {'base': allx, '+A': allx + A_, '+K': allx + K_, '+T': allx + T_, '+A+K+T': allx + A_ + K_ + T_}
    fitted = {}
    for vn, nm in variants.items():
        w, Dm, b = C2.fit(Fv, nm, yva, gate=gv, gated=True); fitted[vn] = (nm, w, Dm); log('%-8s 調整窓 %.4f' % (vn, b))
    del Fv
    ce = {vn: [] for vn in fitted}
    for a in range(0, len(yte), 250):
        Xc = Xte[a:a + 250]; Fc, gc = C2.all_feats(Xc, G); Fc.update(new_feats(Xc, G, bs=250))
        with torch.no_grad():
            for vn, (nm, w, Dm) in fitted.items():
                ce[vn].append(Fn.cross_entropy(C2.logits(Fc, nm, w, gc, Dm), yte[a:a + 250], reduction='none') / LN2)
        if (a // 250) % 20 == 0:
            log('テスト %d / %d' % (a, len(yte)))
    first = Xte[:, -1] == SEPID; res = {}
    for vn in fitted:
        c = torch.cat(ce[vn]); res[vn] = dict(test=c.mean().item(), turn_first=c[first].mean().item(), turn_inner=c[~first].mean().item())
        log('テスト %-8s 全体 %.4f, ターン先頭 %.4f, ターン内 %.4f' % (vn, res[vn]['test'], res[vn]['turn_first'], res[vn]['turn_inner']))
    wfull = {n: round(float(fitted['+A+K+T'][1][i]), 3) for i, n in enumerate(fitted['+A+K+T'][0]) if n in A_ + K_ + T_}
    log('新しい部品の重み (+A+K+T): %s' % wfull)
    samples = {}
    for vn in ('base', '+A+K+T'):
        nm, w, Dm = fitted[vn]; torch.manual_seed(0); samples[vn] = []
        for pr in D.PROMPTS:
            toks = [STOI.get(t, 0) for t in D.TOK.findall(pr.lower())] + [SEPID]; seen = 0; out = []
            for _ in range(120):
                c = toks[-SEQ:]; X1 = torch.zeros(1, SEQ, dtype=torch.long); X1[0, SEQ - len(c):] = torch.tensor(c)
                with torch.no_grad():
                    Fc, gc = C2.all_feats(X1, G); Fc.update(new_feats(X1, G)); z = C2.logits(Fc, nm, w, gc, Dm)[0]
                z[0] = -1e9; vals, idx = torch.topk(z / 0.8, 40); nx = int(idx[torch.multinomial(torch.softmax(vals, 0), 1)])
                toks.append(nx); out.append(itos[nx])
                if nx == SEPID:
                    seen += 1
                    if seen >= 3:
                        break
            samples[vn].append(dict(prompt=pr, reply=' '.join(out))); log('[%s] 問い: %s\n    返事: %s' % (vn, pr, ' '.join(out)))
    log('RESULT_DLG2 ' + json.dumps(dict(cfg=cfg, res=res, weights=wfull, samples=samples)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
