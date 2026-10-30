"""
DLG3: (1) 丸暗記を除いた「きれいな」測り直し, (2) 検索(K)の強化, (3) 予測クラス。 DailyDialog だけ。
  きれい : 検証/テストの対話のうち、 長い発言 (5語以上) が1つでも訓練と完全一致するものを除く。 混合器の調整も きれいな検証窓だけで行う。
           訓練側も、 最初の3発言が同じ重複対話は1つにする。
  検索   : k16 = 近傍16 (DLG2と同じ), k64 = 近傍64, kc = 近傍の選び方に「今の返事の途中の語」も使う (gamma=1),
           kt = 2ターン前の発言も使う (delta=0.5), kct = 両方
  クラス : cfg['pcls'] = true なら 語のクラス (c123/wc/cc4/回路の class 版) を 予測でまとめるクラスに差し替える
使い方: python3 exp_dlg3.py '{"ntune":4000,"ntest":20000,"trees":8,"ntree":600000,"depth":14,"pcls":false}'
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
import exp_dlg as D
from exp_dlg2 import turn_masks, bag, split_turns, build_cross, turn_vec, Quest, build_forest, SEPID, LR
from pcls import predictive_classes
from exp_cb import V, SEQ, LN2, log

KV = {'k16': (16, 20.0, 0.0, 0.0), 'k64': (64, 20.0, 0.0, 0.0), 'kc': (16, 20.0, 1.0, 0.0), 'kt': (16, 20.0, 0.0, 0.5), 'kct': (16, 20.0, 1.0, 0.5)}


def long_turns(d):
    out, cur = [], []
    for t in d:
        if t == D.SEP:
            out.append(tuple(cur)); cur = []
        else:
            cur.append(t)
    if cur:
        out.append(tuple(cur))
    return [t for t in out if len(t) >= 5]


def build_knn3(dialogs, E, idf):
    k1, k2, kr, reps = [], [], [], []
    z = torch.zeros(E.shape[1])

    def vec(tokens):
        a = np.array([w for w in tokens if w > 1], np.int64)
        if len(a) == 0:
            return z
        ta = torch.from_numpy(a); v = (idf[ta][:, None] * E[ta]).sum(0)
        return v / (v.norm() + 1e-9)
    for d in dialogs:
        T = split_turns(d)
        for t in range(1, len(T)):
            a = vec(T[t - 1])
            if not a.any():
                continue
            k1.append(a); k2.append(vec(T[t - 2]) if t >= 2 else z); kr.append(vec(T[t]))
            r = T[t][:LR]; reps.append(r + [V] * (LR - len(r)))
    return torch.stack(k1), torch.stack(k2), torch.stack(kr), torch.tensor(reps, dtype=torch.long)


def knn3(X, M, G, K, beta, gamma, delta):
    n = len(X); s = turn_vec(X, M['p1'], G.E, G.idf) @ G.KK.T
    if delta:
        s = s + delta * (turn_vec(X, M['p2'], G.E, G.idf) @ G.KK2.T)
    if gamma:
        hc = ((M['curlen'] > 0) & M['cur'].any(1)).float()[:, None]
        s = s + gamma * hc * (turn_vec(X, M['cur'], G.E, G.idf) @ G.KR.T)
    sv, si = torch.topk(s, K, dim=1); w = torch.softmax(beta * sv / (1.0 + gamma + delta), 1)
    rep = G.RR[si]; rl = (rep != V).sum(2).clamp(min=1).float()
    dist = torch.zeros(n, V + 1)
    dist.scatter_add_(1, rep.reshape(n, -1), (w / rl)[:, :, None].expand(-1, -1, LR).reshape(n, -1))
    kpos = M['curlen'].clamp(max=LR - 1)
    tok = rep[torch.arange(n)[:, None], torch.arange(K)[None, :], kpos[:, None]]
    dpos = torch.zeros(n, V + 1); dpos.scatter_add_(1, tok, w)
    hp = M['hasprev'][:, None]
    return (torch.log(torch.where(hp, dist[:, :V], G.pu[None, :]) + 1e-4), torch.log(torch.where(hp, dpos[:, :V], G.pu[None, :]) + 1e-4))


def new_feats3(X, G, bs=250):
    outs = {}
    for a in range(0, len(X), bs):
        Xc = X[a:a + bs]; n = len(Xc); M = turn_masks(Xc)
        outs.setdefault('xt1', []).append(bag(Xc, M['p1'], n) @ G.px1); outs.setdefault('xt2', []).append(bag(Xc, M['p2'], n) @ G.px2)
        for nm, (K, beta, gamma, delta) in KV.items():
            kb, kp = knn3(Xc, M, G, K, beta, gamma, delta)
            outs.setdefault(nm + '_bag', []).append(kb); outs.setdefault(nm + '_pos', []).append(kp)
        Qd = G.quest.dense(Xc, M)
        pm = sum(tr.predict(Qd, G.pu, G.lam, G.depth) for tr in G.forest) / len(G.forest)
        outs.setdefault('tree', []).append(torch.log(pm + 1e-9))
    return {k: torch.cat(v) for k, v in outs.items()}


def build_model3(x, pcls, K=200, beta=1280.0):
    G = B.Model(); C2.K = K
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = B.build_counts(x)
    G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float(); G.tri = B.build_tri(x)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, beta) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, S, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :64] * S[:64].sqrt()[None, :]; G.E = E / (E.norm(dim=1, keepdim=True) + 1e-9); del C, ppmi
    km = B.kmeans(G.E, K)
    G.cls = predictive_classes(G.C1, K, km, log=log) if pcls else km
    G.sp = {nm: C2.build_sparse(x, G.cls.numpy(), spec, C2.MINCNT.get(nm, 1)) for nm, spec in C2.SPECS.items()}
    return G


def main(cfg):
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0); rng = np.random.default_rng(0)
    ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000); ntrees = cfg.get('trees', 8); ntree = cfg.get('ntree', 600000)
    depth = cfg.get('depth', 14); min_leaf = cfg.get('min_leaf', 40); pcls = cfg.get('pcls', False)
    dl = D.load_dialogs()
    # --- きれいな分割
    trainT = set(t for d in dl['train'] for t in long_turns(d))
    seen = set(); keep = []
    for d in dl['train']:
        k = tuple(split_turns_tok(d)[:3])
        if k in seen:
            continue
        seen.add(k); keep.append(d)
    log('訓練対話 %d -> 重複を除いて %d' % (len(dl['train']), len(keep))); dl['train'] = keep
    clean = {}
    for sp_ in ('validation', 'test'):
        fr = np.array([sum(t in trainT for t in long_turns(d)) / max(len(long_turns(d)), 1) for d in dl[sp_]])
        clean[sp_] = [d for d, f in zip(dl[sp_], fr) if f == 0]
        log('%s: 対話 %d のうち、 訓練と完全一致する長い発言が無い対話 %d' % (sp_, len(dl[sp_]), len(clean[sp_])))
    cnt = Counter(t for d in dl['train'] for t in d); cnt.pop(D.SEP, None)
    itos = ['<unk>', D.SEP] + [w for w, _ in cnt.most_common(V - 2)]; STOI = {w: i for i, w in enumerate(itos)}
    ids = [np.array([STOI.get(t, 0) for t in d], np.int64) for d in dl['train']]
    xd = np.concatenate(ids); log('対話 %d 語, 未知語率 %.3f, 予測クラス %s' % (len(xd), float((xd == 0).mean()), pcls))
    Xva, yva = D.windows(clean['validation'], STOI, ntune, 1)
    Xte_f, yte_f = D.windows(dl['test'], STOI, ntest, 2); Xte_c, yte_c = D.windows(clean['test'], STOI, ntest, 2)
    log('調整窓(きれい) %d, テスト窓 全体 %d / きれい %d' % (len(yva), len(yte_f), len(yte_c)))
    G = build_model3(xd, pcls); log('基本の表 完了')
    nturn = 0; df = np.zeros(V)
    for d in ids:
        for T in split_turns(d):
            nturn += 1; df[np.unique(np.array(T))] += 1
    G.idf = torch.from_numpy(np.log(nturn / (1 + df)).astype(np.float32)).clamp(min=0)
    G.px1 = B.pmi_from_counts(build_cross(ids, 1), G.pu, 80.0); G.px2 = B.pmi_from_counts(build_cross(ids, 2), G.pu, 80.0)
    G.KK, G.KK2, G.KR, G.RR = build_knn3(ids, G.E, G.idf); log('ターン対 %d 個を索引化' % len(G.KK))
    G.quest = Quest(xd, G.cls, STOI.get('?', 0)); G.lam = 8.0; G.depth = depth
    Xtr, ytr = D.windows(dl['train'], STOI, ntree, 3); mats = []
    import scipy.sparse as sp
    for a in range(0, len(Xtr), 50000):
        Xc = Xtr[a:a + 50000]; mats.append(sp.csr_matrix(G.quest.dense(Xc, turn_masks(Xc)).numpy().astype(np.uint8)))
    Qm = sp.vstack(mats).tocsr(); yw = ytr.numpy(); ycls = G.cls.numpy()[yw]
    G.forest = build_forest(Qm, ycls, yw, ntrees, depth, min_leaf, rng); del Qm, mats
    t0 = time.time(); Fv, gv = C2.collect(Xva, G); Fv.update(new_feats3(Xva, G)); log('調整窓の特徴 %.0f 秒, 専門家 %d 個' % (time.time() - t0, len(Fv)))
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    A_ = ['xt1', 'xt2']; T_ = ['tree']
    kn = lambda nm: [nm + '_bag', nm + '_pos']
    variants = {'base': allx, '+A+T': allx + A_ + T_}
    for nm in KV:
        variants['+A+%s+T' % nm] = allx + A_ + kn(nm) + T_
    variants['+A+allK+T'] = allx + A_ + sum([kn(nm) for nm in KV], []) + T_
    if cfg.get('only'):
        variants = {k: v for k, v in variants.items() if k in cfg['only']}
    fitted = {}
    for vn, nm in variants.items():
        w, Dm, b = C2.fit(Fv, nm, yva, gate=gv, gated=True); fitted[vn] = (nm, w, Dm); log('%-14s 調整窓(きれい) %.4f' % (vn, b))
    del Fv
    res = {}
    for tag, Xte, yte in ((() if cfg.get('skip_test') else (('full', Xte_f, yte_f), ('clean', Xte_c, yte_c)))):
        ce = {vn: [] for vn in fitted}
        for a in range(0, len(yte), 250):
            Xc = Xte[a:a + 250]; Fc, gc = C2.all_feats(Xc, G); Fc.update(new_feats3(Xc, G))
            with torch.no_grad():
                for vn, (nm, w, Dm) in fitted.items():
                    ce[vn].append(Fn.cross_entropy(C2.logits(Fc, nm, w, gc, Dm), yte[a:a + 250], reduction='none') / LN2)
        first = Xte[:, -1] == SEPID
        for vn in fitted:
            c = torch.cat(ce[vn]); res.setdefault(vn, {})[tag] = dict(test=c.mean().item(), turn_first=c[first].mean().item(), turn_inner=c[~first].mean().item())
        log('--- テスト窓 %s (%d) ---' % (tag, len(yte)))
        for vn in fitted:
            r = res[vn][tag]; log('%-14s 全体 %.4f, ターン先頭 %.4f, ターン内 %.4f' % (vn, r['test'], r['turn_first'], r['turn_inner']))
    best = min((v for v in fitted if v != 'base'), key=lambda v: res[v]['clean']['test']) if res else '+A+allK+T'
    if res:
        log('きれいなテストで最良: %s' % best)
    samples = {}
    for vn in ('base', best):
        nm, w, Dm = fitted[vn]; torch.manual_seed(0); samples[vn] = []
        for pr in cfg.get('prompts', D.PROMPTS):
            toks = [STOI.get(t, 0) for t in D.TOK.findall(pr.lower())] + [SEPID]; seen_s = 0; out = []
            for _ in range(120):
                c = toks[-SEQ:]; X1 = torch.zeros(1, SEQ, dtype=torch.long); X1[0, SEQ - len(c):] = torch.tensor(c)
                with torch.no_grad():
                    Fc, gc = C2.all_feats(X1, G); Fc.update(new_feats3(X1, G)); z = C2.logits(Fc, nm, w, gc, Dm)[0]
                z[0] = -1e9; vals, idx = torch.topk(z / 0.8, 40); nx = int(idx[torch.multinomial(torch.softmax(vals, 0), 1)])
                toks.append(nx); out.append(itos[nx])
                if nx == SEPID:
                    seen_s += 1
                    if seen_s >= 3:
                        break
            samples[vn].append(dict(prompt=pr, reply=' '.join(out))); log('[%s] 問い: %s\n    返事: %s' % (vn, pr, ' '.join(out)))
    log('RESULT_DLG3 ' + json.dumps(dict(cfg=cfg, res=res, best=best, samples=samples)))


def split_turns_tok(d):
    out, cur = [], []
    for t in d:
        if t == D.SEP:
            out.append(tuple(cur)); cur = []
        else:
            cur.append(t)
    if cur:
        out.append(tuple(cur))
    return out


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
