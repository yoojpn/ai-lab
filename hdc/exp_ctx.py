"""E3: 長い文脈の3部品を足した 表引きの網 (層数固定)。 すべて文書の境界の内側だけを見る。
  部品1: 複数の時間幅の要約。 直前 32/128/512/2048 語の窓から 32 語ずつ等間隔に取って 語の行の平均 → 状態に足す
  部品2: 稀な語の枠。 直前 4096 語 (文書内) のうち 語の稀さ (-log 頻度) が大きい上位 32 語 → 稀さで重みづけした平均を状態に足す
  部品3: 連続キャッシュ。 直前 512 語 (文書内) の各位置 j について 「j の前の2語」 を鍵にして 今の前の2語と近いほど 位置 j の語をコピーする確率を持たせる
        最終確率 = (1-g) * 網の確率 + g * キャッシュの確率  (g は 一番近い位置の近さから決まる)
  評価: 文書内の位置 (0-8/8-64/64-512/512-) ごとの bit/語、 長い文脈の位置を別に多めに取った集合、 n-gram への上乗せ、 生成サンプル
使い方: python3 exp_ctx.py '{"L":4,"feats":[1,2,3],"ntr":20000000}'
"""
import sys, json, time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import engine as E

V = 32768
SPARSE = True    # GPU では False (密な勾配 + LazyAdam)
W1 = [32, 128, 512, 2048]; NS = 32; W2 = 4096; K2 = 32; MC = 512; DK = 32


class Net(nn.Module):
    def __init__(s, d, c, L, n, kk, k, feats, rar):
        super().__init__()
        s.d, s.c, s.L, s.n, s.kk, s.k, s.feats = d, c, L, n, kk, k, feats
        s.E = nn.Embedding(V, d, sparse=SPARSE); nn.init.normal_(s.E.weight, 0, 0.1)
        s.P = nn.Parameter(torch.randn(c, d) * 0.1 + 1.0)
        s.K1 = nn.ParameterList([nn.Parameter(torch.randn(n, d // 2) * 0.5) for _ in range(L)])
        s.K2 = nn.ParameterList([nn.Parameter(torch.randn(n, d // 2) * 0.5) for _ in range(L)])
        s.Val = nn.ModuleList([nn.Embedding(n * n, d, sparse=SPARSE) for _ in range(L)])
        for v in s.Val: nn.init.normal_(v.weight, 0, 0.02)
        s.tau = nn.Parameter(torch.ones(max(L, 1)))
        s.U = nn.Parameter(torch.randn(V, d) * 0.05)
        s.register_buffer('rar', torch.tensor(rar, dtype=torch.float32))
        if 1 in feats: s.G1 = nn.Parameter(torch.full((len(W1), d), 0.1))
        if 2 in feats: s.G2 = nn.Parameter(torch.full((d,), 0.1))
        if 3 in feats:
            s.Ek = nn.Embedding(V, DK, sparse=SPARSE); nn.init.normal_(s.Ek.weight, 0, 0.3)
            s.Pk = nn.Parameter(torch.ones(2, DK)); s.theta = nn.Parameter(torch.tensor(8.0)); s.ga = nn.Parameter(torch.tensor(1.0)); s.gb = nn.Parameter(torch.tensor(-2.0))

    def sparse_params(s):
        out = [s.E.weight] + [v.weight for v in s.Val]
        if 3 in s.feats: out.append(s.Ek.weight)
        return out

    def dense_params(s):
        sp = s.sparse_params(); return [p for p in s.parameters() if all(p is not q for q in sp)]

    @staticmethod
    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def hidden(s, inp):
        x = (s.E(inp['ctx']) * s.P[None]).sum(1)
        if 1 in s.feats:
            e = s.E(inp['f1_ids']); m = inp['f1_m'][..., None]                       # (B,4,NS,d), (B,4,NS,1)
            mean = (e * m).sum(2) / m.sum(2).clamp_min(1.0)                         # (B,4,d)
            x = x + (mean * s.G1[None]).sum(1)
        if 2 in s.feats:
            r = s.rar[inp['f2_ids']].masked_fill(~inp['f2_v'], -1.0)                # (B,4088)
            tv, ti = r.topk(K2, dim=1); w = tv.clamp_min(0.0)                       # 稀さ (無効は 0)
            ids = inp['f2_ids'].gather(1, ti)
            s2 = (s.E(ids) * w[..., None]).sum(1) / (w.sum(1, keepdim=True) + 1e-6)
            x = x + s.G2[None] * s2
        n, kk, k, h = s.n, s.kk, s.k, s.d // 2
        for l in range(s.L):
            q = s.rms(x); q1, q2 = q[:, :h], q[:, h:]
            s1 = q1 @ s.K1[l].t() / math.sqrt(h); s2_ = q2 @ s.K2[l].t() / math.sqrt(h)
            v1, i1 = s1.topk(kk, dim=1); v2, i2 = s2_.topk(kk, dim=1)
            comb = (v1[:, :, None] + v2[:, None, :]).flatten(1)
            sc, ix = comb.topk(k, dim=1)
            a = F.softmax(sc / s.tau[l].abs().clamp_min(0.05), dim=1)
            ia = i1.gather(1, ix // kk); ib = i2.gather(1, ix % kk); rows = ia * n + ib
            x = x + (a[:, :, None] * s.Val[l](rows)).sum(1)
        return s.rms(x)

    def cache(s, inp):
        kq = s.Pk[0] * s.Ek(inp['f3_c1']) + s.Pk[1] * s.Ek(inp['f3_c2'])           # (B,MC,DK)
        kq = F.normalize(kq, dim=-1)
        q = F.normalize(s.Pk[0] * s.Ek(inp['ctx'][:, 0]) + s.Pk[1] * s.Ek(inp['ctx'][:, 1]), dim=-1)
        sc = s.theta * (kq @ q[:, :, None])[:, :, 0]
        v = inp['f3_v']; any_ = v.any(1)
        sc = sc.masked_fill(~v, -1e9); a = F.softmax(sc, dim=1) * any_[:, None]
        mx = torch.minimum(sc.max(1).values.clamp_min(-1.0), s.theta.detach().abs() + 1.0)
        g = torch.sigmoid(s.ga * mx + s.gb) * any_
        return a, g

    def logp(s, inp, y):
        lp = F.log_softmax(F.linear(s.hidden(inp), s.U), dim=1).gather(1, y[:, None])[:, 0]
        if 3 not in s.feats: return lp
        a, g = s.cache(inp); pc = (a * (inp['f3_val'] == y[:, None]).float()).sum(1)
        return torch.log((1 - g) * lp.exp() + g * pc + 1e-12)

    def dist(s, inp):
        pm = F.softmax(F.linear(s.hidden(inp), s.U), dim=1)
        if 3 not in s.feats: return pm
        a, g = s.cache(inp); pcd = torch.zeros_like(pm).scatter_add_(1, inp['f3_val'], a)
        return (1 - g)[:, None] * pm + g[:, None] * pcd


class LazyAdam:
    """SparseAdam と同じ計算 (勾配が 0 でない行だけ更新) を 密な勾配で GPU 上で行う。"""
    def __init__(s, params, lr, betas=(0.9, 0.999), eps=1e-8):
        s.param_groups = [{'params': list(params), 'lr': lr}]; s.b1, s.b2 = betas; s.eps = eps; s.st = {}
        s.t = torch.zeros((), device=s.param_groups[0]['params'][0].device)   # 歩数 (GPU の上のテンソル: CUDA グラフに記録できる)

    def zero_grad(s):
        for g in s.param_groups:
            for p in g['params']: p.grad = None

    @torch.no_grad()
    def step(s):
        s.t.add_(1); bc1 = 1 - s.b1 ** s.t; bc2 = 1 - s.b2 ** s.t
        for g in s.param_groups:
            for p in g['params']:
                G = p.grad
                if G is None: continue
                if id(p) not in s.st: s.st[id(p)] = (torch.zeros_like(p), torch.zeros_like(p))
                m, v = s.st[id(p)]; mask = (G != 0).any(1, keepdim=True)
                m.copy_(torch.where(mask, s.b1 * m + (1 - s.b1) * G, m)); v.copy_(torch.where(mask, s.b2 * v + (1 - s.b2) * G * G, v))
                upd = (g['lr'] * bc2.sqrt() / bc1) * m / (v.sqrt() + s.eps)
                p.sub_(torch.where(mask, upd, torch.zeros_like(upd)))


def make_inputs_t(T, pos, ds, feats, offs8, o1, offs2):
    """make_inputs の torch 版 (全部 GPU の上)。 T: long テンソル、 pos/ds: long テンソル、 offs8/o1/offs2: long テンソル"""
    inp = {'ctx': T[(pos[:, None] - offs8[None]).clamp_min(0)]}
    if 1 in feats:
        ids, ms = [], []
        for o in o1:
            idx = pos[:, None] - o[None]; v = idx >= ds[:, None]; ids.append(T[idx.clamp_min(0)] * v); ms.append(v)
        inp['f1_ids'] = torch.stack(ids, 1); inp['f1_m'] = torch.stack(ms, 1).float()
    if 2 in feats:
        idx = pos[:, None] - offs2[None]; v = idx >= ds[:, None]
        inp['f2_ids'] = T[idx.clamp_min(0)] * v; inp['f2_v'] = v
    if 3 in feats:
        j = pos[:, None] - torch.arange(1, MC + 1, device=T.device)[None]; v = (j - 2) >= ds[:, None]
        inp['f3_val'] = T[j.clamp_min(0)] * v; inp['f3_c1'] = T[(j - 1).clamp_min(0)] * v; inp['f3_c2'] = T[(j - 2).clamp_min(0)] * v; inp['f3_v'] = v
    return inp


DEV = torch.device('cpu')   # main() が cfg の device か、 GPU があれば cuda に切り替える


def make_inputs(T, pos, ds, feats, offs8, o1, offs2):
    """T: int64 配列。 pos: 予測する位置 (B,)。 ds: その位置の文書の開始 (B,)。 文書の外は 0 (無効)"""
    inp = {'ctx': torch.from_numpy(T[np.maximum(pos[:, None] - offs8[None], 0)])}
    if 1 in feats:
        ids, ms = [], []
        for o in o1:
            idx = pos[:, None] - o[None]; v = idx >= ds[:, None]; ids.append(T[np.maximum(idx, 0)] * v); ms.append(v)
        inp['f1_ids'] = torch.from_numpy(np.stack(ids, 1)); inp['f1_m'] = torch.from_numpy(np.stack(ms, 1).astype(np.float32))
    if 2 in feats:
        idx = pos[:, None] - offs2[None]; v = idx >= ds[:, None]
        inp['f2_ids'] = torch.from_numpy(T[np.maximum(idx, 0)] * v); inp['f2_v'] = torch.from_numpy(v)
    if 3 in feats:
        j = pos[:, None] - np.arange(1, MC + 1)[None]; v = (j - 2) >= ds[:, None]
        inp['f3_val'] = torch.from_numpy(T[np.maximum(j, 0)] * v); inp['f3_c1'] = torch.from_numpy(T[np.maximum(j - 1, 0)] * v)
        inp['f3_c2'] = torch.from_numpy(T[np.maximum(j - 2, 0)] * v); inp['f3_v'] = torch.from_numpy(v)
    return {k_: v_.to(DEV) for k_, v_ in inp.items()}


def main(cfg):
    global MC, W2, NS, W1, DEV, SPARSE
    DEV = torch.device(cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    t0 = time.time(); lg = lambda m: print('[%6ds] %s' % (time.time() - t0, m), flush=True)
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(cfg.get('seed', 0)); rng = np.random.default_rng(cfg.get('seed', 0))
    MC = cfg.get('mc', MC); W2 = cfg.get('w2', W2)
    W = '/tmp/tok'; d = cfg.get('d', 128); c = cfg.get('c', 8); L = cfg['L']; n = cfg.get('n', 256); kk = cfg.get('kk', 4); k = cfg.get('k', 4); B = cfg.get('B', 512)
    feats = list(cfg.get('feats', [])); ntr = cfg['ntr']; lr = cfg.get('lr', 2e-3); lrs = cfg.get('lrs', 5e-3)
    offs8 = np.arange(1, c + 1); o1 = [1 + (np.arange(NS) * w) // NS for w in W1]; offs2 = np.arange(c + 1, W2 + 1)
    lp_ = W + '/lam.json'; lam = {int(a): b for a, b in json.load(open(lp_)).items()} if os.path.exists(lp_) else {2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}
    M = E.Model({o: '%s/final_o%d' % (W, o) for o in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam, kn_base=True)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(W + '/tok.json'); lg('事前確認: n-gram 表と tokenizer は使えます。 feats=%s' % feats)
    Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); Dtr = np.load(W + '/train_docs.npy'); nb = 40; bl = ntr // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(c, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]).astype(np.int64); N = len(S)
    u = np.arange(N); ci = np.clip((u - c) // bl, 0, nb - 1); loc = u - c - ci * bl; ptr = starts[ci] + loc
    dsT = np.maximum(Dtr[np.maximum(np.searchsorted(Dtr, ptr, 'right') - 1, 0)], starts[ci]); DS = c + ci * bl + (dsT - starts[ci]); DS[:c] = u[:c]; del u, ci, loc, ptr, dsT
    cnt = np.bincount(S, minlength=V).astype(np.float64); pu = (cnt + 1) / (cnt.sum() + V); rar = -np.log2(pu); rar[0] = 0.0; rar[1] = 0.0
    pos = np.arange(c, N); pos = pos[S[pos] != 0]; rng.shuffle(pos); steps = len(pos) // B
    fast = (DEV.type == 'cuda' and int(cfg.get('fast', 1)) == 1) or int(cfg.get('fast', 1)) == 2; SPARSE = not fast   # fast=2: CPU で速い経路の試験
    net = Net(d, c, L, n, kk, k, feats, rar).to(DEV); npar = sum(p.numel() for p in net.parameters())
    lg('L=%d feats=%s B=%d 学習語 %d 歩数 %d パラメータ %.1fM 窓: 部品2=%d 部品3=%d' % (L, feats, B, len(pos), steps, npar / 1e6, W2, MC))
    use_graph = bool(fast and DEV.type == 'cuda' and int(cfg.get('graph', 1)))
    if use_graph:
        opt_s = LazyAdam(net.sparse_params(), lr=torch.tensor(lrs, device=DEV)); opt_d = torch.optim.Adam(net.dense_params(), lr=torch.tensor(lr, device=DEV), capturable=True)
    else:
        opt_s = LazyAdam(net.sparse_params(), lr=lrs) if fast else torch.optim.SparseAdam(net.sparse_params(), lr=lrs); opt_d = torch.optim.Adam(net.dense_params(), lr=lr)
    warm = cfg.get('warm', 200)
    if fast:
        S_t = torch.from_numpy(S).to(DEV); DS_t = torch.from_numpy(DS).to(DEV); pos_t = torch.from_numpy(pos).to(DEV)
        offs8_t = torch.from_numpy(offs8).to(DEV); o1_t = [torch.from_numpy(o).to(DEV) for o in o1]; offs2_t = torch.from_numpy(offs2).to(DEV)
        lg('GPU の速い経路: 入力の準備を GPU で行い、 疎な更新は LazyAdam')

    def nn_logp(T, p, ds, bs=1024):
        out = np.zeros(len(p)); net.eval()
        with torch.no_grad():
            for a in range(0, len(p), bs):
                pp = p[a:a + bs]; inp = make_inputs(T, pp, ds[a:a + bs], feats, offs8, o1, offs2); y = torch.from_numpy(T[pp]).to(DEV)
                out[a:a + bs] = (net.logp(inp, y) / math.log(2)).cpu().numpy()
        net.train(); return out

    def load_split(sp_):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16).astype(np.int64); D = np.load('%s/%s_docs.npy' % (W, sp_)); return T, D
    def ds_of(D, p): return D[np.maximum(np.searchsorted(D, p, 'right') - 1, 0)]
    def evalset(T, D, nn_, nlong):
        p = rng.integers(16 + c, len(T), int(nn_ * 1.02)); p = p[T[p] != 0][:nn_]
        q = rng.integers(16 + c, len(T), 4000000); q = q[(T[q] != 0) & ((q - ds_of(D, q)) >= 256)][:nlong]; return np.r_[p, q], len(p)
    ev = {}
    for sp_, nn_ in (('val', cfg['nval']), ('test', cfg['ntest'])):
        T, D = load_split(sp_); p, nmain = evalset(T, D, nn_, cfg.get('nlong', 40000)); ev[sp_] = dict(T=T, D=D, p=p, nmain=nmain, ds=ds_of(D, p))
    offs = np.arange(1, c + 1); curve = []; run = 0.0; cnt_ = 0; ts = time.time()
    if use_graph:
        pos_buf = pos_t[:B].clone(); run_t = torch.zeros((), device=DEV); cg = None; side = torch.cuda.Stream()
        def g_step():
            inp = make_inputs_t(S_t, pos_buf, DS_t[pos_buf], feats, offs8_t, o1_t, offs2_t); y = S_t[pos_buf]
            loss = -net.logp(inp, y).mean(); loss.backward(); opt_s.step(); opt_d.step(); run_t.add_(loss.detach() / math.log(2)); return loss
    for st in range(steps):
        f = min(1.0, (st + 1) / warm) * max(0.1, 1 - st / steps)
        if use_graph:
            opt_d.param_groups[0]['lr'].fill_(lr * f); opt_s.param_groups[0]['lr'].fill_(lrs * f)
            pos_buf.copy_(pos_t[st * B:(st + 1) * B])
            if st < 3:                                   # 準備運転 (最初の 3 歩は 通常の経路で、 横の流れで行う。 実際の学習の歩としても数える)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    opt_s.zero_grad(); opt_d.zero_grad(); g_step()
                torch.cuda.current_stream().wait_stream(side)
            else:
                if cg is None:                           # 1 歩を記録する (記録に失敗したら 通常の速い経路に戻す)
                    try:
                        torch.cuda.synchronize(); opt_s.zero_grad(); opt_d.zero_grad(); cg = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cg): loss_g = g_step()
                    except Exception as e_:
                        lg('CUDA グラフは使えませんでした (%r)。 通常の速い経路に戻します' % (e_,)); use_graph = False; cg = None; torch.cuda.synchronize(); run = float(run_t); continue
                    with torch.no_grad():                # 自己確認: 同じ重み・同じ入力で、 通常の経路の損失と記録の再生の損失が一致するか
                        l_e = float(-net.logp(make_inputs_t(S_t, pos_buf, DS_t[pos_buf], feats, offs8_t, o1_t, offs2_t), S_t[pos_buf]).mean())
                    cg.replay(); l_g = float(loss_g)
                    lg('CUDA グラフの自己確認: 通常 %.5f | 再生 %.5f (差 %.1e)' % (l_e, l_g, abs(l_e - l_g)))
                    if abs(l_e - l_g) > 2e-3: raise RuntimeError('CUDA グラフの再生が通常の経路と合いません')
                else:
                    cg.replay()
            cnt_ += 1
        else:
            for g in opt_d.param_groups: g['lr'] = lr * f
            for g in opt_s.param_groups: g['lr'] = lrs * f
            if fast:
                pp_t = pos_t[st * B:(st + 1) * B]; inp = make_inputs_t(S_t, pp_t, DS_t[pp_t], feats, offs8_t, o1_t, offs2_t); y = S_t[pp_t]
            else:
                pp = pos[st * B:(st + 1) * B]; inp = make_inputs(S, pp, DS[pp], feats, offs8, o1, offs2); y = torch.from_numpy(S[pp]).to(DEV)
            loss = -net.logp(inp, y).mean()
            run = run + loss.detach() / math.log(2); cnt_ += 1
            opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); opt_s.step(); opt_d.step()
        if (st + 1) % cfg.get('log_every', 500) == 0:
            run = float(run_t) if use_graph else float(run)
            if use_graph: run_t.zero_()
            curve.append((st + 1, run / cnt_)); lg('歩 %d/%d: 先に予測した損失 %.4f bit/語 (直近) | %.2f 歩/秒' % (st + 1, steps, run / cnt_, cfg.get('log_every', 500) / (time.time() - ts))); run = 0.0; cnt_ = 0; ts = time.time()
    lg('ONLINE_CTX ' + json.dumps(dict(L=L, feats=feats, steps=steps, curve=curve, last10000=float(np.mean([v for _, v in curve[-5:]])) if curve else None), ensure_ascii=False))
    # ---- 生成サンプル (網だけ。 n-gram は混ぜない) ----
    try:
        Tv, Dv = ev['val']['T'], ev['val']['D']; Lv = np.load(W + '/val_lang.npy'); lens = np.diff(np.r_[Dv, len(Tv)]); cand = {0: [], 1: []}
        for i in rng.permutation(len(Dv))[:20000]:
            if lens[i] >= 60 and len(cand[int(Lv[i])]) < 3: cand[int(Lv[i])].append(i)
        net.eval()
        with torch.no_grad():
            for lang_, name in ((1, '日本語'), (0, '英語')):
                for i in cand[lang_][:3]:
                    toks = Tv[Dv[i]:Dv[i] + 20].tolist(); T_ = np.array([0] * c + toks, dtype=np.int64)
                    for _ in range(60):
                        inp = make_inputs(T_, np.array([len(T_)]), np.array([c]), feats, offs8, o1, offs2); pr = net.dist(inp)[0] ** (1 / 0.8); pr[0] = 0; tv, ti = pr.topk(40); pr = torch.zeros_like(pr).scatter_(0, ti, tv); pr = pr / pr.sum()
                        T_ = np.r_[T_, int(torch.multinomial(pr, 1))]
                    lg('GEN_SAMPLE[%s] 入力: %s || 続き: %s' % (name, tk.decode(toks).replace('\n', ' '), tk.decode(T_[c + 20:].tolist()).replace('\n', ' ')))
        net.train()
    except Exception as e:
        lg('生成サンプルは失敗: %r' % e)
    # ---- 評価 ----
    parts = {}
    for sp_ in ('val', 'test'):
        e_ = ev[sp_]; T = e_['T']; p = e_['p']; lp, hi = M.logprob_at(T.astype(np.uint16), p, return_hi=True)
        parts[sp_] = dict(ng=2.0 ** lp, hi=hi, nn=2.0 ** nn_logp(T, p, e_['ds']), uni=pu[T[p]], depth=p - e_['ds'], nmain=e_['nmain'])
    G = np.r_[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    bins = lambda pl: np.clip((-np.log2(np.maximum(pl, 1e-12)) // 2).astype(int), 0, 7)
    bits = lambda P, m=None: float(-np.log2(P if m is None else P[m]).mean())
    def fit_g(pv, lv):
        bv = bins(pv); gb = np.zeros(8)
        for b_ in range(8):
            mm = bv == b_
            if mm.sum() < 200: continue
            gb[b_] = G[int(np.argmax([np.log2((1 - g) * pv[mm] + g * lv[mm]).mean() for g in G]))]
        return gb
    V_, T_ = parts['val'], parts['test']; mv, mt = slice(0, V_['nmain']), slice(0, T_['nmain'])
    g1 = fit_g(V_['ng'][mv], V_['uni'][mv]); mix = lambda P, X, g: (1 - g[bins(P)]) * P + g[bins(P)] * X
    pv1 = mix(V_['ng'], V_['uni'], g1); pt1 = mix(T_['ng'], T_['uni'], g1)
    g2 = fit_g(pv1[mv], V_['nn'][mv]); pt2 = mix(pt1, T_['nn'], g2)
    # ---- 合法な混合: 重みを 「予測する前に分かること」 (n-gram が見つけた最高次数 hi) だけで決める ----
    # 上の mix は 重みの選び方 (bins) に 『正解の語の n-gram 確率』 を使っている = 答えを見て重みを決めている (漏れ)。 こちらは漏れなし。
    hv_, ht_ = V_['hi'].astype(int), T_['hi'].astype(int)
    def fit_ok(pv, lv, hb):
        gb = np.zeros(6)
        for b_ in range(1, 6):
            mm = hb == b_
            if mm.sum() < 200: continue
            gb[b_] = G[int(np.argmax([np.log2((1 - g) * pv[mm] + g * lv[mm]).mean() for g in G]))]
        return gb
    mixok = lambda P, X, g, hb: (1 - g[hb]) * P + g[hb] * X
    g1o = fit_ok(V_['ng'][mv], V_['uni'][mv], hv_[mv]); pv1o = mixok(V_['ng'], V_['uni'], g1o, hv_); pt1o = mixok(T_['ng'], T_['uni'], g1o, ht_)
    g2o = fit_ok(pv1o[mv], V_['nn'][mv], hv_[mv]); pt2o = mixok(pt1o, T_['nn'], g2o, ht_)
    lg('合法な混合(重みは hi だけで決める): n-gram %.4f -> 較正 %.4f -> 混合 %.4f (上乗せ %.4f) | 長い文脈: 較正 %.4f -> 混合 %.4f (上乗せ %.4f) | 漏れありの上乗せ %.4f' % (
        bits(T_['ng'][mt]), bits(pt1o[mt]), bits(pt2o[mt]), bits(pt1o[mt]) - bits(pt2o[mt]), bits(pt1o[T_['nmain']:]), bits(pt2o[T_['nmain']:]), bits(pt1o[T_['nmain']:]) - bits(pt2o[T_['nmain']:]), bits(pt1[mt]) - bits(pt2[mt])))
    dep = T_['depth']; un = T_['hi'][mt] <= 2
    buckets = [(0, 8), (8, 64), (64, 512), (512, 10 ** 9)]; per = {}
    for lo, hi_ in buckets:
        m = np.zeros(len(dep), bool); m[mt] = (dep[mt] >= lo) & (dep[mt] < hi_)
        if m.sum() >= 200: per['%d-%d' % (lo, hi_)] = dict(n=int(m.sum()), nn=bits(T_['nn'], m), base=bits(pt1, m), final=bits(pt2, m), gain=bits(pt1, m) - bits(pt2, m))
    ml = np.zeros(len(dep), bool); ml[T_['nmain']:] = True
    res = dict(L=L, feats=feats, params_M=npar / 1e6, steps=steps, online_curve=curve, test_nn_bits=bits(T_['nn'][mt]), test_ngram_bits=bits(T_['ng'][mt]), test_base_bits=bits(pt1[mt]), test_final_bits=bits(pt2[mt]), extra2=bits(pt1[mt]) - bits(pt2[mt]),
               ok=dict(base=bits(pt1o[mt]), final=bits(pt2o[mt]), extra2=bits(pt1o[mt]) - bits(pt2o[mt]), long_base=bits(pt1o[T_['nmain']:]), long_final=bits(pt2o[T_['nmain']:])),
               unseen_frac=float(un.mean()), unseen_nn_bits=bits(T_['nn'][mt], un), unseen_base_bits=bits(pt1[mt], un), unseen_final_bits=bits(pt2[mt], un), by_depth=per,
               long256=dict(n=int(ml.sum()), nn=bits(T_['nn'], ml), ngram=bits(T_['ng'], ml), base=bits(pt1, ml), final=bits(pt2, ml), gain=bits(pt1, ml) - bits(pt2, ml)), train_seconds=time.time() - t0)
    lg('結果: 単独 %.4f | n-gram %.4f -> 較正 %.4f -> 混合 %.4f (上乗せ %.4f) | 初出文脈(%.0f%%) 単独 %.4f' % (res['test_nn_bits'], res['test_ngram_bits'], res['test_base_bits'], res['test_final_bits'], res['extra2'], 100 * res['unseen_frac'], res['unseen_nn_bits']))
    lg('長い文脈(文書内256語以降, n=%d): 単独 %.4f | n-gram較正 %.4f -> 混合 %.4f (上乗せ %.4f)' % (res['long256']['n'], res['long256']['nn'], res['long256']['base'], res['long256']['final'], res['long256']['gain']))
    for kx, vx in per.items(): lg('  文書内位置 %s (n=%d): 単独 %.4f | 較正 %.4f -> 混合 %.4f (上乗せ %.4f)' % (kx, vx['n'], vx['nn'], vx['base'], vx['final'], vx['gain']))
    lg('RESULT_CTX ' + json.dumps(dict(cfg=cfg, res=res), ensure_ascii=False))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
