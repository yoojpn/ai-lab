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



def _kq_eager(ew, vv, e0, p0, p1):
    kq = p0 * torch.where(vv, ew[:, 1:MC + 1], e0) + p1 * torch.where(vv, ew[:, 2:MC + 2], e0)
    return F.normalize(kq, dim=-1)
BF16 = False
CV = False
CHL = -1             # ctxh=l: l 層の後で 中間表現どうしで Attention (鍵と値が その位置の中間表現)。 -1 で なし
BAL = 0.0           # bal=a: ルーティングの偏りを減らす補助損失 (Switch 型: 選ばれた割合 × 平均確率) の係数。 0 で なし
HNG = 0             # hng=B: 直前2語・3語のハッシュ (2^B 行の表) で行を引いて x に足す (語の組み合わせを表引きの段階で作る)。 0 で なし
VFAC = 0            # vfac=R: Val の行 (ia,ib) を R 個の 共有部品の積の和 VA[ia]*VB[ib] で作る (行どうしで部品を共有)。 0 で なし
VFULL = True        # vfull=0: 全行の表 Val をやめて 共有部品だけにする (vfac>0 のとき)
CL = 0              # ctxl=N: 下の N 層で、 その層の中間表現から 文脈の語を引き直す (Attention を重ねる)
KQF = _kq_eager        # main() が cfg compile=1 のとき torch.compile 版に差し替える


class RowGather(torch.autograd.Function):
    """表の行を引く。 逆伝播では 大きな表の密な勾配を作らず、 触った行だけ sink (表と同じ形の 常に 0 の置き場) に足し込む。
    オプティマイザは 触った行だけ sink から読んで更新し、 その行を 0 に戻す (vsink=1)。"""
    @staticmethod
    def forward(ctx, W, idx, sink):
        ctx.save_for_backward(idx); ctx.sink = sink
        return W[idx]

    @staticmethod
    def backward(ctx, g):
        idx, = ctx.saved_tensors
        ctx.sink.index_add_(0, idx.reshape(-1), g.reshape(-1, g.shape[-1]).to(ctx.sink.dtype))
        return None, None, None


class Net(nn.Module):
    VSINK = False                                    # main() が cfg vsink=1 のとき True にして net.Vsink を用意する

    def _hn(s, inp):
        c_ = inp['ctx']; a, b, c3 = c_[:, 0], c_[:, 1], c_[:, 2]; HT = 1 << HNG
        h2 = (a * 1000003 + b * 7919 + 17) % HT; h3 = ((a * 1000003 + b * 7919) * 31 + c3 * 104729 + 5) % HT
        s._hrows = torch.cat([h2, h3]); return s.H(h2) + s.H(h3)

    def _val(s, l, rows):
        o = 0
        if len(s.Val): o = RowGather.apply(s.Val[l].weight, rows, s.Vsink[l]) if s.VSINK else s.Val[l](rows)
        if VFAC:
            ia = rows // s.n; ib = rows % s.n; o = o + (s.VA[l][:, ia] * s.VB[l][:, ib]).sum(0)
        return o

    def __init__(s, d, c, L, n, kk, k, feats, rar):
        super().__init__()
        s.d, s.c, s.L, s.n, s.kk, s.k, s.feats = d, c, L, n, kk, k, feats
        s.E = nn.Embedding(V, d, sparse=SPARSE); nn.init.normal_(s.E.weight, 0, 0.1)
        s.P = nn.Parameter(torch.randn(c, d) * 0.1 + 1.0)
        s.K1 = nn.ParameterList([nn.Parameter(torch.randn(n, d // 2) * 0.5) for _ in range(L)])
        s.K2 = nn.ParameterList([nn.Parameter(torch.randn(n, d // 2) * 0.5) for _ in range(L)])
        s.Val = nn.ModuleList([nn.Embedding(n * n, d, sparse=SPARSE) for _ in range(L)] if (VFULL or not VFAC) else [])
        for v in s.Val: nn.init.normal_(v.weight, 0, 0.02)
        if VFAC:
            f_ = (0.02 / VFAC ** 0.5) ** 0.5            # R 個の積の和の標準偏差が 0.02 になる大きさ
            s.VA = nn.ParameterList([nn.Parameter(torch.randn(VFAC, n, d) * f_) for _ in range(L)]); s.VB = nn.ParameterList([nn.Parameter(torch.randn(VFAC, n, d) * f_) for _ in range(L)])
        s._rows = [None] * L
        if HNG: s.H = nn.Embedding(1 << HNG, d, sparse=SPARSE); nn.init.normal_(s.H.weight, 0, 0.05); s._hrows = None
        s.tau = nn.Parameter(torch.ones(max(L, 1)))
        s.U = nn.Parameter(torch.randn(V, d) * 0.05)
        s.register_buffer('rar', torch.tensor(rar, dtype=torch.float32))
        if 1 in feats: s.G1 = nn.Parameter(torch.full((len(W1), d), 0.1))
        if 2 in feats: s.G2 = nn.Parameter(torch.full((d,), 0.1))
        if 3 in feats:
            s.Ek = nn.Embedding(V, DK, sparse=SPARSE); nn.init.normal_(s.Ek.weight, 0, 0.3)
            if CHL >= 0: s.Wqh = nn.Parameter(torch.randn(d, DK) * d ** -0.5); s.Wkh = nn.Parameter(torch.randn(d, DK) * d ** -0.5); s.Gh = nn.Parameter(torch.full((d,), 0.02)); s.thh = nn.Parameter(torch.tensor(4.0))
            if CL: s.Wq = nn.ParameterList([nn.Parameter(torch.randn(d, DK) * d ** -0.5) for _ in range(CL)]); s.Gl = nn.Parameter(torch.full((CL, d), 0.02)); s.thl = nn.Parameter(torch.full((CL,), 4.0))
            if CV: s.Gc = nn.Parameter(torch.full((d,), 0.1))                      # ctxv=1: キャッシュの点で 文脈の語の意味ベクトルを 足し合わせて 入力に加える
            s.Pk = nn.Parameter(torch.ones(2, DK)); s.theta = nn.Parameter(torch.tensor(8.0)); s.ga = nn.Parameter(torch.tensor(1.0)); s.gb = nn.Parameter(torch.tensor(-2.0))

    def sparse_params(s):
        out = [s.E.weight] + [v.weight for v in s.Val]
        if HNG: out.append(s.H.weight)
        if 3 in s.feats: out.append(s.Ek.weight)
        return out

    def dense_params(s):
        sp = s.sparse_params(); return [p for p in s.parameters() if all(p is not q for q in sp)]

    @staticmethod
    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def hidden(s, inp, cv=None):
        x = (s.E(inp['ctx']) * s.P[None]).sum(1)
        if HNG: x = x + s._hn(inp)
        if cv is not None: x = x + s.Gc[None] * cv
        if 1 in s.feats:
            ids1 = inp['f1_ids']; m1 = inp['f1_m']; w1 = m1 / m1.sum(2, keepdim=True).clamp_min(1.0)   # 重みつきの平均を 1 命令で (embedding_bag)
            mean = F.embedding_bag(ids1.reshape(-1, ids1.shape[2]), s.E.weight, per_sample_weights=w1.reshape(-1, ids1.shape[2]), mode='sum', sparse=s.E.sparse).view(ids1.shape[0], ids1.shape[1], -1)   # (B,4,d)
            x = x + (mean * s.G1[None]).sum(1)
        if 2 in s.feats:
            r = s.rar[inp['f2_ids']].masked_fill(~inp['f2_v'], -1.0)                # (B,4088)
            tv, ti = r.topk(K2, dim=1); w = tv.clamp_min(0.0)                       # 稀さ (無効は 0)
            ids = inp['f2_ids'].gather(1, ti)
            s2 = F.embedding_bag(ids, s.E.weight, per_sample_weights=w / (w.sum(1, keepdim=True) + 1e-6), mode='sum', sparse=s.E.sparse)
            x = x + s.G2[None] * s2
        n, kk, k, h = s.n, s.kk, s.k, s.d // 2
        for l in range(s.L):
            if l < CL and 3 in s.feats:                                              # 層ごとの Attention: 今の中間表現で 鍵 (直前2語から作った kq) を引き直し 文脈の語を足す
                kq = s._kq; qh = F.normalize(s.rms(x) @ s.Wq[l], dim=-1).to(kq.dtype)
                sc_ = s.thl[l] * (kq @ qh[:, :, None])[:, :, 0].float(); v_ = inp['f3_v']
                al = F.softmax(sc_.masked_fill(~v_, -1e9), dim=1) * v_.any(1)[:, None]
                x = x + s.Gl[l][None] * F.embedding_bag(inp['f3_val'], s.E.weight, per_sample_weights=al, mode='sum', sparse=s.E.sparse)
            q = s.rms(x); q1, q2 = q[:, :h], q[:, h:]
            s1 = q1 @ s.K1[l].t() / math.sqrt(h); s2_ = q2 @ s.K2[l].t() / math.sqrt(h)
            v1, i1 = s1.topk(kk, dim=1); v2, i2 = s2_.topk(kk, dim=1)
            if BAL and torch.is_grad_enabled():                                          # 鍵の使われ方を ならす (選ばれた割合 f と 平均確率 P の積)
                for sc_, ii_ in ((s1, i1), (s2_, i2)):
                    f_ = torch.zeros(n, device=sc_.device).scatter_add_(0, ii_.reshape(-1), torch.ones(ii_.numel(), device=sc_.device)) / ii_.numel()
                    s._aux = s._aux + n * (f_.detach() * F.softmax(sc_.float(), dim=1).mean(0)).sum()
            comb = (v1[:, :, None] + v2[:, None, :]).flatten(1)
            sc, ix = comb.topk(k, dim=1)
            a = F.softmax(sc / s.tau[l].abs().clamp_min(0.05), dim=1)
            ia = i1.gather(1, ix // kk); ib = i2.gather(1, ix % kk); rows = ia * n + ib; s._rows[l] = rows.flatten()
            x = x + (a[:, :, None] * s._val(l, rows)).sum(1)
        return s.rms(x)

    def cache(s, inp):
        if 'f3_win' in inp:                                                         # 窓の語を 1 回だけ引いて、 ずらして使う (c1, c2 を別々に引くのと同じ値)
            ew = s.Ek(inp['f3_win']); vv = inp['f3_v'][..., None]; e0 = s.Ek.weight[0]
            kq = KQF(ew, vv, e0, s.Pk[0], s.Pk[1])                                  # 掛け算・選択・足し算・正規化を 1 つの計算にまとめる (compile=1 のとき)
        else:
            kq = F.normalize(s.Pk[0] * s.Ek(inp['f3_c1']) + s.Pk[1] * s.Ek(inp['f3_c2']), dim=-1)           # (B,MC,DK)
        q = F.normalize(s.Pk[0] * s.Ek(inp['ctx'][:, 0]) + s.Pk[1] * s.Ek(inp['ctx'][:, 1]), dim=-1)
        if BF16: kq = kq.to(torch.bfloat16); q = q.to(torch.bfloat16)               # bf16=1: 大きな鍵のテンソルだけ 16 ビットで持つ (メモリ読み書きが半分)
        s._kq = kq
        sc = s.theta * (kq @ q[:, :, None])[:, :, 0].float()
        v = inp['f3_v']; any_ = v.any(1)
        sc = sc.masked_fill(~v, -1e9); a = F.softmax(sc, dim=1) * any_[:, None]
        mx = torch.minimum(sc.max(1).values.clamp_min(-1.0), s.theta.detach().abs() + 1.0)
        g = torch.sigmoid(s.ga * mx + s.gb) * any_
        return a, g

    def ctxv(s, inp, a):
        if not CV: return None
        return F.embedding_bag(inp['f3_val'], s.E.weight, per_sample_weights=a, mode='sum', sparse=s.E.sparse)    # (B,d): 点 a で 重みづけした 文脈の語の意味ベクトル

    def _base(s, inp, cv=None):
        x = (s.E(inp['ctx']) * s.P[None]).sum(1)
        if HNG: x = x + s._hn(inp)
        if cv is not None: x = x + s.Gc[None] * cv
        if 1 in s.feats:
            ids1 = inp['f1_ids']; m1 = inp['f1_m']; w1 = m1 / m1.sum(2, keepdim=True).clamp_min(1.0)
            mean = F.embedding_bag(ids1.reshape(-1, ids1.shape[2]), s.E.weight, per_sample_weights=w1.reshape(-1, ids1.shape[2]), mode='sum', sparse=s.E.sparse).view(ids1.shape[0], ids1.shape[1], -1)
            x = x + (mean * s.G1[None]).sum(1)
        if 2 in s.feats:
            r = s.rar[inp['f2_ids']].masked_fill(~inp['f2_v'], -1.0)
            tv, ti = r.topk(K2, dim=1); w = tv.clamp_min(0.0)
            ids = inp['f2_ids'].gather(1, ti)
            s2 = F.embedding_bag(ids, s.E.weight, per_sample_weights=w / (w.sum(1, keepdim=True) + 1e-6), mode='sum', sparse=s.E.sparse)
            x = x + s.G2[None] * s2
        return x

    def _layer(s, l, x):
        n, kk, k, h = s.n, s.kk, s.k, s.d // 2
        q = s.rms(x); q1, q2 = q[:, :h], q[:, h:]
        s1 = q1 @ s.K1[l].t() / math.sqrt(h); s2_ = q2 @ s.K2[l].t() / math.sqrt(h)
        v1, i1 = s1.topk(kk, dim=1); v2, i2 = s2_.topk(kk, dim=1)
        if BAL and torch.is_grad_enabled():                                          # 鍵の使われ方を ならす (選ばれた割合 f と 平均確率 P の積)
            for sc_, ii_ in ((s1, i1), (s2_, i2)):
                f_ = torch.zeros(n, device=sc_.device).scatter_add_(0, ii_.reshape(-1), torch.ones(ii_.numel(), device=sc_.device)) / ii_.numel()
                s._aux = s._aux + n * (f_.detach() * F.softmax(sc_.float(), dim=1).mean(0)).sum()
        comb = (v1[:, :, None] + v2[:, None, :]).flatten(1)
        sc, ix = comb.topk(k, dim=1)
        a = F.softmax(sc / s.tau[l].abs().clamp_min(0.05), dim=1)
        ia = i1.gather(1, ix // kk); ib = i2.gather(1, ix % kk); rows = ia * n + ib; s._rows[l] = rows.flatten()
        return x + (a[:, :, None] * s._val(l, rows)).sum(1)

    def logp_band_h(s, inp, X, y, ds_all, M, H):
        """ctxh: 連続した列で、 履歴 H 位置 + 予測 M 位置の中間表現を作り、 CHL 層の後で 予測位置が 直前 H 位置の中間表現へ Attention する。
        inp: 履歴+予測の全位置 (nb*(H+M)) の入力。 X: 長さ (H+M)+MC+2 の語の列。 ds_all: (nb,H+M) 各位置の文書の開始 (X の中の番号)。 y: 予測位置の語 (nb*M)"""
        nb = X.shape[0]; Ls = H + M; dev = X.device
        a_c, g_c, cv, Tw = s.band(X, ds_all, Ls)
        x = s._base(inp, None if cv is None else cv.reshape(nb * Ls, -1))
        for l in range(CHL + 1): x = s._layer(l, x)
        xs = x.view(nb, Ls, -1); r = s.rms(xs)
        K = F.normalize(r @ s.Wkh, dim=-1); q = F.normalize(r[:, H:] @ s.Wqh, dim=-1)
        Kw = K.unfold(1, H, 1)[:, :M]                                                  # (nb,M,DK,H): 位置 m の直前 H 位置の鍵
        sc = s.thh * torch.matmul(q[:, :, None, :], Kw)[:, :, 0].float()               # (nb,M,H)
        v = (torch.arange(M, device=dev)[:, None] + torch.arange(H, device=dev)[None])[None] >= ds_all[:, H:, None]
        sc = sc.masked_fill(~v, -1e9); at = F.softmax(sc, dim=2) * v.any(2)[..., None]
        A = torch.zeros(nb, M, Ls, dtype=at.dtype, device=dev); rows = torch.arange(M, device=dev)[:, None]; A[:, rows, rows + torch.arange(H, device=dev)[None]] = at
        xq = xs[:, H:] + s.Gh[None, None] * torch.bmm(A, r)
        xq = xq.reshape(nb * M, -1)
        for l in range(CHL + 1, s.L): xq = s._layer(l, xq)
        lp = F.log_softmax(F.linear(s.rms(xq), s.U), dim=1).gather(1, y[:, None])[:, 0]
        pc = (a_c[:, H:] * (Tw[:, H:] == y.view(nb, M)[..., None]).float()).sum(2).reshape(-1); g = g_c[:, H:].reshape(-1)
        return torch.log((1 - g) * lp.exp() + g * pc + 1e-12)

    def band(s, X, ds_loc, M):
        """連続した M 個の位置の 文脈キャッシュ (鍵・点・値の足し合わせ) を 列ごとに 1 回で作る (位置ごとに 窓を引き直さない)。
        X: (nb, Ls) 長さ Ls = M+MC+2 の語の列 (位置 MC+2+m が m 番目の予測する位置)。 ds_loc: (nb, M) 各位置の文書の開始 (X の中の番号)。 ctxl には未対応"""
        nb, Ls = X.shape; dev = X.device; Ek = s.Ek(X)
        KQ = F.normalize(s.Pk[0] * Ek[:, 1:-1] + s.Pk[1] * Ek[:, :-2], dim=-1)                    # (nb,Ls-2,DK): 鍵 (直前 2 語から)
        Kw = KQ.unfold(1, MC, 1)[:, :M]                                                          # (nb,M,DK,MC): 位置 m の窓の鍵 (引き直さず ずらして使う)
        pi = MC + 2 + torch.arange(M, device=dev)
        q = F.normalize(s.Pk[0] * Ek[:, pi - 1] + s.Pk[1] * Ek[:, pi - 2], dim=-1)               # (nb,M,DK)
        sc = s.theta * torch.matmul(q[:, :, None, :], Kw)[:, :, 0].float()                       # (nb,M,MC)
        v = (torch.arange(M, device=dev)[:, None] + torch.arange(MC, device=dev)[None])[None] >= ds_loc[:, :, None]
        sc = sc.masked_fill(~v, -1e9); any_ = v.any(2); a = F.softmax(sc, dim=2) * any_[..., None]
        mx = torch.minimum(sc.max(2).values.clamp_min(-1.0), s.theta.detach().abs() + 1.0); g = torch.sigmoid(s.ga * mx + s.gb) * any_
        A = torch.zeros(nb, M, Ls, dtype=a.dtype, device=dev); rows = torch.arange(M, device=dev)[:, None]; A[:, rows, rows + 2 + torch.arange(MC, device=dev)[None]] = a
        cv = torch.bmm(A, s.E(X)) if CV else None                                                # (nb,M,d): 点で重みづけした語の意味ベクトル (行列積 1 回)
        return a, g, cv, X.unfold(1, MC, 1)[:, 2:2 + M]

    def logp_band(s, inp, X, y, ds_loc, M):
        a, g, cv, Tw = s.band(X, ds_loc, M); nb = X.shape[0]
        lp = F.log_softmax(F.linear(s.hidden(inp, None if cv is None else cv.reshape(nb * M, -1)), s.U), dim=1).gather(1, y[:, None])[:, 0]
        pc = (a * (Tw == y.view(nb, M)[..., None]).float()).sum(2).reshape(-1)
        return torch.log((1 - g.reshape(-1)) * lp.exp() + g.reshape(-1) * pc + 1e-12)

    def logp(s, inp, y):
        if 3 in s.feats:
            a, g = s.cache(inp); cv = s.ctxv(inp, a)
        else: cv = None
        lp = F.log_softmax(F.linear(s.hidden(inp, cv), s.U), dim=1).gather(1, y[:, None])[:, 0]
        if 3 not in s.feats: return lp
        pc = (a * (inp['f3_val'] == y[:, None]).float()).sum(1)
        return torch.log((1 - g) * lp.exp() + g * pc + 1e-12)

    def dist(s, inp):
        if 3 in s.feats:
            a, g = s.cache(inp); cv = s.ctxv(inp, a)
        else: cv = None
        pm = F.softmax(F.linear(s.hidden(inp, cv), s.U), dim=1)
        if 3 not in s.feats: return pm
        pcd = torch.zeros_like(pm).scatter_add_(1, inp['f3_val'], a)
        return (1 - g)[:, None] * pm + g[:, None] * pcd


class LazyAdam:
    """SparseAdam と同じ計算 (勾配が 0 でない行だけ更新) を 密な勾配で GPU 上で行う。"""
    def __init__(s, params, lr, betas=(0.9, 0.999), eps=1e-8, rowsrc=None, sinks=None):
        s.sinks = sinks or {}        # {id(重み): sink}  あれば p.grad の代わりに sink の触った行を読み、 読んだ行を 0 に戻す
        s.rowsrc = rowsrc or {}    # {id(重み): その歩に使った行の番号を返す関数}  (あれば その行だけ更新: 同じ番号が重なっても同じ値を書くだけ)
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
                G = s.sinks[id(p)] if id(p) in s.sinks else p.grad
                if G is None: continue
                if id(p) not in s.st: s.st[id(p)] = (torch.zeros_like(p), torch.zeros_like(p))
                m, v = s.st[id(p)]
                if id(p) in s.rowsrc:
                    idx = s.rowsrc[id(p)](); Gs = G[idx]; mn = s.b1 * m[idx] + (1 - s.b1) * Gs; vn = s.b2 * v[idx] + (1 - s.b2) * Gs * Gs
                    upd = (g['lr'] * bc2.sqrt() / bc1) * mn / (vn.sqrt() + s.eps)
                    m[idx] = mn; v[idx] = vn; p[idx] = p[idx] - upd
                    if id(p) in s.sinks: G.index_fill_(0, idx, 0.0)       # 読んだ行を 0 に戻す (CUDA グラフに記録できる形)
                    continue
                mask = (G != 0).any(1, keepdim=True)
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
        inp['f3_win'] = T[(pos[:, None] - torch.arange(1, MC + 3, device=T.device)[None]).clamp_min(0)]
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
        inp['f3_win'] = torch.from_numpy(T[np.maximum(pos[:, None] - np.arange(1, MC + 3)[None], 0)])
    return {k_: v_.to(DEV) for k_, v_ in inp.items()}


def main(cfg):
    global MC, W2, NS, W1, DEV, SPARSE, K2
    DEV = torch.device(cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    t0 = time.time(); lg = lambda m: print('[%6ds] %s' % (time.time() - t0, m), flush=True)
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(cfg.get('seed', 0)); rng = np.random.default_rng(cfg.get('seed', 0))
    MC = cfg.get('mc', MC); W2 = cfg.get('w2', W2); NS = cfg.get('ns', NS); K2 = cfg.get('k2', K2)
    global KQF, BF16, CV, CL, CHL, VFAC, VFULL, HNG, BAL
    BAL = float(cfg.get('bal', 0.0))
    HNG = int(cfg.get('hng', 0))
    VFAC = int(cfg.get('vfac', 0)); VFULL = bool(int(cfg.get('vfull', 1)))
    CV = bool(int(cfg.get('ctxv', 0))); CL = int(cfg.get('ctxl', 0)); CHL = int(cfg.get('ctxh', -1)); HIST = int(cfg.get('hist', 512))
    BF16 = bool(int(cfg.get('bf16', 0)))
    if int(cfg.get('tf32', 0)): torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision('high')   # tf32=1: 行列積を TF32 で (出力層など)
    if int(cfg.get('compile', 0)): KQF = torch.compile(_kq_eager)
    W = '/tmp/tok'; d = cfg.get('d', 128); c = cfg.get('c', 8); L = cfg['L']; n = cfg.get('n', 256); kk = cfg.get('kk', 4); k = cfg.get('k', 4); B = cfg.get('B', 512)
    SEQB = int(cfg.get('seqb', 0)); NBLK = int(cfg.get('nblk', 1))        # seqb=M: 連続した M 位置を 1 つの列として 文脈キャッシュを列ごとに 1 回で作る (nblk 本で 1 歩)
    if SEQB: B = SEQB * NBLK
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
    if SEQB:
        starts = np.arange(c + (HIST + MC + 2 if CHL >= 0 else 0), N - SEQB, SEQB); rng.shuffle(starts); pos = (starts[:, None] + np.arange(SEQB)[None]).reshape(-1); steps = len(pos) // B
    fast = (DEV.type == 'cuda' and int(cfg.get('fast', 1)) == 1) or int(cfg.get('fast', 1)) == 2; SPARSE = not fast   # fast=2: CPU で速い経路の試験
    net = Net(d, c, L, n, kk, k, feats, rar).to(DEV); npar = sum(p.numel() for p in net.parameters())
    lg('L=%d feats=%s B=%d 学習語 %d 歩数 %d パラメータ %.1fM 窓: 部品2=%d 部品3=%d' % (L, feats, B, len(pos), steps, npar / 1e6, W2, MC))
    use_graph = bool(fast and DEV.type == 'cuda' and int(cfg.get('graph', 1)))
    rowsrc = {id(v_.weight): (lambda l_=l_: net._rows[l_]) for l_, v_ in enumerate(net.Val)} if int(cfg.get('rowwise', 1)) else {}
    if HNG and rowsrc is not None and int(cfg.get('rowwise', 1)): rowsrc[id(net.H.weight)] = (lambda: net._hrows)
    sinks = {}
    if int(cfg.get('vsink', 0)) and fast and rowsrc:         # Val の密な勾配 (毎歩 全体の 0 埋め・足し算) をやめ、 触った行だけ足し込む
        Net.VSINK = True; net.Vsink = [torch.zeros_like(v_.weight) for v_ in net.Val]; sinks = {id(v_.weight): net.Vsink[l_] for l_, v_ in enumerate(net.Val)}
        for v_ in net.Val: v_.weight.requires_grad_(True)
    if use_graph:
        opt_s = LazyAdam(net.sparse_params(), lr=torch.tensor(lrs, device=DEV), rowsrc=rowsrc, sinks=sinks); opt_d = torch.optim.Adam(net.dense_params(), lr=torch.tensor(lr, device=DEV), capturable=True)
    else:
        opt_s = LazyAdam(net.sparse_params(), lr=lrs, rowsrc=rowsrc, sinks=sinks) if fast else torch.optim.SparseAdam(net.sparse_params(), lr=lrs); opt_d = torch.optim.Adam(net.dense_params(), lr=lr)
    warm = cfg.get('warm', 200)
    if fast:
        S_t = torch.from_numpy(S).to(DEV); DS_t = torch.from_numpy(DS).to(DEV); pos_t = torch.from_numpy(pos).to(DEV)
        offs8_t = torch.from_numpy(offs8).to(DEV); o1_t = [torch.from_numpy(o).to(DEV) for o in o1]; offs2_t = torch.from_numpy(offs2).to(DEV)
        lg('GPU の速い経路: 入力の準備を GPU で行い、 疎な更新は LazyAdam')
        feats_n3 = [f_ for f_ in feats if f_ != 3]
        def loss_pos(pp_t):                                  # 1 歩の損失 (seqb なら 列ごとの一括計算)
            y = S_t[pp_t]
            if SEQB and CHL >= 0:                           # ctxh: 履歴 HIST 位置 + 予測 SEQB 位置の中間表現どうしで Attention
                pb = pp_t.view(NBLK, SEQB); p0 = pb[:, 0]; Lq = HIST + SEQB; ar_ = torch.arange(Lq, device=DEV)[None]
                pa = (p0[:, None] - HIST + ar_).reshape(-1); Xe = S_t[(p0[:, None] - HIST - MC - 2 + torch.arange(Lq + MC + 2, device=DEV)[None]).clamp_min(0)]
                dsa = DS_t[pa].view(NBLK, Lq) - (p0 - HIST - MC - 2)[:, None]
                inp = make_inputs_t(S_t, pa, DS_t[pa], feats_n3, offs8_t, o1_t, offs2_t)
                lp = net.logp_band_h(inp, Xe, y, dsa, SEQB, HIST); m_ = (y != 0).float(); return -(lp * m_).sum() / m_.sum().clamp_min(1.0)
            if SEQB:
                pb = pp_t.view(NBLK, SEQB); p0 = pb[:, 0]; Ls = SEQB + MC + 2
                Xs = S_t[(p0[:, None] - (MC + 2) + torch.arange(Ls, device=DEV)[None]).clamp_min(0)]; dsl = DS_t[pb] - (p0 - (MC + 2))[:, None]
                inp = make_inputs_t(S_t, pp_t, DS_t[pp_t], feats_n3, offs8_t, o1_t, offs2_t)
                lp = net.logp_band(inp, Xs, y, dsl, SEQB); m_ = (y != 0).float(); return -(lp * m_).sum() / m_.sum().clamp_min(1.0)
            return -net.logp(make_inputs_t(S_t, pp_t, DS_t[pp_t], feats, offs8_t, o1_t, offs2_t), y).mean()
        if BAL:
            _lp0 = loss_pos
            def loss_pos(pp_t):
                net._aux = 0.0; l_ = _lp0(pp_t); return l_ + BAL * net._aux

    def nn_logp_blocks(T, D, starts, bs=4):                  # ctxh: 連続した列 (starts から SEQB 位置) ごとに 履歴つきで 損失を出す
        out = np.zeros((len(starts), SEQB)); net.eval(); Tt = torch.from_numpy(T.astype(np.int64)).to(DEV); Dt = torch.from_numpy(np.asarray(D).astype(np.int64)).to(DEV); Lq = HIST + SEQB
        with torch.no_grad():
            for a in range(0, len(starts), bs):
                st_ = torch.from_numpy(starts[a:a + bs].astype(np.int64)).to(DEV); nb_ = len(st_)
                pa = (st_[:, None] - HIST + torch.arange(Lq, device=DEV)[None]).reshape(-1); dsa = Dt[(torch.searchsorted(Dt, pa, right=True) - 1).clamp_min(0)]
                Xe = Tt[(st_[:, None] - HIST - MC - 2 + torch.arange(Lq + MC + 2, device=DEV)[None]).clamp_min(0)]
                inp = make_inputs_t(Tt, pa, dsa, [f_ for f_ in feats if f_ != 3], offs8_t, o1_t, offs2_t); y = Tt[(st_[:, None] + torch.arange(SEQB, device=DEV)[None]).reshape(-1)]
                lp = net.logp_band_h(inp, Xe, y, dsa.view(nb_, Lq) - (st_ - HIST - MC - 2)[:, None], SEQB, HIST)
                out[a:a + nb_] = (lp / math.log(2)).view(nb_, SEQB).cpu().numpy()
        net.train(); return out

    def nn_logp(T, p, ds, bs=1024, starts=None, D=None):
        if CHL >= 0 and starts is not None:
            full = nn_logp_blocks(T, D, starts).reshape(-1); Pall = (starts[:, None] + np.arange(SEQB)[None]).reshape(-1)
            u, iu = np.unique(Pall, return_index=True); return full[iu[np.searchsorted(u, p)]]
        out = np.zeros(len(p)); net.eval()
        with torch.no_grad():
            for a in range(0, len(p), bs):
                pp = p[a:a + bs]; inp = make_inputs(T, pp, ds[a:a + bs], feats, offs8, o1, offs2); y = torch.from_numpy(T[pp]).to(DEV)
                out[a:a + bs] = (net.logp(inp, y) / math.log(2)).cpu().numpy()
        net.train(); return out

    def load_split(sp_):
        T = np.fromfile('%s/%s.bin' % (W, sp_), dtype=np.uint16).astype(np.int64); D = np.load('%s/%s_docs.npy' % (W, sp_)); return T, D
    def ds_of(D, p): return D[np.maximum(np.searchsorted(D, p, 'right') - 1, 0)]
    rng_ev = np.random.default_rng(int(cfg.get('eval_seed', 777)))      # 評価の位置は 学習の乱数と切り離して固定 (構成を変えても 同じ位置で測る)
    def evalset_h(T, D, nn_, nlong):                        # ctxh: 評価の位置は 連続した列 (SEQB 位置) をいくつか
        nbk = -(-int(nn_ * 1.02) // SEQB); lo = c + HIST + MC + 2 + 16
        starts = np.sort(rng_ev.integers(lo, len(T) - SEQB, nbk)); Pall = (starts[:, None] + np.arange(SEQB)[None]).reshape(-1)
        p = Pall[T[Pall] != 0][:nn_]; q = Pall[(T[Pall] != 0) & ((Pall - ds_of(D, Pall)) >= 256)][:nlong]; return np.r_[p, q], len(p), starts
    def evalset(T, D, nn_, nlong):
        p = rng_ev.integers(16 + c, len(T), int(nn_ * 1.02)); p = p[T[p] != 0][:nn_]
        q = rng_ev.integers(16 + c, len(T), 4000000); q = q[(T[q] != 0) & ((q - ds_of(D, q)) >= 256)][:nlong]; return np.r_[p, q], len(p)
    ev = {}
    for sp_, nn_ in (('val', cfg['nval']), ('test', cfg['ntest'])):
        T, D = load_split(sp_); starts_ = None
        if CHL >= 0 or int(cfg.get('eval_blocks', 0)): p, nmain, starts_ = evalset_h(T, D, nn_, cfg.get('nlong', 40000))      # eval_blocks=1: ctxh なしでも 同じ (連続した列の) 位置で測る
        else: p, nmain = evalset(T, D, nn_, cfg.get('nlong', 40000))
        ev[sp_] = dict(T=T, D=D, p=p, nmain=nmain, ds=ds_of(D, p), starts=starts_)
    offs = np.arange(1, c + 1); curve = []; run = 0.0; cnt_ = 0; ts = time.time()
    if use_graph:
        pos_buf = pos_t[:B].clone(); run_t = torch.zeros((), device=DEV); cg = None; side = torch.cuda.Stream()
        def g_step():
            loss = loss_pos(pos_buf); loss.backward(); opt_s.step(); opt_d.step(); run_t.add_(loss.detach() / math.log(2)); return loss
    if cfg.get('prof') and fast and DEV.type == 'cuda':       # 時間の内訳を測る (学習はしない): 入力の準備 / 順方向 / 逆方向 / 疎な更新 / 密な更新 を CUDA イベントで、 さらに 重い計算の上位を torch.profiler で
        ev_ = lambda: torch.cuda.Event(enable_timing=True)
        def one_step(st_, tm):
            pp_t = pos_t[st_ * B:(st_ + 1) * B]; e = [ev_() for _ in range(6)]; e[0].record(); e[1].record()
            loss = loss_pos(pp_t); e[2].record()
            opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); e[3].record()
            opt_s.step(); e[4].record(); opt_d.step(); e[5].record()
            if tm is not None: tm.append(e)
        for st_ in range(10): one_step(st_, None)
        torch.cuda.synchronize(); tm = []
        for st_ in range(10, 40): one_step(st_, tm)
        torch.cuda.synchronize()
        names = ['入力の準備', '順方向', '逆方向', '疎な更新(E, Val, Ek)', '密な更新']
        tot_ = [np.mean([e[i].elapsed_time(e[i + 1]) for e in tm]) for i in range(5)]
        lg('PROF_PHASE ' + json.dumps({n_: round(float(t_), 3) for n_, t_ in zip(names, tot_)}, ensure_ascii=False) + ' 合計 %.3f ミリ秒/歩 (= %.1f 歩/秒, 通常の経路)' % (sum(tot_), 1000 / sum(tot_)))
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as pr_:
            for st_ in range(40, 50): one_step(st_, None)
            torch.cuda.synchronize()
        sd_ = lambda r: getattr(r, 'self_device_time_total', None) or getattr(r, 'self_cuda_time_total', 0.0)
        rows_ = sorted(pr_.key_averages(), key=lambda r: -sd_(r))[:14]; tt_ = max(sum(sd_(r) for r in pr_.key_averages()), 1.0)
        for r in rows_: lg('PROF_KERNEL %5.1f%% %8.2f ミリ秒/歩 x%d %s' % (100 * sd_(r) / tt_, sd_(r) / 1000 / 10, r.count // 10, r.key[:90]))
        lg('PROF_TOTAL GPU の実計算 %.2f ミリ秒/歩' % (tt_ / 1000 / 10)); return
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
                        l_e = float(loss_pos(pos_buf))
                    cg.replay(); l_g = float(loss_g)
                    lg('CUDA グラフの自己確認: 通常 %.5f | 再生 %.5f (差 %.1e)' % (l_e, l_g, abs(l_e - l_g)))
                    if abs(l_e - l_g) > 2e-3: raise RuntimeError('CUDA グラフの再生が通常の経路と合いません')
                else:
                    cg.replay()
            cnt_ += 1
        else:
            for g in opt_d.param_groups: g['lr'] = lr * f
            for g in opt_s.param_groups: g['lr'] = lrs * f
            if fast: loss = loss_pos(pos_t[st * B:(st + 1) * B])
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
        parts[sp_] = dict(ng=2.0 ** lp, hi=hi, nn=2.0 ** nn_logp(T, p, e_['ds'], starts=e_.get('starts'), D=e_['D']), uni=pu[T[p]], depth=p - e_['ds'], nmain=e_['nmain'])
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
    def save_up(tag):                                    # 重みを保存して HF の非公開データセットへ置く (失敗しても 学習は続ける)
        try:
            p_ = '/content/%s.pt' % tag if os.path.isdir('/content') else '/tmp/%s.pt' % tag
            torch.save(dict(cfg=cfg, state={k_: v_.detach().cpu() for k_, v_ in net.state_dict().items()}), p_); lg('保存: %s (%.0f MB)' % (p_, os.path.getsize(p_) / 1e6))
            tok_ = os.environ.get('HF_TOKEN_UP')
            if tok_:
                from huggingface_hub import HfApi
                HfApi().upload_file(path_or_fileobj=p_, path_in_repo='models/%s.pt' % tag, repo_id=cfg.get('save_repo', 'Youichi1/ai-lab-data'), repo_type='dataset', token=tok_); lg('HFへ保存: models/%s.pt' % tag)
            else: lg('HF_TOKEN_UP がないので HF への保存は省略')
        except Exception as e_:
            lg('重みの保存に失敗: %r' % (e_,))
    if cfg.get('save'): save_up(cfg['save'] + '_base')
    if cfg.get('dlg'):                                   # ---- 会話データで 調整して、 質問に答えさせてみる (prep_dialog.py で 作ったデータを使う) ----
        dd = cfg.get('dlg_dir', '/tmp/dlg_out')
        Tq = np.fromfile(dd + '/dlg_train.bin', dtype=np.uint16).astype(np.int64); Dq = np.load(dd + '/dlg_train_docs.npy'); test_items = json.load(open(dd + '/dlg_test.json'))
        S2 = np.concatenate([np.zeros(c, np.int64), Tq]); dsq = Dq[np.maximum(np.searchsorted(Dq, np.arange(len(Tq)), 'right') - 1, 0)]
        DS2 = np.concatenate([np.arange(c), c + dsq]); pos2 = np.arange(c, len(S2)); pos2 = pos2[S2[pos2] != 0]
        lg('DLG 調整データ: %d 語, %d 例, 検証 %d 問' % (len(Tq), len(Dq), len(test_items)))
        def ans_loss():
            net.eval(); tot = {}
            with torch.no_grad():
                for it in test_items:
                    p_ids = tk.encode(it['prompt']).ids; a_ids = tk.encode(it['ref'] + '\n\n').ids[:200]
                    T_ = np.array([0] * c + p_ids + a_ids, dtype=np.int64); ps = np.arange(c + len(p_ids), len(T_))
                    inp = make_inputs(T_, ps, np.full(len(ps), c), feats, offs8, o1, offs2); y = torch.from_numpy(T_[ps]).to(DEV)
                    v_ = float((-net.logp(inp, y) / math.log(2)).sum()); s_ = tot.setdefault(it['src'], [0.0, 0]); s_[0] += v_; s_[1] += len(ps)
            net.train(); return {k_: round(v_[0] / v_[1], 4) for k_, v_ in tot.items()}
        def gen(prompt, nmax=120, temp=0.7, topk_=40):
            T_ = np.array([0] * c + tk.encode(prompt).ids, dtype=np.int64); n0 = len(T_); net.eval()
            with torch.no_grad():
                for i_ in range(nmax):
                    inp = make_inputs(T_, np.array([len(T_)]), np.array([c]), feats, offs8, o1, offs2); pr = net.dist(inp)[0] ** (1 / temp); pr[0] = 0; tv, ti = pr.topk(topk_)
                    pr = torch.zeros_like(pr).scatter_(0, ti, tv); pr = pr / pr.sum(); T_ = np.r_[T_, int(torch.multinomial(pr, 1))]
                    if i_ % 8 == 7 and '###' in tk.decode(T_[n0:].tolist()): break
            net.train(); return tk.decode(T_[n0:].tolist()).split('###')[0].strip().replace('\n', ' / ')
        pick = {}
        for it in test_items:
            if len(pick.setdefault(it['src'], [])) < cfg.get('dlg_ngen', 3): pick[it['src']].append(it)
        show = lambda tag: [lg('GEN_CHAT[%s][%s] 質問: %s || 正解: %s || 網: %s' % (tag, k_, it['prompt'].replace('### 質問\n', '').replace('\n### 回答\n', '')[:150].replace('\n', ' '), it['ref'][:120].replace('\n', ' '), gen(it['prompt']))) for k_, v_ in pick.items() for it in v_]
        l0 = ans_loss(); lg('DLG 検証の答えの損失 (調整前, bit/語): %s' % json.dumps(l0, ensure_ascii=False)); show('調整前')
        B2 = cfg.get('dlg_B', 256); ep = cfg.get('dlg_epochs', 2); flr = cfg.get('dlg_lr', 0.4); spe = len(pos2) // B2; steps2 = min(ep * spe, cfg.get('dlg_max', 10 ** 9))
        order = np.concatenate([rng.permutation(len(pos2)) for _ in range(ep)]); pos_all = pos2[order]
        S2_t = torch.from_numpy(S2).to(DEV); DS2_t = torch.from_numpy(DS2).to(DEV); pos_all_t = torch.from_numpy(pos_all).to(DEV)
        ts2 = time.time(); run2 = 0.0; cn2 = 0
        for st in range(steps2):
            f = min(1.0, (st + 1) / 100) * max(0.05, 1 - st / steps2) * flr
            for g in opt_d.param_groups: g['lr'].fill_(lr * f) if torch.is_tensor(g['lr']) else g.__setitem__('lr', lr * f)
            for g in opt_s.param_groups: g['lr'].fill_(lrs * f) if torch.is_tensor(g['lr']) else g.__setitem__('lr', lrs * f)
            if fast:
                pp_t = pos_all_t[st * B2:(st + 1) * B2]; inp = make_inputs_t(S2_t, pp_t, DS2_t[pp_t], feats, offs8_t, o1_t, offs2_t); y = S2_t[pp_t]
            else:
                pp = pos_all[st * B2:(st + 1) * B2]; inp = make_inputs(S2, pp, DS2[pp], feats, offs8, o1, offs2); y = torch.from_numpy(S2[pp]).to(DEV)
            loss = -net.logp(inp, y).mean(); run2 += float(loss) / math.log(2); cn2 += 1
            opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); opt_s.step(); opt_d.step()
            if (st + 1) % cfg.get('dlg_log', 1000) == 0:
                lg('DLG 歩 %d/%d: 損失 %.4f bit/語 | %.1f 歩/秒' % (st + 1, steps2, run2 / cn2, cfg.get('dlg_log', 1000) / (time.time() - ts2))); run2 = 0.0; cn2 = 0; ts2 = time.time()
        l1 = ans_loss(); lg('DLG 検証の答えの損失 (調整後, bit/語): %s' % json.dumps(l1, ensure_ascii=False)); show('調整後')
        lg('RESULT_DLG ' + json.dumps(dict(before=l0, after=l1), ensure_ascii=False))
        if cfg.get('save'): save_up(cfg['save'] + '_dlg')


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
