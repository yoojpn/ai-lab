"""対照: 普通の小さな Transformer (GPT 型) を 同じ語の列・同じ語の切り方・同じ評価位置で学習して 単独の損失を測る。
学習の語: exp_ctx と同じ (train.bin から 40 個の等間隔ブロック, 合計 ntr 語, 1 エポック)。 評価の位置: exp_ctx の eval_blocks=1 と同じ (val.bin, 乱数 777, 512 位置ずつ 40 ブロック, 先頭 20000 位置)。
使い方: python3 tfm_ref.py '{"d":384,"L":6,"ntr":6000000}'   (環境変数 TRAIN, VAL)"""
import os, sys, json, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

TRAIN = os.environ.get('TRAIN', '/tmp/tok/train.bin'); VAL = os.environ.get('VAL', '/tmp/tok/val.bin')
V = 32768; C = 8; CTX = 512


class Block(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.n1 = nn.RMSNorm(d); s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False); s.n2 = nn.RMSNorm(d); s.f1 = nn.Linear(d, 4 * d, bias=False); s.f2 = nn.Linear(4 * d, d, bias=False)

    def forward(s, x):
        B, T, D = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.nh, D // s.nh).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + s.f2(F.gelu(s.f1(s.n2(x))))


class GPT(nn.Module):
    def __init__(s, d, L, nh):
        super().__init__(); s.E = nn.Embedding(V, d); s.P = nn.Embedding(CTX, d); s.bl = nn.ModuleList([Block(d, nh) for _ in range(L)]); s.nf = nn.RMSNorm(d)
        nn.init.normal_(s.E.weight, 0, 0.02); nn.init.normal_(s.P.weight, 0, 0.02)
        for b in s.bl: nn.init.normal_(b.o.weight, 0, 0.02 / math.sqrt(2 * L)); nn.init.normal_(b.f2.weight, 0, 0.02 / math.sqrt(2 * L))

    def forward(s, t, last=False):
        x = s.E(t) + s.P(torch.arange(t.shape[1], device=t.device))[None]
        for b in s.bl: x = b(x)
        return F.linear(s.nf(x[:, -1] if last else x), s.E.weight)                                  # 入力と出力の埋め込みは共有


def main(cfg):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); d = cfg['d']; L = cfg['L']; ntr = cfg['ntr']; bs = cfg.get('bs', 16); lr = cfg.get('lr', 1e-3); nh = cfg.get('nh', max(d // 64, 1)); torch.manual_seed(cfg.get('seed', 0))
    Ttr = np.memmap(TRAIN, dtype=np.uint16, mode='r'); nb = 40; bl = ntr // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
    S = np.concatenate([np.zeros(C, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]).astype(np.int64)
    nseq = (len(S) - 1) // CTX; X = torch.from_numpy(S[:nseq * CTX].reshape(nseq, CTX)); Y = torch.from_numpy(S[1:nseq * CTX + 1].reshape(nseq, CTX))
    perm = torch.randperm(nseq, generator=torch.Generator().manual_seed(1)); steps = nseq // bs
    m = GPT(d, L, nh).to(dev); npar = sum(p.numel() for p in m.parameters()); opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    print('Transformer d=%d L=%d 頭=%d 変数 %.1fM  学習語 %d  歩数 %d (1 歩 %d 列 x %d 語)' % (d, L, nh, npar / 1e6, len(S), steps, bs, CTX), flush=True)
    t0 = time.time(); run = 0.0
    for st in range(steps):
        f = min((st + 1) / 100, 0.5 * (1 + math.cos(math.pi * st / steps)) * 0.9 + 0.1)
        for g in opt.param_groups: g['lr'] = lr * f
        ix = perm[st * bs:(st + 1) * bs]; x = X[ix].to(dev); y = Y[ix].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'): lg = m(x)
        loss = F.cross_entropy(lg.float().view(-1, V), y.view(-1), ignore_index=0); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); run += loss.item() / math.log(2)
        if (st + 1) % max(steps // 10, 1) == 0: print('  歩 %d/%d 学習損失 %.3f bit  %.0f秒' % (st + 1, steps, run / max(steps // 10, 1), time.time() - t0), flush=True); run = 0.0
    # 評価: exp_ctx (eval_blocks=1, hist=512, mc=512, seqb=512, c=8) と同じ位置
    Tv = np.fromfile(VAL, dtype=np.uint16).astype(np.int64); lo = C + 512 + 512 + 2 + 16; nvb = -(-int(20000 * 1.02) // 512)
    sv = np.sort(np.random.default_rng(777).integers(lo, len(Tv) - 512, nvb)); Pall = (sv[:, None] + np.arange(512)[None]).reshape(-1); p = Pall[Tv[Pall] != 0][:20000]
    m.eval(); tot = 0.0; Tvt = torch.from_numpy(Tv)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        for i in range(0, len(p), 128):
            pp = p[i:i + 128]; w = torch.stack([Tvt[q - CTX + 1:q + 1] for q in pp]).to(dev)          # 各位置 q を 最後の語にして 直前 511 語を文脈に
            lg = m(w[:, :-1], last=True).float(); y = torch.from_numpy(Tv[pp]).to(dev); tot += (-F.log_softmax(lg, -1).gather(1, y[:, None])[:, 0]).sum().item() / math.log(2)
    print('TFM_RESULT d=%d L=%d ntr=%d 変数 %.1fM: 単独 %.4f bit/token (評価 %d 位置, %.0f秒)' % (d, L, ntr, npar / 1e6, tot / len(p), len(p), time.time() - t0), flush=True)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
