"""長い文脈: Grave らの連続キャッシュ (過去の隠れ状態を鍵、 その次の語を値にして 内積で引く) を 保存済みの網に 推論時に足して 効くかを測る。
  p = (1-λ) p_網 + λ p_キャッシュ、  p_キャッシュ(y) = Σ_{i: w_i=y} exp(θ h_p·h_i) / Σ_i exp(θ h_p·h_i)   (窓 W 語、 文書内)
使い方: python3 longcache.py <重み.pt> <val.bin> <val_docs.npy> <文書数> <1文書の位置数> <出力.npz>"""
import sys, time, numpy as np, torch
sys.path.insert(0, '/home/claude/ai-lab/hdc')
import chat_net as C, exp_ctx as X
net, cfg = C.load(sys.argv[1]); c = cfg.get('c', 8); feats = list(cfg['feats']); T = np.fromfile(sys.argv[2], dtype=np.uint16).astype(np.int64); D = np.load(sys.argv[3])
ND, NP = int(sys.argv[4]), int(sys.argv[5]); lens = np.diff(np.r_[D, len(T)]); cand = np.where(lens >= NP + 8)[0]; rng = np.random.default_rng(11); docs = rng.choice(cand, min(ND, len(cand)), replace=False)
offs8 = np.arange(1, c + 1); o1 = [1 + (np.arange(X.NS) * w) // X.NS for w in X.W1]; offs2 = np.arange(c + 1, X.W2 + 1)
out_h, out_w, out_lp, out_d = [], [], [], []; t0 = time.time()
with torch.no_grad():
    for di, k in enumerate(docs):
        s = D[k]; P = s + np.arange(NP); H, LP = [], []
        for i in range(0, NP, 100):
            pp = P[i:i + 100]; inp = X.make_inputs(T, pp, np.full(len(pp), s), feats, offs8, o1, offs2); y = torch.from_numpy(T[pp])
            a, g = net.cache(inp); cv = net.ctxv(inp, a); h = net.hidden(inp, cv); ls = torch.log_softmax(torch.nn.functional.linear(h, net._U()), 1); lp = ls.gather(1, y[:, None])[:, 0]
            pc = (a * (inp['f3_val'] == y[:, None]).float()).sum(1); lp = torch.log((1 - g) * lp.exp() + g * pc + 1e-12)
            H.append(h.numpy().astype(np.float16)); LP.append(lp.numpy())
        out_h.append(np.concatenate(H)); out_lp.append(np.concatenate(LP)); out_w.append(T[P]); out_d.append(np.full(NP, di)); print('文書 %d/%d  %.0fs  平均 %.3f bit' % (di + 1, len(docs), time.time() - t0, -np.concatenate(LP).mean() / np.log(2)), flush=True)
np.savez(sys.argv[6], h=np.stack(out_h), w=np.stack(out_w), lp=np.stack(out_lp))
