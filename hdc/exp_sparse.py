"""FFNの活性化の疎さ: 学習済み(通常のReLU FFN)で、各トークンで発火するニューロンの割合と、上位20%のニューロンが占める出力エネルギー。
 発火割合が小さいほど、疎なカーネルがあれば FFN の計算を減らせる。使い方: python3 exp_sparse.py '{"N":16000000,"epochs":1}'"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import exp_y
from lab import log
cfg = json.loads(sys.argv[1]); net, Xte, yt = exp_y.run(cfg)
net.eval(); stats = []
acts = {}
def mk(i):
    def hook(m, inp, out): acts[i] = torch.relu(out).detach()
    return hook
hs = [L.linear1.register_forward_hook(mk(i)) for i, L in enumerate(net.layers)]
with torch.no_grad(): net.hidden(Xte[:512])
for i, a in acts.items():
    a = a.reshape(-1, a.shape[-1]); frac = (a > 0).float().mean().item()
    srt = a.pow(2).sort(1, descending=True).values; k = int(0.2 * a.shape[1]); top = (srt[:, :k].sum(1) / (srt.sum(1) + 1e-9)).mean().item()
    top5 = (srt[:, :int(0.05 * a.shape[1])].sum(1) / (srt.sum(1) + 1e-9)).mean().item()
    stats.append(dict(layer=i, active_frac=frac, energy_top20pct=top, energy_top5pct=top5)); log('層%d 発火割合 %.3f 上位20%%ニューロンのエネルギー %.3f 上位5%% %.3f' % (i, frac, top, top5))
log('RESULT-SPARSE ' + json.dumps(stats))
