"""ローカルの val の 評価位置 (cache_poe.py 等と同じ idx_te) で GPT-2 の 語ごとの bit を出す。 文脈 = 文書の頭から 最大 500 語。
我々の語 i の文字範囲に 重なる GPT-2 の語の bit を (重なりの割合で) 足す。 出力: <out>.npy (位置ごとの bit, 割り当てできなければ nan)
使い方: python3 gpt2_local.py <W> <out.npy> [n=4000]"""
import sys, os, numpy as np, torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
W, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 4000; torch.set_num_threads(1)
val = np.fromfile(W + '/val.bin', dtype=np.uint16); pad = np.concatenate([np.zeros(64, np.uint16), val]); docs = np.load(W + '/val_docs.npy') + 64
idx = np.random.default_rng(777).integers(64, len(val) - 1, N) + 64
OT = Tokenizer.from_file(W + '/tok.json'); tk = AutoTokenizer.from_pretrained('gpt2'); md = AutoModelForCausalLM.from_pretrained('gpt2').eval(); bos = tk.bos_token_id
res = np.full(N, np.nan, np.float32)
for n, i in enumerate(idx):
    ds = docs[np.searchsorted(docs, i, side='right') - 1]; lo = max(ds, i - 500, 8)
    ctx = [int(x) for x in pad[lo:i]]; full = [int(x) for x in pad[lo:i + 1]]
    if len(ctx) < 1: continue
    pre = OT.decode(ctx); txt = OT.decode(full)
    if not txt.startswith(pre) or len(txt) == len(pre): continue
    a, b = len(pre), len(txt); enc = tk(txt, add_special_tokens=False, return_offsets_mapping=True); ids = [bos] + enc['input_ids']; off = enc['offset_mapping']
    if len(ids) > 1000: continue
    with torch.no_grad(): lp = torch.log_softmax(md(torch.tensor([ids])).logits[0].float(), -1)
    bits = (-lp[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0] / np.log(2)).numpy(); tot = 0.0
    for k, (s, e) in enumerate(off):
        ov = max(0, min(e, b) - max(s, a))
        if ov > 0 and e > s: tot += bits[k] * ov / (e - s)
    res[n] = tot
    if n % 200 == 0: print(n, np.nanmean(res[:n + 1]), flush=True); np.save(OUT, res)
np.save(OUT, res); print('done', np.nanmean(res), np.isnan(res).mean())
