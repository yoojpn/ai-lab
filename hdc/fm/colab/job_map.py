import subprocess, os, time, json, sys
import numpy as np
os.environ.setdefault('HF_TOKEN', '')   # 事前に 環境に 設定
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); return (r.stdout + r.stderr)[-1500:]
sh('pip -q install tokenizers huggingface_hub sentencepiece protobuf')
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ['HF_TOKEN']); R = 'Youichi1/ai-lab-data'; W = '/content/mp'; TK = os.environ['HF_TOKEN']; os.makedirs(W, exist_ok=True)
LOG = []
def lg(m):
    LOG.append('[%s] %s' % (time.strftime('%H:%M:%S'), m)); print(LOG[-1], flush=True); open(W + '/map_log.txt', 'w').write('\n'.join(LOG))
    try: api.upload_file(path_or_fileobj=W + '/map_log.txt', path_in_repo='results/map/map_log.txt', repo_id=R, repo_type='dataset')
    except Exception as e: pass
def up(local, remote):
    for i in range(6):
        try: api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=R, repo_type='dataset'); return
        except Exception as e: time.sleep(60)
DOMS = ['ja_web', 'ja_wiki', 'en_edu', 'en_wiki', 'code']; NDOC = 120; DLEN = 600; ORD = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
hf_hub_download(R, 'fm/code/fmeng.cpp', repo_type='dataset', token=TK, local_dir=W); hf_hub_download(R, 'tok/tok.json', repo_type='dataset', token=TK, local_dir=W)
print(sh('cd %s/fm/code && g++ -O2 -march=native -fopenmp fmeng.cpp -o fmeng 2>&1 | tail -3' % W))
# ---- 文書を選ぶ ----
D = {}
rng = np.random.default_rng(0)
for dn in DOMS:
    v = np.fromfile(hf_hub_download(R, 'fm/val/val_%s.bin' % dn, repo_type='dataset', token=TK, local_dir=W), dtype=np.uint16).astype(np.int32)
    z = np.flatnonzero(v == 3); st = np.r_[0, z + 1]; en = np.r_[z, len(v)]; ok = np.flatnonzero(en - st >= 200); rng.shuffle(ok); ok = np.sort(ok[:NDOC])
    docs = [v[st[i]:min(en[i], st[i] + DLEN)] for i in ok]; D[dn] = docs; lg('%s: 文書 %d, 語 %d' % (dn, len(docs), sum(len(d) for d in docs)))
def longest(d):
    n = len(d); cand = np.zeros(n, np.int32); ml = np.zeros(n, np.int32)
    for t in range(2, n):
        c = np.flatnonzero(d[:t - 1] == d[t - 1])
        if len(c) == 0: continue
        m = np.ones(len(c), np.int32); alive = np.ones(len(c), bool)
        for k in range(1, 24):
            ok = alive & (c - k >= 0) & (d[np.maximum(c - k, 0)] == d[max(t - 1 - k, 0)]) & (t - 1 - k >= 0); m[ok] += 1; alive = ok
            if not alive.any(): break
        j = len(c) - 1 - np.argmax(m[::-1]); cand[t] = d[c[j] + 1]; ml[t] = m[j]
    return cand, ml
Q = []; META = {}
for dn in DOMS:
    toks = []; dix = []; pos = []; cands = []; mls = []
    for i, d in enumerate(D[dn]):
        c, m = longest(d); toks.append(d); dix.append(np.full(len(d), i, np.int32)); pos.append(np.arange(len(d), dtype=np.int32)); cands.append(c); mls.append(m)
        for t in range(len(d)):
            row = np.full(24, -1, np.int32); cx = np.r_[0, d[:t]][-23:]; row[23 - len(cx):23] = cx; row[23] = d[t]; Q.append(row)
            row2 = row.copy(); row2[23] = c[t]; Q.append(row2)
    META[dn] = dict(tok=np.concatenate(toks), doc=np.concatenate(dix), pos=np.concatenate(pos), cand=np.concatenate(cands), mlen=np.concatenate(mls))
Q = np.stack(Q); Q.tofile(W + '/q.bin'); lg('問 %d' % len(Q))
# ---- 索引 ----
files = api.list_repo_files(R, repo_type='dataset'); lst = []
for s in sorted(f for f in files if f.startswith('fm/shard_') and f.endswith('.bwt.xz')):
    a = hf_hub_download(R, s, repo_type='dataset', token=TK, local_dir=W + '/sh'); b = hf_hub_download(R, s.replace('.bwt.xz', '.meta.json'), repo_type='dataset', token=TK, local_dir=W + '/sh'); lst.append(a + ' ' + b)
open(W + '/list.txt', 'w').write('\n'.join(lst) + '\n'); lg('分割 %d を取得' % len(lst))
t = time.time(); r = subprocess.run([W + '/fm/code/fmeng', W + '/list.txt', ','.join(map(str, ORD)), 'batch', W + '/q.bin', W + '/o.bin', '8'], capture_output=True, text=True); lg('fmeng 終了 %.0fs rc=%d %s' % (time.time() - t, r.returncode, r.stderr[-300:]))
out = np.fromfile(W + '/o.bin', dtype=np.int64).reshape(len(Q), len(ORD), 3); k = 0; sv = {}
for dn in DOMS:
    n = len(META[dn]['tok']); o = out[k:k + 2 * n]; k += 2 * n
    for key, val in META[dn].items(): sv[dn + '_' + key] = val
    sv[dn + '_base'] = o[0::2]; sv[dn + '_cq'] = o[1::2]
np.savez_compressed(W + '/map.npz', **sv); up(W + '/map.npz', 'results/map/map.npz'); lg('map.npz 保存')
subprocess.run('rm -rf %s/sh' % W, shell=True)
# ---- 物差し: 小型の公開 LLM を 同じ文書で ----
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8); OT = Tokenizer.from_file(W + '/tok/tok.json'); NL = 60; RES = {}
MODELS = {'gpt2': ['en_edu', 'en_wiki', 'code'], 'HuggingFaceTB/SmolLM2-135M': ['en_edu', 'en_wiki', 'code'], 'HuggingFaceTB/SmolLM2-360M': ['en_edu', 'en_wiki', 'code'], 'rinna/japanese-gpt2-small': ['ja_web', 'ja_wiki'], 'Qwen/Qwen2.5-0.5B': DOMS}
TXT = {dn: [OT.decode([int(x) for x in d]) for d in D[dn][0::2]] for dn in DOMS}
json.dump({dn: [len(t.encode('utf-8')) for t in TXT[dn]] for dn in DOMS}, open(W + '/bytes.json', 'w')); up(W + '/bytes.json', 'results/map/bytes.json')
for mn, doms in MODELS.items():
    try:
        tk = AutoTokenizer.from_pretrained(mn, use_fast=('rinna' not in mn)); md = AutoModelForCausalLM.from_pretrained(mn).eval(); npar = sum(p.numel() for p in md.parameters())
        mx = min(getattr(md.config, 'n_positions', None) or getattr(md.config, 'max_position_embeddings', 1024) or 1024, 1024); bos = tk.bos_token_id if tk.bos_token_id is not None else tk.eos_token_id
        for dn in doms:
            t0 = time.time(); per = []
            for tx in TXT[dn]:
                ids = [bos] + tk(tx, add_special_tokens=False)['input_ids']; bits = 0.0; half = mx // 2
                for i in range(1, len(ids), half):
                    lo = max(0, i - half); x = torch.tensor([ids[lo:i + half]])
                    with torch.no_grad(): lp = torch.log_softmax(md(x).logits[0].float(), -1)
                    tg = x[0, 1:]; ll = lp[:-1].gather(1, tg[:, None])[:, 0]; bits += float(-ll[i - lo - 1:].sum()) / np.log(2)
                per.append(bits)
            nb = sum(len(t.encode('utf-8')) for t in TXT[dn]); RES.setdefault(mn, {'params': npar})[dn] = {'bpb': sum(per) / nb, 'per_doc_bits': per}
            lg('LLM %s (%.0fM) %s: %.4f bit/byte (%.0fs)' % (mn, npar / 1e6, dn, sum(per) / nb, time.time() - t0))
            json.dump(RES, open(W + '/llm.json', 'w')); up(W + '/llm.json', 'results/map/llm.json')
        del md
    except Exception as e:
        import traceback; lg('LLM 失敗 %s: %s' % (mn, traceback.format_exc()[-400:]))
lg('MAP_DONE'); print('MAP_DONE')
