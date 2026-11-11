import subprocess, os, time, json, sys
import numpy as np
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); return (r.stdout + r.stderr)[-1500:]
sh('pip -q install tokenizers huggingface_hub sentencepiece protobuf')
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ['HF_TOKEN']); R = 'Youichi1/ai-lab-data'; W = '/content/mp'; TK = os.environ['HF_TOKEN']; os.makedirs(W, exist_ok=True)
LOG = []
def lg(m):
    LOG.append('[%s] %s' % (time.strftime('%H:%M:%S'), m)); print(LOG[-1], flush=True); open(W + '/diff_log.txt', 'w').write('\n'.join(LOG))
    try: api.upload_file(path_or_fileobj=W + '/diff_log.txt', path_in_repo='results/map/diff_log.txt', repo_id=R, repo_type='dataset')
    except Exception as e: pass
def up(local, remote):
    for i in range(6):
        try: api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=R, repo_type='dataset'); return
        except Exception as e: time.sleep(60)

os.makedirs(W + '/tok', exist_ok=True)
hf_hub_download(R, 'tok/tok.json', repo_type='dataset', token=TK, local_dir=W); mp = hf_hub_download(R, 'results/map/map.npz', repo_type='dataset', token=TK, local_dir=W)
sh('pip -q install transformers torch')
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8); OT = Tokenizer.from_file(W + '/tok.json'); Z = np.load(mp)
DOMS = sorted({k.rsplit('_', 1)[0] for k in Z.files if k.endswith('_tok')})
MODELS = {'gpt2': ['en_edu', 'en_wiki', 'code'], 'HuggingFaceTB/SmolLM2-135M': ['en_edu', 'en_wiki', 'code'], 'Qwen/Qwen2.5-0.5B': ['ja_web', 'ja_wiki', 'en_edu', 'en_wiki', 'code']}
OUT = {}
def spans(ids):
    """我々の語 i が 文字列の どこから どこまでか (文字数)。 前から順に decode した長さの差"""
    ends = []; 
    for i in range(len(ids)): ends.append(len(OT.decode([int(x) for x in ids[:i + 1]])))
    return np.array(ends)
for mn, doms in MODELS.items():
    try:
        tk = AutoTokenizer.from_pretrained(mn, use_fast=True); md = AutoModelForCausalLM.from_pretrained(mn).eval()
        mx = min(getattr(md.config, 'n_positions', None) or getattr(md.config, 'max_position_embeddings', 1024) or 1024, 1024); bos = tk.bos_token_id if tk.bos_token_id is not None else tk.eos_token_id
        for dn in doms:
            t0 = time.time(); tok = Z[dn + '_tok']; doc = Z[dn + '_doc']; pos = Z[dn + '_pos']; arr = np.full(len(tok), np.nan, np.float32)
            for d in sorted(set(doc.tolist())):
                if d % 2: continue
                ix = np.flatnonzero(doc == d); ix = ix[np.argsort(pos[ix])]; ids = tok[ix]; text = OT.decode([int(x) for x in ids]); ends = spans(ids)
                enc = tk(text, add_special_tokens=False, return_offsets_mapping=True); lid = [bos] + enc['input_ids']; off = np.array(enc['offset_mapping'])
                bits = np.zeros(len(lid) - 1); half = mx // 2
                for i in range(1, len(lid), half):
                    lo = max(0, i - half); x = torch.tensor([lid[lo:i + half]])
                    with torch.no_grad(): lp = torch.log_softmax(md(x).logits[0].float(), -1)
                    ll = lp[:-1].gather(1, x[0, 1:][:, None])[:, 0]; bb = (-ll / np.log(2)).numpy(); bits[i - 1:lo + len(bb)] = bb[i - 1 - lo:]
                # LLM の語 k の bit を その文字範囲に 一様に ばらまいて、 我々の語の範囲で 足す
                cum = np.zeros(len(text) + 1); a, b = off[:, 0], off[:, 1]
                for k in range(len(bits)):
                    if b[k] > a[k]: cum[a[k] + 1:b[k] + 1] += bits[k] / (b[k] - a[k])
                cum = np.cumsum(cum); st = np.r_[0, ends[:-1]]; arr[ix] = (cum[np.minimum(ends, len(text))] - cum[np.minimum(st, len(text))]).astype(np.float32)
            OUT['%s__%s' % (dn, mn.split('/')[-1])] = arr; lg('%s %s: %.0fs, 語あたり %.3f bit' % (mn, dn, time.time() - t0, np.nanmean(arr)))
            np.savez_compressed(W + '/llm_tok.npz', **OUT); up(W + '/llm_tok.npz', 'results/map/llm_tok.npz')
        del md
    except Exception as e:
        import traceback; lg('失敗 %s: %s' % (mn, traceback.format_exc()[-500:]))
lg('DIFF_DONE'); print('DIFF_DONE')
