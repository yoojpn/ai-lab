import subprocess, os, sys, time, json, glob
os.environ['HF_TOKEN'] = 'HF_TOKEN_HERE'
def sh(c, show=True):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); o = (r.stdout + r.stderr)
    if show: print(o[-2500:], flush=True)
    return o
sh('nproc; free -g | head -2')
sh('pip -q install tokenizers huggingface_hub pyarrow 2>&1 | tail -1')
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ['HF_TOKEN']); R = 'Youichi1/ai-lab-data'; W = '/content/ev'; os.makedirs(W, exist_ok=True)
TK = os.environ['HF_TOKEN']
for f in ('fmeng.cpp', 'fm_chat.py', 'chat2.py', 'chat3.py', 'align_copy.py', 'engine.py'): hf_hub_download(R, 'fm/code/' + f, repo_type='dataset', token=TK, local_dir=W + '/code')
for f in ('E.npy', 'cls64.npy', 'combo_lam.npy', 'uni.npy'): hf_hub_download(R, 'fm/chat2/' + f, repo_type='dataset', token=TK, local_dir=W + '/a')
for f in ('dlgW.tar', 'al.pkl', 'typ.tar', 'icl_tasks.json'): hf_hub_download(R, 'fm/chat3/' + f, repo_type='dataset', token=TK, local_dir=W + '/a')
for f in ('qness.npy', 'qa.json'): hf_hub_download(R, 'fm/stem/' + f, repo_type='dataset', token=TK, local_dir=W + '/a')
hf_hub_download(R, 'tok/tok.json', repo_type='dataset', token=TK, local_dir=W + '/tokhf'); os.environ['TOKJSON'] = W + '/tokhf/tok/tok.json'
CD = W + '/code/fm/code'
print(sh('cd ' + CD + ' && g++ -O2 -march=native -fopenmp fmeng.cpp -o fmeng 2>&1 | tail -3'), flush=True)
files = api.list_repo_files(R, repo_type='dataset'); shards = sorted(f for f in files if f.startswith('fm/shard_') and f.endswith('.bwt.xz')); lst = []
for s in shards:
    m = s.replace('.bwt.xz', '.meta.json')
    if m not in files: continue
    a = hf_hub_download(R, s, repo_type='dataset', token=TK, local_dir=W + '/sh'); b = hf_hub_download(R, m, repo_type='dataset', token=TK, local_dir=W + '/sh'); lst.append(a + ' ' + b)
open(W + '/list.txt', 'w').write('\n'.join(lst) + '\n'); print('shards', len(lst), flush=True)
os.environ['FMENG'] = CD + '/fmeng'; sys.path.insert(0, CD)
import numpy as np, fm_chat
from chat2 import Chat2
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.environ['TOKJSON'])
lamj = os.path.join(W, 'lam.json')

import pickle, tarfile
sh('mkdir -p /content/dlgW /content/typ && tar -xf /content/ev/a/fm/chat3/dlgW.tar -C /content/dlgW && tar -xf /content/ev/a/fm/chat3/typ.tar -C /content/typ')
import align_copy, chat2
from chat3 import Chat3
fm = fm_chat.FM(W + '/list.txt'); print('index ready', flush=True)
A = W + '/a/fm/chat2/'; hf_hub_download(R, 'fm/chat3/al_jaen.pkl', repo_type='dataset', token=TK, local_dir=W + '/a'); al = pickle.load(open(W + '/a/fm/chat3/al_jaen.pkl', 'rb')); align_copy.D = len(al.S)
dd = chat2.local_dist('/content/dlgW', None)
ch = Chat3(fm.dist, np.load(A + 'E.npy'), np.load(A + 'uni.npy'), np.load(A + 'combo_lam.npy')[:9], np.load(A + 'cls64.npy'), tk, al, dd)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.append(s)

import re
CENT = np.load('/content/typ/cent.npy'); KT = len(CENT); dts = [chat2.local_dist('/content/typ/z%d' % k, None) for k in range(KT)]
qn = np.load(W + '/a/fm/stem/qness.npy'); QA = json.load(open(W + '/a/fm/stem/qa.json')); STOP = {tk.encode('。').ids[0], 2}
HAS = np.array([bool(re.search(r'\w', tk.decode([i]))) for i in range(32768)])
def uvec(seg):
    seg = np.asarray(seg, np.int64); v = (ch.Em[seg] * ch.idf[seg][:, None]).sum(0); n = np.linalg.norm(v); return v / n if n > 0 else v
def feats4(ctx, hist_end, p, q0, q1):
    f = np.zeros((4, 32768), np.float32); f[:3] = ch.feats(ctx, hist_end, p)
    z = int(np.argmax(CENT @ uvec(ctx[q0:q1]))) if q1 > q0 else 0
    f[3] = np.clip(np.log(np.maximum(dts[z](list(ctx)), 1e-12)) - np.log(np.maximum(dd(list(ctx)), 1e-12)), -3, 3); return f
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.append(s)
def save(tag):
    open(W + '/final2_out.txt', 'w').write('\n'.join(out)); api.upload_file(path_or_fileobj=W + '/final2_out.txt', path_in_repo='results/final2/out_%s.txt' % tag, repo_id=R, repo_type='dataset')
# --- 重み: 索引外の 会話の bit + 名前の 記憶 を 合わせた 目的で 8 重み (頭 8 語 / 以降 × 4 特徴)

import json as _j
TT = _j.load(open('/content/ev/a/fm/chat3/icl_tasks.json'))
def gen6(ids, aw, dw):
    g = []
    for _ in range(6):
        p = np.asarray(fm.dist(ids + g) if aw == 0 else ch.dist3(ids + g, len(ids), w=[dw, aw, 0.0], late=True), np.float64); p[0] = 0; g.append(int(np.argmax(p)))
    return tk.decode(g)
res = {}
for t in TT:
    ids = [0] + tk.encode(t['prompt']).ids; a = t['answer'].strip()
    for aw in (0, 1.0, 2.0, 3.0, 5.0):
        r = res.setdefault((t['task'], aw), [0, 0]); r[0] += gen6(ids, aw, 0.2).strip().startswith(a); r[1] += 1
for k, v in sorted(res.items()): P('ICL', k[0], 'align w', k[1], '%d/%d' % tuple(v))
save('icl2'); P('ICL2_DONE')
