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
for f in ('dlgW.tar', 'al.pkl', 'typ.tar'): hf_hub_download(R, 'fm/chat3/' + f, repo_type='dataset', token=TK, local_dir=W + '/a')
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
A = W + '/a/fm/chat2/'
try:
    hf_hub_download(R, 'fm/chat3/al_jaen.pkl', repo_type='dataset', token=TK, local_dir=W + '/a'); al = pickle.load(open(W + '/a/fm/chat3/al_jaen.pkl', 'rb')); print('al_jaen')
except Exception as e:
    al = pickle.load(open(W + '/a/fm/chat3/al.pkl', 'rb')); print('al ja only')
align_copy.D = len(al.S)
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
    open(W + '/final_out.txt', 'w').write('\n'.join(out)); api.upload_file(path_or_fileobj=W + '/final_out.txt', path_in_repo='results/final/out_%s.txt' % tag, repo_id=R, repo_type='dataset')

def stem(ids, th=1.0):
    ids = list(ids)
    while len(ids) > 2 and qn[ids[-1]] > th: ids.pop()
    return ids
def gen(ctx, n=36, beta=2.0):
    g = []
    for _ in range(n):
        p = fm.dist(ctx + g); p[0] = 0
        if beta: p = p * np.exp(-beta * np.maximum(qn, 0))
        if len(g) < 1: p[2] = 0
        for i in range(len(g) - 2):
            if g[i:i + 2] == g[-2:]: p[g[i + 2]] = 0
        x = int(np.argmax(p)); g.append(x)
        if x in STOP: break
    return tk.decode(g)
ch.w = [0.2, 2.0, 0.1, 0.2, 2.0, 0.0]
def respond(utts):
    last = tk.encode(utts[-1]).ids; isq = float(np.mean([qn[x] for x in last[-3:]])) > 1.0
    res = {'question_like': bool(isq)}
    res['general'] = ch.reply3(utts, plain=False, seed=1, maxlen=48)
    if isq: res['stem'] = gen([0] + stem(last))
    return res
import time as _t
done = set(); t_end = _t.time() + float(os.environ.get('SERVE_SEC', 3300)); print('SERVING', flush=True)
while _t.time() < t_end:
    try:
        p = hf_hub_download(R, 'talk/in.json', repo_type='dataset', token=TK, local_dir=W + '/talk', force_download=True); req = json.load(open(p))
    except Exception as e:
        _t.sleep(15); continue
    if req['id'] in done: _t.sleep(10); continue
    try:
        if req.get('mode') == 'raw':
            res = {'raw': []}
            for pr in req['prompts']:
                ids = [0] + tk.encode(pr).ids; o = {}
                for nm, fn in (('plain', lambda c: fm.dist(c)), ('feat', lambda c: ch.dist3(c, len(ids), w=[0.5, 2.0, 0.0], late=True))):
                    g = []
                    for _ in range(int(req.get('n', 8))):
                        p = np.asarray(fn(ids + g), np.float64); p[0] = 0; x = int(np.argmax(p)); g.append(x)
                    o[nm] = tk.decode(g)
                res['raw'].append(o)
        else: res = respond(req['utts'])
    except Exception as e: res = {'error': str(e)[:300]}
    res['id'] = req['id']; res['utts'] = req.get('utts'); done.add(req['id'])
    open(W + '/talk/out.json', 'w').write(json.dumps(res, ensure_ascii=False)); api.upload_file(path_or_fileobj=W + '/talk/out.json', path_in_repo='talk/out_%s.json' % req['id'], repo_id=R, repo_type='dataset')
print('TALK_DONE', flush=True)
