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
for f in ('fmeng.cpp', 'fm_chat.py'): hf_hub_download(R, 'fm/code/' + f, repo_type='dataset', token=TK, local_dir=W + '/code')
for f in ('E.npy', 'cls64.npy', 'combo_lam.npy', 'uni.npy'): hf_hub_download(R, 'fm/chat2/' + f, repo_type='dataset', token=TK, local_dir=W + '/a')
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
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.environ['TOKJSON'])
lamj = os.path.join(W, 'lam.json')

fm = fm_chat.FM(W + '/list.txt'); print('index ready', flush=True)
def top(ctx, k=6):
    p = fm.dist(ctx); i = np.argsort(-p)[:k]; return ' '.join('%s(%.2f)' % (tk.decode([int(x)]), p[x]) for x in i)
def greedy(prefix, n=24, seed=0):
    rng = np.random.default_rng(seed); c = [0] + tk.encode(prefix).ids; g = []
    for _ in range(n):
        p = fm.dist(c + g); p[0] = 0; i = np.argsort(-p)[:5]; q = p[i] ** 2; q /= q.sum(); x = int(rng.choice(i, p=q))
        if x == 2: break
        g.append(x)
    return tk.decode(g)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.append(s)
probes = [('日本の首都は', '日本の首都はどこですか？'), ('水の化学式は', '水の化学式は何ですか？'), ('The capital of France is', 'What is the capital of France?'), ('光の速さは', '光の速さはどれくらいですか？'),
          ('夏目漱石の代表作は', '夏目漱石の代表作を教えてください。'), ('Pythonでリストを逆順にするには、', 'Pythonでリストを逆順にするには？'), ('The largest planet in the solar system is', 'Which is the largest planet in the solar system?'), ('富士山の高さは', '富士山の高さは何メートルですか？')]
for decl, q in probes:
    P('宣言文:', decl, '| 次の語:', top([0] + tk.encode(decl).ids)); P('   続き:', decl, greedy(decl).replace('\n', ' ')[:160])
    qc = [0] + tk.encode(q).ids + [2]; P('質問文:', q, '| 次の語:', top(qc)); P('   続き:', greedy(q + '\n').replace('\n', ' ')[:160])
open(W + '/probe_out.txt', 'w').write('\n'.join(out)); api.upload_file(path_or_fileobj=W + '/probe_out.txt', path_in_repo='results/probe/out.txt', repo_id=R, repo_type='dataset')
print('PROBE_DONE', flush=True)
