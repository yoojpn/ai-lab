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
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.environ['TOKJSON'])
lamj = os.path.join(W, 'lam.json')

fm = fm_chat.FM(W + '/list.txt'); print('index ready', flush=True)
qn = np.load(W + '/a/fm/stem/qness.npy'); QA = json.load(open(W + '/a/fm/stem/qa.json')); STOP = {tk.encode('。').ids[0], 2}
def stem(ids, th, mode):
    ids = list(ids)
    if mode == 'tail':
        while len(ids) > 2 and qn[ids[-1]] > th: ids.pop()
        return ids
    k = [x for x in ids if qn[x] <= th]; return k if len(k) >= 2 else ids
def gen(ctx, n=30):
    g = []
    for _ in range(n):
        p = fm.dist(ctx + g); p[0] = 0
        if len(g) < 1: p[2] = 0
        for i in range(len(g) - 2):
            if g[i:i + 2] == g[-2:]: p[g[i + 2]] = 0
        x = int(np.argmax(p)); g.append(x)
        if x in STOP: break
    return tk.decode(g)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.append(s)
MODES = [('質問そのまま', None, None), ('末尾を剥がす 1.0', 1.0, 'tail'), ('末尾を剥がす 2.0', 2.0, 'tail'), ('全部の質問語を除く 2.0', 2.0, 'all'), ('全部の質問語を除く 3.0', 3.0, 'all')]
for lang in ('ja', 'en'):
    res = {m[0]: 0 for m in MODES}; n = len(QA[lang])
    for q, ans in QA[lang]:
        ids = tk.encode(q).ids
        for nm, th, md in MODES:
            ctx = [0] + ids + [2] if md is None else [0] + stem(ids, th, md)
            g = gen(ctx); hit = any(a in g for a in ans); res[nm] += hit
            if lang == 'ja' and nm != MODES[0][0] and len(out) < 400: P('  [%s] %s -> %s %s' % (nm, q, g.replace('\n', ' ')[:60], '○' if hit else '×'))
            elif nm == MODES[0][0] and lang == 'ja' and len(out) < 400: P('  [質問そのまま] %s -> %s %s' % (q, g.replace('\n', ' ')[:60], '○' if hit else '×'))
    for nm, _, _ in MODES: P('正答を含む割合', lang, nm, '%d/%d = %.0f%%' % (res[nm], n, 100 * res[nm] / n))
open(W + '/stem_out.txt', 'w').write('\n'.join(out)); api.upload_file(path_or_fileobj=W + '/stem_out.txt', path_in_repo='results/stem/out.txt', repo_id=R, repo_type='dataset')
print('STEM_DONE', flush=True)
