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
import re
HAS = np.array([bool(re.search(r'\w', tk.decode([i]))) for i in range(32768)])
rng = np.random.default_rng(0)
def gen(ctx, n=30, beta=2.0, nopunct=False, temp=0.0, topk=8):
    g = []
    for _ in range(n):
        p = fm.dist(ctx + g); p[0] = 0
        if beta: p = p * np.exp(-beta * np.maximum(qn, 0))
        if len(g) < 1: p[2] = 0
        if nopunct and len(g) < 1: p = p * HAS
        for i in range(len(g) - 2):
            if g[i:i + 2] == g[-2:]: p[g[i + 2]] = 0
        if temp == 0: x = int(np.argmax(p))
        else:
            i = np.argsort(-p)[:topk]; q = p[i] ** (1 / temp); q /= q.sum(); x = int(rng.choice(i, p=q))
        g.append(x)
        if x in STOP: break
    return g
def key(g):
    for x in g:
        if HAS[x]: return x
    return -1
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.append(s)
def save(tag):
    open(W + '/ans_out.txt', 'w').write('\n'.join(out)); api.upload_file(path_or_fileobj=W + '/ans_out.txt', path_in_repo='results/ans/out_%s.txt' % tag, repo_id=R, repo_type='dataset')
NS = 6
for lang in ('ja',):
    n = len(QA[lang]); h = {'A 貪欲(基準)': 0, 'B 句読点を禁止': 0, 'C 標本の多数決': 0, 'D 多数決(語の重みつき)': 0, '標本のどれか': 0, '質問そのまま': 0}
    for q, ans in QA[lang]:
        ids = tk.encode(q).ids; st = [0] + stem(ids, 1.0, 'tail'); hit = lambda g: any(a in tk.decode(g) for a in ans)
        gA = gen(st, 24, 2.0); gB = gen(st, 24, 2.0, True); h['A 貪欲(基準)'] += hit(gA); h['B 句読点を禁止'] += hit(gB)
        S = [gen(st, 24, 2.0, True, 0.8, 8) for _ in range(NS)] + [gB]; ks = [key(g) for g in S]; cnt = {}
        for g, k in zip(S, ks): cnt[k] = cnt.get(k, 0) + 1
        win = max(cnt, key=cnt.get); gC = next(g for g, k in zip(S, ks) if k == win); h['C 標本の多数決'] += hit(gC)
        wt = {}
        for g, k in zip(S, ks): wt[k] = wt.get(k, 0) + (1.0 if qn[k] < 1 and k >= 0 else 0.3)
        win2 = max(wt, key=wt.get); gD = next(g for g, k in zip(S, ks) if k == win2); h['D 多数決(語の重みつき)'] += hit(gD); h['標本のどれか'] += any(hit(g) for g in S)
        if len(out) % 10 == 0: save('part')
        if lang == 'ja': P('  %s | A: %s | C: %s %s' % (q, tk.decode(gA).replace('\n', ' ')[:30], tk.decode(gC).replace('\n', ' ')[:30], '○' if hit(gC) else '×'))
    for k_, v in h.items(): P('正答を含む割合', lang, k_, '%d/%d = %.0f%%' % (v, n, 100 * v / n))
    save(lang)
print('ANS_DONE', flush=True)
