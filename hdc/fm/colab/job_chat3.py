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
for f in ('dlgW.tar', 'al.pkl', 'te_clean.pkl'): hf_hub_download(R, 'fm/chat3/' + f, repo_type='dataset', token=TK, local_dir=W + '/a')
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
sh('mkdir -p /content/dlgW && tar -xf /content/ev/a/fm/chat3/dlgW.tar -C /content/dlgW')
import align_copy, chat2
from chat3 import Chat3
fm = fm_chat.FM(W + '/list.txt'); print('index ready', flush=True)
A = W + '/a/fm/chat2/'; al = pickle.load(open(W + '/a/fm/chat3/al.pkl', 'rb')); align_copy.D = len(al.S)
dd = chat2.local_dist('/content/dlgW', None)
ch = Chat3(fm.dist, np.load(A + 'E.npy'), np.load(A + 'uni.npy'), np.load(A + 'combo_lam.npy')[:9], np.load(A + 'cls64.npy'), tk, al, dd)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.append(s)
# 1. 重みを 100億語の索引で 数え直す (リークのない 評価会話)
te = pickle.load(open(W + '/a/fm/chat3/te_clean.pkl', 'rb')); rng = np.random.default_rng(0); pos = []
for ci, (S, starts) in enumerate(te):
    for ui in range(1, len(starts)):
        e = starts[ui + 1] if ui + 1 < len(starts) else len(S)
        for t in range(starts[ui] + 1, e, 11): pos.append((ci, ui, t))
rng.shuffle(pos); pos = pos[:int(os.environ.get('NPOS', 700))]; NP = len(pos); P('positions', NP)
Pb = np.zeros((NP, 32768), np.float32); F = np.zeros((3, NP, 32768), np.float16); Y = np.zeros(NP, np.int64); t0 = time.time()
for k, (ci, ui, t) in enumerate(pos):
    S, starts = te[ci]; ctx = S[:t].tolist(); Y[k] = S[t]; p = ch.dist(ctx); Pb[k] = p; F[:, k] = ch.feats(ctx, starts[ui], p)
    if k % 100 == 0: print('feat', k, time.time() - t0, flush=True)
half = NP // 2
def bits(w, a, b):
    tot = 0.0
    for k in range(a, b):
        s = sum(w[i] * F[i, k].astype(np.float32) for i in range(3) if w[i] != 0) if any(w) else np.zeros(32768, np.float32)
        e = Pb[k] * np.exp(s - s.max()); tot -= np.log2(max(e[Y[k]] / e.sum(), 1e-30))
    return tot / (b - a)
w = [0.0, 0.0, 0.0]; P('100億語の索引: 基準 tune/test bits', bits(w, 0, half), bits(w, half, NP))
for sweep in range(2):
    for i in range(3):
        best = (bits(w, 0, half), w[i])
        for v in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):
            w2 = list(w); w2[i] = v; b = bits(w2, 0, half)
            if b < best[0]: best = (b, v)
        w[i] = best[1]; P('  fit', sweep, i, w, best[0])
P('最終 重み', w, 'test bits', bits(w, half, NP))
for nm, ww in (('文書のみ', [w[0], 0, 0]), ('アライメントのみ', [0, w[1], 0]), ('会話n-gramのみ', [0, 0, w[2]]), ('文書+アライメント', [w[0], w[1], 0])): P(nm, 'test bits', bits(ww, half, NP))
ch.w = w; del F, Pb
# 2. 会話サンプル
D = [['私の名前は田中太郎です。好きな食べ物はカレーです。', 'こんにちは。', '私の名前を覚えていますか？'], ['My name is Michael Faraday and I study electricity.', 'Nice to meet you!', 'What is my name?'],
     ['こんにちは、今日は何してた？'], ['日本の首都はどこですか？'], ['最近疲れてるんだよね'], ['Can you recommend a good book about space?'], ['おすすめの映画を教えて'], ['Pythonでリストを逆順にするには？']]
for u in D:
    for plain in (True, False):
        for seed in (1, 2):
            P(('特徴なし' if plain else '特徴あり'), '|', u[-1][:30], '->', ch.reply3(u, plain=plain, seed=seed, maxlen=60)[:220].replace('\n', ' '))
# 3. 記憶テスト (名前の語の 平均 対数確率)
names = ['Faraday', 'Michael', 'Sarah', 'Thompson', 'Alexander', 'Jennifer', 'Roberts', 'Elizabeth', 'Anderson', 'Oliver', 'Natalie', 'Winston', 'Gabriel', 'Patricia', 'Hamilton', 'Marcus']
fill = ['The weather is nice today.', 'I went to the market and bought some apples.', 'Do you like music?', 'It is raining outside.', 'Let me tell you about my trip to the mountains.', 'The train was late this morning.', 'I like reading books about history.', 'What is your favorite color?']
def lp(ctx, h0, tgt, mode):
    t = 0.0; c = list(ctx)
    for x in tgt:
        p = fm.dist(c) if mode == 0 else ch.dist3(c, h0); t += np.log(max(p[x], 1e-12)); c.append(x)
    return t / len(tgt)
for nf in (0, 2, 5, 8):
    r = {0: [], 1: []}
    for nm in names:
        u = ['Hello, my name is %s.' % nm] + fill[:nf] + ['Do you remember my name?']; ctx = [0]
        for x in u: ctx += tk.encode(x).ids + [2]
        h0 = len(ctx); ctx += tk.encode('Yes, your name is').ids; tgt = tk.encode(' ' + nm).ids
        for m in (0, 1): r[m].append(lp(ctx, h0, tgt, m))
    P('記憶テスト 間の発話数 %d: 特徴なし %.3f  v3 %.3f  (差 %.3f)' % (nf, np.mean(r[0]), np.mean(r[1]), np.mean(r[1]) - np.mean(r[0])))
open(W + '/chat3_out.txt', 'w').write('\n'.join(out)); api.upload_file(path_or_fileobj=W + '/chat3_out.txt', path_in_repo='results/chat3/out.txt', repo_id=R, repo_type='dataset')
print('CHAT3_DONE', flush=True)
