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
for f in ('fmeng.cpp', 'fm_chat.py', 'chat2.py'): hf_hub_download(R, 'fm/code/' + f, repo_type='dataset', token=TK, local_dir=W + '/code')
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
from chat2 import Chat2
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.environ['TOKJSON'])
lamj = os.path.join(W, 'lam.json')
fm = fm_chat.FM(W + '/list.txt'); print('index ready', flush=True)
A = W + '/a/fm/chat2/'; ch = Chat2(fm.dist, np.load(A + 'E.npy'), np.load(A + 'uni.npy'), np.load(A + 'combo_lam.npy')[:9], np.load(A + 'cls64.npy'), tk)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.append(s)
# 1. 会話サンプル
D = [['私の名前は田中太郎です。好きな食べ物はカレーです。', 'こんにちは。', '私の名前を覚えていますか？'],
     ['My name is Michael Faraday and I study electricity.', 'Nice to meet you!', 'What is my name?'],
     ['こんにちは、今日は何してた？'], ['日本の首都はどこですか？'], ['最近疲れてるんだよね'], ['Can you recommend a good book about space?']]
for u in D:
    for plain in (True, False):
        for seed in (1, 2):
            P(('特徴なし' if plain else '特徴あり'), '|', u[-1][:30], '->', ch.reply(u, plain=plain, seed=seed, maxlen=60)[:200].replace('\n', ' '))
# 2. 記憶テスト: 名前を言った後に つなぎの発話を n 個挟み、 「あなたの名前は」 の続きで 名前の語の対数確率を測る (強制)
names = ['Faraday', 'Michael', 'Sarah', 'Thompson', 'Alexander', 'Jennifer', 'Roberts', 'Elizabeth', 'Anderson', 'Oliver', 'Natalie', 'Winston', 'Gabriel', 'Patricia', 'Hamilton', 'Marcus']
fill = ['The weather is nice today.', 'I went to the market and bought some apples.', 'Do you like music?', 'It is raining outside.', 'Let me tell you about my trip to the mountains.', 'The train was late this morning.', 'I like reading books about history.', 'What is your favorite color?']
def lp(ctx, tgt, feat):
    t = 0.0; c = list(ctx)
    for x in tgt:
        p = ch.next_dist(c, c) if feat else fm.dist(c); t += np.log(max(p[x], 1e-12)); c.append(x)
    return t / len(tgt)
for nf in (0, 2, 5, 8):
    r = {False: [], True: []}
    for nm in names:
        u = ['Hello, my name is %s.' % nm] + fill[:nf] + ['Do you remember my name?']; ctx = [0]
        for x in u: ctx += tk.encode(x).ids + [2]
        ctx += tk.encode('Yes, your name is').ids; tgt = tk.encode(' ' + nm).ids
        base = [0]
        for x in ['Hello, my name is Zzz.'] + fill[:nf] + ['Do you remember my name?']: base += tk.encode(x).ids + [2]
        base += tk.encode('Yes, your name is').ids
        for feat in (False, True): r[feat].append(lp(ctx, tgt, feat))
    P('記憶テスト 間の発話数 %d: 名前の語の平均 対数確率  特徴なし %.3f  特徴あり %.3f  (差 %.3f)' % (nf, np.mean(r[False]), np.mean(r[True]), np.mean(r[True]) - np.mean(r[False])))
open(W + '/chat2_out.txt', 'w').write('\n'.join(out)); api.upload_file(path_or_fileobj=W + '/chat2_out.txt', path_in_repo='results/chat2/out.txt', repo_id=R, repo_type='dataset')
print('CHAT2_DONE', flush=True)
