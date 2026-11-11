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
# 1. 索引に 入っていない 検証用の 会話 (fm/val/val_dlg_*.bin) で 測る。 重みは se + izumi + rpc_test の 前半で 数え、 後半と 各ソース別に 評価
SRC = ['dlg_se', 'dlg_izumi', 'dlg_rpc_test', 'dlg_ja_oasst', 'dlg_oasst1']; rng = np.random.default_rng(0); pos = []; docs = {}
for sname in SRC:
    p = hf_hub_download(R, 'fm/val/val_%s.bin' % sname, repo_type='dataset', token=TK, local_dir=W + '/valx'); T = np.fromfile(p, dtype=np.uint16).astype(np.int64)
    cuts = np.where(T == 0)[0]; ds = []
    for i in range(len(cuts)):
        S = T[cuts[i]:(cuts[i + 1] if i + 1 < len(cuts) else len(T))][:1200]
        if len(S) < 40: continue
        starts = [1] + [int(x) + 1 for x in np.where(S[:-1] == 2)[0]]; ds.append((S, [x for x in starts if x < len(S)]))
    docs[sname] = ds; pp = []
    for di, (S, starts) in enumerate(ds):
        for ui in range(1, len(starts)):
            e = starts[ui + 1] if ui + 1 < len(starts) else len(S)
            for t in range(starts[ui] + 1, e, 13): pp.append((sname, di, ui, t))
    rng.shuffle(pp); pp = pp[:int(os.environ.get('NPS', 260))]; pos += pp; P(sname, 'docs', len(ds), 'positions', len(pp))
order = {'dlg_se': 0, 'dlg_izumi': 0, 'dlg_rpc_test': 0}
fitp = [x for x in pos if x[0] in order]; othp = [x for x in pos if x[0] not in order]; rng.shuffle(fitp)
pos = fitp + othp; NP = len(pos); nfit = len(fitp); half = nfit // 2
Pb = np.zeros((NP, 32768), np.float32); F = np.zeros((3, NP, 32768), np.float16); Y = np.zeros(NP, np.int64); t0 = time.time()
for k, (sname, di, ui, t) in enumerate(pos):
    S, starts = docs[sname][di]; ctx = S[:t].tolist(); Y[k] = S[t]; p = ch.dist(ctx); Pb[k] = p; F[:, k] = ch.feats(ctx, starts[ui], p)
    if k % 100 == 0: print('feat', k, time.time() - t0, flush=True)
def bits(w, a, b):
    tot = 0.0
    for k in range(a, b):
        s = sum(w[i] * F[i, k].astype(np.float32) for i in range(3) if w[i] != 0) if any(w) else np.zeros(32768, np.float32)
        e = Pb[k] * np.exp(s - s.max()); tot -= np.log2(max(e[Y[k]] / e.sum(), 1e-30))
    return tot / (b - a)
w = [0.0, 0.0, 0.0]; P('100億語の索引 (索引外の会話): 基準 tune/test bits', bits(w, 0, half), bits(w, half, nfit))
for sweep in range(2):
    for i in range(3):
        best = (bits(w, 0, half), w[i])
        for v in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):
            w2 = list(w); w2[i] = v; b = bits(w2, 0, half)
            if b < best[0]: best = (b, v)
        w[i] = best[1]; P('  fit', sweep, i, w, best[0])
P('最終 重み', w, 'test bits', bits(w, half, nfit))
for nm, ww in (('基準', [0, 0, 0]), ('v3 (fit)', w), ('v3 (24M 重み)', [0.5, 1.0, 0.75])):
    j = nfit
    for sname in SRC:
        idx = [k for k in range(NP) if pos[k][0] == sname and (sname not in order or k >= half)]
        if idx: P('  ', nm, sname, 'bits', sum(bits(ww, k, k + 1) for k in idx) / len(idx))
for nm, ww in (('文書のみ', [w[0], 0, 0]), ('アライメントのみ', [0, w[1], 0]), ('会話n-gramのみ', [0, 0, w[2]]), ('文書+アライメント', [w[0], w[1], 0])): P(nm, 'test bits', bits(ww, half, nfit))
ch.w = [0.5, 1.0, 0.75] if os.environ.get('W24', '1') == '1' else w; del F, Pb
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
open(W + '/chat4_out.txt', 'w').write('\n'.join(out)); api.upload_file(path_or_fileobj=W + '/chat4_out.txt', path_in_repo='results/chat4/out.txt', repo_id=R, repo_type='dataset')
print('CHAT4_DONE', flush=True)
