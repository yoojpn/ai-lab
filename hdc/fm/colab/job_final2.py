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
SRC = ['dlg_se', 'dlg_izumi', 'dlg_rpc_test']; rng = np.random.default_rng(0); pos = []; docs = {}
for sname in SRC:
    p = hf_hub_download(R, 'fm/val/val_%s.bin' % sname, repo_type='dataset', token=TK, local_dir=W + '/valx'); T = np.fromfile(p, dtype=np.uint16).astype(np.int64)
    ends = np.where(T == 2)[0] + 1; ds = []; i = 0
    while i < len(ends) - 1:
        j = i + 1
        while j < len(ends) - 1 and ends[j] - (ends[i - 1] if i > 0 else 0) < 350: j += 1
        a0 = ends[i - 1] if i > 0 else 0; S = np.concatenate([[0], T[a0:ends[j]]])[:1200]; i = j + 1
        if len(S) < 40: continue
        starts = [1] + [int(x) + 1 for x in np.where(S[:-1] == 2)[0]]; ds.append((S, [x for x in starts if x < len(S)]))
    docs[sname] = ds; pp = []
    for di, (S, starts) in enumerate(ds):
        for ui in range(1, len(starts)):
            e = starts[ui + 1] if ui + 1 < len(starts) else len(S)
            for t in range(starts[ui] + 1, e, 13): pp.append((sname, di, ui, t))
    rng.shuffle(pp); pos += pp[:int(os.environ.get('NPS', 220))]
rng.shuffle(pos); NB = len(pos)
names = ['Faraday', 'Michael', 'Sarah', 'Thompson', 'Alexander', 'Jennifer', 'Roberts', 'Elizabeth', 'Anderson', 'Oliver', 'Natalie', 'Winston']
fill = ['The weather is nice today.', 'I went to the market and bought some apples.', 'Do you like music?', 'It is raining outside.', 'Let me tell you about my trip to the mountains.']
rc = []; LANG = []
for nm in names:
    for nf in (0, 5):
        u = ['Hello, my name is %s.' % nm] + fill[:nf] + ['Do you remember my name?']; ctx = [0]; qs = None
        for x in u:
            ids = tk.encode(x).ids; qs = (len(ctx), len(ctx) + len(ids) + 1); ctx += ids + [2]
        h0 = len(ctx); ctx += tk.encode('Yes, your name is').ids; tgt = tk.encode(' ' + nm).ids; c = list(ctx)
        for x in tgt: rc.append((list(c), h0, qs, x)); c.append(x); LANG.append('en')
JN = ['田中太郎', '佐藤花子', '鈴木一郎', '高橋美咲', '伊藤健太', '渡辺優子', '山本大輔', '中村恵']; JF = ['今日はいい天気ですね。', '最近、映画を見ました。', 'お腹がすきました。', '電車が遅れました。', '本を読んでいます。']
for nm in JN:
    for nf in (0, 5):
        u = ['私の名前は%sです。' % nm] + JF[:nf] + ['私の名前を覚えていますか？']; ctx = [0]; qs = None
        for x in u:
            ids = tk.encode(x).ids; qs = (len(ctx), len(ctx) + len(ids) + 1); ctx += ids + [2]
        h0 = len(ctx); ctx += tk.encode('はい、あなたの名前は').ids; tgt = tk.encode(nm).ids; c = list(ctx)
        for x in tgt: rc.append((list(c), h0, qs, x)); c.append(x); LANG.append('ja')
NR = len(rc); NP = NB + NR; P('bit 位置', NB, '記憶 位置', NR)
Pb = np.zeros((NP, 32768), np.float32); F = np.zeros((4, NP, 32768), np.float16); Y = np.zeros(NP, np.int64); PH = np.zeros(NP, np.int64); t0 = time.time()
for k in range(NB):
    sname, di, ui, t = pos[k]; S, starts = docs[sname][di]; ctx = S[:t].tolist(); Y[k] = S[t]; PH[k] = int(t - starts[ui] >= 8); p = ch.dist(ctx); Pb[k] = p; F[:, k] = feats4(ctx, starts[ui], p, starts[ui - 1], starts[ui])
    if k % 100 == 0: print('feat', k, time.time() - t0, flush=True)
for r, (ctx, h0, qs, x) in enumerate(rc):
    k = NB + r; Y[k] = x; PH[k] = 1; p = ch.dist(ctx); Pb[k] = p; F[:, k] = feats4(ctx, h0, p, qs[0], qs[1])
def nll(w, a, b):
    tot = 0.0
    for k in range(a, b):
        wk = w[PH[k] * 4:PH[k] * 4 + 4]; s = sum(wk[i] * F[i, k].astype(np.float32) for i in range(4) if wk[i] != 0) if any(wk) else np.zeros(32768, np.float32)
        e = Pb[k] * np.exp(s - s.max()); tot -= np.log2(max(e[Y[k]] / e.sum(), 1e-30))
    return tot / (b - a)
half = NB // 2; GAM = float(os.environ.get('GAM', 0.15))
def J(w): return nll(w, 0, half) + GAM * nll(w, NB, NP)
w = [0.0] * 8; P('基準: bit tune/test', nll(w, 0, half), nll(w, half, NB), '記憶 (bit/語)', nll(w, NB, NP))
for sweep in range(2):
    for i in range(8):
        best = (J(w), w[i])
        for v in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
            w2 = list(w); w2[i] = v; j = J(w2)
            if j < best[0]: best = (j, v)
        w[i] = best[1]; P('  fit', sweep, i, w, best[0])
P('最終 重み (joint)', w, 'bit test', nll(w, half, NB), '記憶 (bit/語)', nll(w, NB, NP))
for lg in ('en', 'ja'):
    idx = [NB + r for r in range(NR) if LANG[r] == lg]; P('  記憶', lg, '基準', sum(nll([0.0] * 8, k, k + 1) for k in idx) / len(idx), 'joint', sum(nll(w, k, k + 1) for k in idx) / len(idx))
w0 = [0.35, 1.5, 0.1, 0.0, 0.1, 1.5, 0.0, 0.0]; P('前の 重み (bit のみ)', w0, 'bit test', nll(w0, half, NB), '記憶 (bit/語)', nll(w0, NB, NP))
ch.w = w; save('weights'); del F, Pb
# --- 文脈内学習 (ICL): 確信度つきコピー特徴 の 有無
import json as _j
TT = _j.load(open('/content/ev/a/fm/chat3/icl_tasks.json')); TT = [t for i, t in enumerate(TT) if i % 3 == 0]
def gen6(ids, feat):
    g = []
    for _ in range(6):
        p = np.asarray(fm.dist(ids + g) if not feat else ch.dist3(ids + g, len(ids), w=[w[0], w[1], 0.0], late=True), np.float64); p[0] = 0; g.append(int(np.argmax(p)))
    return tk.decode(g)
res = {}
for t in TT:
    ids = [0] + tk.encode(t['prompt']).ids; a = t['answer'].strip()
    for nm, ft in (('素', False), ('特徴', True)):
        r = res.setdefault((t['task'], nm), [0, 0]); r[0] += gen6(ids, ft).strip().startswith(a); r[1] += 1
for k, v in sorted(res.items()): P('ICL', k[0], k[1], '%d/%d' % tuple(v))
save('icl')
# --- 会話サンプル (特徴つき) と 名前の 記憶 (強制)
D = [['こんにちは、今日は何してた？'], ['最近疲れてるんだよね'], ['日本の首都はどこですか？'], ['私の名前は田中太郎です。', 'こんにちは。', '私の名前を覚えていますか？'], ['My name is Michael Faraday and I study electricity.', 'Nice to meet you!', 'What is my name?']]
for u in D:
    for seed in (1, 2): P('特徴あり |', u[-1][:30], '->', ch.reply3(u, plain=False, seed=seed, maxlen=50)[:200].replace('\n', ' '))
save('done'); print('FINAL2_DONE', flush=True)
