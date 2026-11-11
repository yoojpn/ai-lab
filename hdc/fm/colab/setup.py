import subprocess, os, json
os.environ.setdefault('HF_TOKEN', '')   # 事前に 環境に 設定
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); return (r.stdout + r.stderr)[-1500:]
print(sh('pip -q install tokenizers huggingface_hub 2>&1 | tail -1'))
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ['HF_TOKEN']); R = 'Youichi1/ai-lab-data'; W = '/content/ev'; TK = os.environ['HF_TOKEN']
for f in ('fmeng.cpp', 'fm_chat.py'): hf_hub_download(R, 'fm/code/' + f, repo_type='dataset', token=TK, local_dir=W + '/code')
hf_hub_download(R, 'tok/tok.json', repo_type='dataset', token=TK, local_dir=W + '/tokhf')
hf_hub_download(R, 'results/fm_eval_result.json', repo_type='dataset', token=TK, local_dir=W + '/res')
json.dump({'lam': json.load(open(W + '/res/results/fm_eval_result.json'))['val_ja_web']['lam']}, open(W + '/lam_chat.json', 'w'))
CD = W + '/code/fm/code'; print(sh('cd ' + CD + ' && g++ -O2 -march=native -fopenmp fmeng.cpp -o fmeng 2>&1 | tail -3'))
files = api.list_repo_files(R, repo_type='dataset'); lst = []
for s in sorted(f for f in files if f.startswith('fm/shard_') and f.endswith('.bwt.xz')):
    m = s.replace('.bwt.xz', '.meta.json')
    a = hf_hub_download(R, s, repo_type='dataset', token=TK, local_dir=W + '/sh'); b = hf_hub_download(R, m, repo_type='dataset', token=TK, local_dir=W + '/sh'); lst.append(a + ' ' + b)
open(W + '/list.txt', 'w').write('\n'.join(lst) + '\n'); print('分割', len(lst))
open(W + '/chatd.py', 'w').write(open('/content/chatd.py').read())
