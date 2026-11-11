import subprocess, os, time, sys
os.environ.setdefault('HF_TOKEN', '')
subprocess.run('pip -q install huggingface_hub scikit-learn scipy 2>&1 | tail -1', shell=True)
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ['HF_TOKEN']); R = 'Youichi1/ai-lab-data'; W = '/content/sl'; TK = os.environ['HF_TOKEN']
for f in ('code/slat/engine.py', 'code/slat/state_lattice.py', 'tok/train.bin', 'tok/val.bin'): hf_hub_download(R, f, repo_type='dataset', token=TK, local_dir=W)
CD = W + '/code/slat'
def up(tag=''):
    out = ''
    for n in sorted(os.listdir(W + '/logs')): 
        t = open(W + '/logs/' + n).read(); out += '=== %s ===\n' % n + '\n'.join(l for l in t.split('\n') if 'RESULT' in l or '状態' in l or 'Error' in l or 'Traceback' in l or '評価' in l) + '\n' + t[-600:] + '\n'
    open(W + '/slat_results.txt', 'w').write(out)
    for i in range(5):
        try: api.upload_file(path_or_fileobj=W + '/slat_results.txt', path_in_repo='results/slat_results%s.txt' % tag, repo_id=R, repo_type='dataset'); break
        except Exception as e: time.sleep(60)
os.makedirs(W + '/logs', exist_ok=True)
JOBS = [('s96_k4096', 96000000, {}), ('s96_k16384', 96000000, {'ZK': '16384'}), ('s24_k16384', 24000000, {'ZK': '16384'}), ('s6_k16384', 6000000, {'ZK': '16384'}), ('s96_k16384_r10', 96000000, {'ZK': '16384', 'ZR': '10'}), ('s384_k16384', 384000000, {'ZK': '16384'})]
PAR = 3; running = []; q = list(JOBS)
while q or running:
    while q and len(running) < PAR:
        name, S, env = q.pop(0); e = dict(os.environ); e.update(env); e['OMP_NUM_THREADS'] = '2'
        p = subprocess.Popen('cd %s && python3 state_lattice.py %s/tok %s/w_%s %d > %s/logs/%s.log 2>&1; rm -rf %s/w_%s' % (CD, W, W, name, S, W, name, W, name), shell=True, env=e); running.append((name, p))
    time.sleep(120); running = [(n, p) for n, p in running if p.poll() is None]; up()
up(); print('SLAT_DONE')
