import json, re, os, time, hashlib, threading, numpy as np, scipy.sparse as sp, pickle
from huggingface_hub import hf_hub_download
HFT = os.environ['HF_TOKEN']; LOG = open('/content/ret.log', 'a')

def L(m): LOG.write('%s %s\n' % (time.strftime('%H:%M:%S'), m)); LOG.flush()
JA = re.compile(r'[぀-ヿ一-鿿]')
def to_utts(d):
    if 'conversations' in d: return [c.get('value', '') for c in d['conversations'] if c.get('value')]
    if 'instruction' in d:
        a = d['instruction'] + (('\n' + d['input']) if d.get('input') else ''); return [a, d.get('output', '')] if d.get('output') else []
    return []
PAIRS = []; SEEN = set()
def add(src, q, a):
    q = q.strip(); a = a.strip()
    if not (2 <= len(q) <= 300 and 2 <= len(a) <= 400 and JA.search(q) and JA.search(a)): return
    h = hashlib.md5((q + '|' + a).encode()).digest()[:8]
    if h in SEEN: return
    SEEN.add(h); PAIRS.append((src, q, a))
def load(src, repo, path, aux=False, adj=True):
    p = hf_hub_download('Youichi1/ai-lab-data' if aux else repo, path, repo_type='dataset', token=HFT, local_dir='/content/ret')
    n0 = len(PAIRS)
    for ln in open(p, encoding='utf-8'):
        ln = ln.strip()
        if not ln: continue
        try: d = json.loads(ln)
        except Exception: continue
        u = d if isinstance(d, list) else to_utts(d)
        if len(u) < 2: continue
        if adj:
            for i in range(len(u) - 1): add(src, u[i], u[i + 1])
        else: add(src, u[0], u[1])
    L('%s: +%d' % (src, len(PAIRS) - n0))
def build():
    try:
        load('rpc', None, 'aux/rpc_train.jsonl', aux=True)
        load('ja_oasst1', 'llm-jp/oasst1-21k-ja', 'oasst1-21k-ja.jsonl'); load('ja_oasst2', 'llm-jp/oasst2-33k-ja', 'oasst2-33k-ja.jsonl'); load('dolly_ja', 'llm-jp/databricks-dolly-15k-ja', 'databricks-dolly-15k-ja.jsonl', adj=False)
        load('izumi', 'izumi-lab/llm-japanese-dataset', 'data-cc-by-sa.jsonl', adj=False)
        L('pairs %d' % len(PAIRS)); pickle.dump(PAIRS, open('/content/ret/pairs.pkl', 'wb')); L('saved'); open('/content/ret/DONE', 'w').write('ok')
    except Exception as e:
        import traceback; L('ERR ' + traceback.format_exc())
threading.Thread(target=build, daemon=True).start(); print('started')
