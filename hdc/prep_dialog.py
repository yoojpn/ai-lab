"""会話データ (人が書いた 質問と答え) を 作る。 dolly-15k (CC BY-SA 3.0)・oasst1 (Apache-2.0)・ichikara-instruction-003 (CC BY-NC-SA) を 取得して、
同じ書式に直し、 BPE で 数字にする。 検証用の 質問は 各データから 取り分けておく。
使い方: python3 prep_dialog.py [出力先=/tmp/dlg_out]     (取得したファイルは /tmp/dlg_raw に置く)
出力: dlg_train.bin (uint16)  dlg_train_docs.npy (各例の開始位置)  dlg_test.json (検証用の 質問と 正解)  dlg_stats.json
"""
import sys, os, json, re, random, urllib.request
import numpy as np
from tokenizers import Tokenizer

RAW = '/tmp/dlg_raw'; OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/dlg_out'; W = '/tmp/tok'
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
MAXT = 512; NTEST = 30


def get(repo, fn):
    p = RAW + '/' + fn.replace('/', '_')
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        urllib.request.urlretrieve('https://huggingface.co/datasets/%s/resolve/main/%s' % (repo, fn), p)
    return p


def fmt(q, a): return '### 質問\n%s\n### 回答\n%s\n\n' % (q.strip(), a.strip())
def prompt_of(q): return '### 質問\n%s\n### 回答\n' % q.strip()


def load_dolly():
    out = []
    for l in open(get('databricks/databricks-dolly-15k', 'databricks-dolly-15k.jsonl'), encoding='utf-8'):
        r = json.loads(l); q = r['instruction'] + (('\n' + r['context']) if r['context'].strip() else '')
        out.append(('dolly', q, r['response']))
    return out


def load_oasst():
    import pandas as pd
    df = pd.concat([pd.read_parquet(get('OpenAssistant/oasst1', f)) for f in ('data/train-00000-of-00001-b42a775f407cee45.parquet', 'data/validation-00000-of-00001-134b8fd0c89408b6.parquet')])
    df = df[df['lang'].isin(['en', 'ja'])]; by = {r.message_id: r for r in df.itertuples()}
    out = []
    for r in df.itertuples():
        if r.role != 'assistant' or r.parent_id not in by: continue
        if r.rank is not None and not (r.rank != r.rank) and r.rank > 0: continue      # 最上位の答えだけ (rank 0)。 順位なしは 残す
        p = by[r.parent_id]
        if p.role != 'prompter': continue
        out.append(('oasst_' + r.lang, p.text, r.text))
    return out


def load_json_lenient(p):
    s = open(p, encoding='utf-8').read()
    try: return json.loads(s)
    except Exception: return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s))


def load_ichikara():
    out = []
    for fn in ['ichikara-instruction-003-001-1.json', 'ichikara-instruction-003-001-2.1.json', 'ichikara-instruction-003-001-2.2.json', 'ichikara-instruction-003-001-5.1.json',
               'ichikara-instruction-003-001-5.2.json', 'ichikara-instruction-003-002-1.json', 'ichikara-instruction-003-003-1.json']:
        try: d = load_json_lenient(get('kinokokoro/ichikara-instruction-003', fn))
        except Exception as e: print('読めない', fn, repr(e)[:80]); continue
        for r in d:
            if r.get('text') and r.get('output'): out.append(('ichikara', r['text'], r['output']))
    return out


def main():
    random.seed(0); tk = Tokenizer.from_file(W + '/tok.json'); ex = load_dolly() + load_oasst() + load_ichikara()
    by = {}
    for e in ex: by.setdefault(e[0], []).append(e)
    train, test = [], []
    for k, v in by.items():
        random.shuffle(v); test += v[:NTEST]; train += v[NTEST:]
    random.shuffle(train)
    toks, starts, n_trunc = [], [], 0
    for src, q, a in train:
        ids = tk.encode(fmt(q, a)).ids
        if len(ids) > MAXT: ids = ids[:MAXT]; n_trunc += 1
        starts.append(len(toks)); toks += ids
    np.asarray(toks, dtype=np.uint16).tofile(OUT + '/dlg_train.bin'); np.save(OUT + '/dlg_train_docs.npy', np.asarray(starts, dtype=np.int64))
    json.dump([dict(src=s, prompt=prompt_of(q), ref=a.strip()) for s, q, a in test], open(OUT + '/dlg_test.json', 'w'), ensure_ascii=False)
    st = dict(n_by_src={k: len(v) for k, v in by.items()}, n_train=len(train), n_test=len(test), train_tokens=len(toks), truncated=n_trunc)
    json.dump(st, open(OUT + '/dlg_stats.json', 'w'), ensure_ascii=False); print('DLG_STATS', json.dumps(st, ensure_ascii=False))


if __name__ == '__main__':
    main()
