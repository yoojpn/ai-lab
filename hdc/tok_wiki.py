"""Wikipedia (HF wikimedia/wikipedia 20231101, en と ja) の一部を、 会話と同じ BPE(tok.json) で uint16 の列にする。
  標本: 記事 id のハッシュ b = md5(id)%100 -> 学習 b < pct / 検証 b = pct / テスト b = pct+1   (pct は言語ごと: en 10, ja 15)
  1 記事 = <bos> 先頭 8000 字 <eos>。 出力は tok_dlg.py と同じ形式 (/tmp/wiki/{train,val,test}.bin, *_docs.npy, *_lang.npy, meta.json)
使い方: python3 tok_wiki.py '{"tok":"/tmp/tok/tok.json","out":"/tmp/wiki","pct":{"en":10,"ja":15}}'"""
import os, sys, json, time, hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    tk = Tokenizer.from_file(cfg['tok']); os.makedirs(cfg['out'], exist_ok=True); dl = cfg['out'] + '/dl'
    outs = {s: open('%s/%s.bin' % (cfg['out'], s), 'wb') for s in ('train', 'val', 'test')}
    for f in outs.values(): f.write(np.zeros(8, np.uint16).tobytes())
    pos = {s: 8 for s in outs}; docs = {s: [] for s in outs}; lang = {s: [] for s in outs}
    api = HfApi(); jobs = []
    for lg in ('en', 'ja'):
        for f in sorted(x.path for x in api.list_repo_tree('wikimedia/wikipedia', repo_type='dataset', path_in_repo='20231101.' + lg)):
            jobs.append((lg, f))
    ex = ThreadPoolExecutor(1); fut = ex.submit(hf_hub_download, 'wikimedia/wikipedia', jobs[0][1], repo_type='dataset', local_dir=dl)
    for ji, (lg, f) in enumerate(jobs):
        p = fut.result()
        if ji + 1 < len(jobs): fut = ex.submit(hf_hub_download, 'wikimedia/wikipedia', jobs[ji + 1][1], repo_type='dataset', local_dir=dl)
        pct = cfg['pct'][lg]; pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=2000, columns=['id', 'text']):
            ids = batch.column('id').to_pylist(); txt = batch.column('text').to_pylist(); sel = []
            for i, t in zip(ids, txt):
                b = int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 100
                s = 'train' if b < pct else 'val' if b == pct else 'test' if b == pct + 1 else None
                if s and len(t) > 200: sel.append((s, t[:8000]))
            if not sel: continue
            enc = tk.encode_batch([t for _, t in sel])
            for (s, _), e in zip(sel, enc):
                a = np.array([0] + e.ids + [1], np.uint16); docs[s].append(pos[s]); lang[s].append(1 if lg == 'ja' else 0); outs[s].write(a.tobytes()); pos[s] += len(a)
        os.remove(p)
        log('%d/%d %s: %s' % (ji + 1, len(jobs), os.path.basename(f), {s: pos[s] - 8 for s in pos}))
    for s, f in outs.items():
        f.close(); np.save('%s/%s_docs.npy' % (cfg['out'], s), np.array(docs[s], np.int64)); np.save('%s/%s_lang.npy' % (cfg['out'], s), np.array(lang[s], np.int8))
    meta = {s: dict(tokens=pos[s] - 8, docs=len(docs[s])) for s in outs}; json.dump(meta, open(cfg['out'] + '/meta.json', 'w')); log('完了 %s' % meta)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
