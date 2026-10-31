"""
会話 (jsonl.gz) を BPE にして uint16 の列にする。
  形式: <bos> 発言1 <turn> 発言2 <turn> ... <eos>   (特別語: <bos>=0, <eos>=1, <turn>=2)
  分割: 本の id のハッシュで 学習 96% / 検証 2% / テスト 2% (本ごと: 同じ本の会話が混ざらない)
  出力: /tmp/tok/{train,val,test}.bin (先頭に 0 を 8 語の余白), {split}_docs.npy (各文書の開始位置, 余白込み), {split}_lang.npy (0=en,1=ja), tok.json (tokenizer), meta.json
使い方: python3 tok_dlg.py '{"in":"/tmp/dlg","out":"/tmp/tok","vocab":32768}'
"""
import os, sys, gzip, json, glob, time, hashlib
import numpy as np
from tokenizers import ByteLevelBPETokenizer


def split_of(b):
    h = int(hashlib.md5(str(b).encode()).hexdigest(), 16) % 50
    return 'test' if h == 0 else 'val' if h == 1 else 'train'


def read(files):
    for f in files:
        with gzip.open(f, 'rt', encoding='utf8') as fh:
            for line in fh:
                r = json.loads(line); yield r


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    files = sorted(glob.glob(cfg['in'] + '/*.jsonl.gz')); os.makedirs(cfg['out'], exist_ok=True)
    tk = ByteLevelBPETokenizer()
    def sample():
        for i, r in enumerate(read(files)):
            if i % 25 == 0 and split_of(r['b']) == 'train':
                yield '\n'.join(r['t'])
    tk.train_from_iterator(sample(), vocab_size=cfg.get('vocab', 32768), min_frequency=2, special_tokens=['<bos>', '<eos>', '<turn>'], show_progress=False)
    assert tk.token_to_id('<bos>') == 0 and tk.token_to_id('<eos>') == 1 and tk.token_to_id('<turn>') == 2
    tk.save(cfg['out'] + '/tok.json'); log('BPE 学習済み 語彙 %d' % tk.get_vocab_size())
    outs = {s: open('%s/%s.bin' % (cfg['out'], s), 'wb') for s in ('train', 'val', 'test')}
    for f in outs.values(): f.write(np.zeros(8, np.uint16).tobytes())
    pos = {s: 8 for s in outs}; docs = {s: [] for s in outs}; lang = {s: [] for s in outs}; nd = {s: 0 for s in outs}
    buf = []
    def flush():
        if not buf: return
        turns = [t for r in buf for t in r['t']]
        enc = tk.encode_batch(turns); k = 0
        for r in buf:
            ids = [0]
            for j in range(len(r['t'])):
                ids += enc[k].ids; k += 1
                ids.append(2 if j + 1 < len(r['t']) else 1)
            s = split_of(r['b']); docs[s].append(pos[s]); lang[s].append(1 if r['l'] == 'ja' else 0)
            a = np.array(ids, np.uint16); outs[s].write(a.tobytes()); pos[s] += len(a); nd[s] += 1
        buf.clear()
    for i, r in enumerate(read(files)):
        if sum(len(t) for t in r['t']) > 8000: continue
        buf.append(r)
        if len(buf) >= 20000:
            flush()
            if (i // 20000) % 10 == 0: log('%d 会話 / 語 %s' % (i, {s: pos[s] - 8 for s in pos}))
    flush()
    for s, f in outs.items():
        f.close(); np.save('%s/%s_docs.npy' % (cfg['out'], s), np.array(docs[s], np.int64)); np.save('%s/%s_lang.npy' % (cfg['out'], s), np.array(lang[s], np.int8))
    meta = {s: dict(tokens=pos[s] - 8, docs=nd[s]) for s in outs}; json.dump(meta, open(cfg['out'] + '/meta.json', 'w')); log('完了 %s' % meta)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
