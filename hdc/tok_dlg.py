"""
会話 (jsonl.gz) を BPE にして uint16 の列にする。
  形式: <bos> 発言1 <turn> 発言2 <turn> ... <eos>   (特別語: <bos>=0, <eos>=1, <turn>=2)
  分割: 本の id のハッシュで 学習 96% / 検証 2% / テスト 2% (本ごと: 同じ本の会話が混ざらない)
  出力: /tmp/tok/{train,val,test}.bin (先頭に 0 を 8 語の余白), {split}_docs.npy (各文書の開始位置, 余白込み), {split}_lang.npy (0=en,1=ja), tok.json (tokenizer), meta.json
使い方: python3 tok_dlg.py '{"in":"/tmp/dlg","out":"/tmp/tok","vocab":32768}'
"""
import os, sys, gzip, json, glob, time, hashlib
import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer


def split_of(b):
    h = int(hashlib.md5(str(b).encode()).hexdigest(), 16) % 50
    return 'test' if h == 0 else 'val' if h == 1 else 'train'


def keys_of(r):
    t = r['t']; k1 = int.from_bytes(hashlib.md5('\x01'.join(t).encode()).digest()[:8], 'little')
    k2 = int.from_bytes(hashlib.md5('\x01'.join(t[:2]).encode()).digest()[8:], 'little') if len(t) >= 2 and len(t[0]) + len(t[1]) >= 60 else None
    return k1, k2


def read(files):
    for f in files:
        with gzip.open(f, 'rt', encoding='utf8') as fh:
            for line in fh:
                r = json.loads(line); yield r


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    files = sorted(glob.glob(cfg['in'] + '/*.jsonl.gz')); os.makedirs(cfg['out'], exist_ok=True)
    # 重複除去: 検証・テストに入る会話を先に集め、 学習からは (検証・テストと同じ会話, 学習内の重複) を除く。 検証・テスト内の重複も除く。
    held = set(); seen_ho = set(); st_drop = {'train': 0, 'val': 0, 'test': 0}
    for r in read(files):
        sp = split_of(r['b'])
        if sp == 'train': continue
        k1, k2 = keys_of(r)
        if k1 in held or (k2 is not None and k2 in held): continue
        held.add(k1)
        if k2 is not None: held.add(k2)
    log('検証・テストの鍵 %d' % len(held))
    tk = ByteLevelBPETokenizer()
    def sample():
        for i, r in enumerate(read(files)):
            if i % 25 == 0 and split_of(r['b']) == 'train':
                yield '\n'.join(r['t'])
    if os.path.exists(cfg['out'] + '/tok.json'):
        tk = Tokenizer.from_file(cfg['out'] + '/tok.json')
    else:
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
    used = set()
    for i, r in enumerate(read(files)):
        if sum(len(t) for t in r['t']) > 8000: continue
        sp = split_of(r['b']); k1, k2 = keys_of(r)
        if sp == 'train':
            if k1 in held or k1 in used or (k2 is not None and (k2 in held or k2 in used)): st_drop['train'] += 1; continue
            used.add(k1)
            if k2 is not None: used.add(k2)
        else:
            if k1 in seen_ho or (k2 is not None and k2 in seen_ho): st_drop[sp] += 1; continue
            seen_ho.add(k1)
            if k2 is not None: seen_ho.add(k2)
        buf.append(r)
        if len(buf) >= 20000:
            flush()
            if (i // 20000) % 10 == 0: log('%d 会話 / 語 %s' % (i, {s: pos[s] - 8 for s in pos}))
    flush()
    for s, f in outs.items():
        f.close(); np.save('%s/%s_docs.npy' % (cfg['out'], s), np.array(docs[s], np.int64)); np.save('%s/%s_lang.npy' % (cfg['out'], s), np.array(lang[s], np.int8))
    meta = {s: dict(tokens=pos[s] - 8, docs=nd[s], dropped_dup=st_drop[s]) for s in outs}; json.dump(meta, open(cfg['out'] + '/meta.json', 'w')); log('完了 %s' % meta)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
