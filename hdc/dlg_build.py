"""段階 A: 会話のトークン列から 頻度・頻出語の表・語クラス・n-gram 表 (次数 1..5) を作る。
使い方: python3 dlg_build.py '{"dir":"/tmp/tok"}'"""
import sys, json, os, time, glob, shutil
import numpy as np
import engine as E, core as C

V = 32768


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    W = cfg.get('dir', '/tmp/tok'); meta = json.load(open(W + '/meta.json')); nt = meta['train']['tokens']
    T = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); log('学習 %d 語, 会話 %d' % (nt, meta['train']['docs']))
    if os.path.exists(W + '/aux.npz'): log('aux 既存')
    else:
        cnt = C.count_tokens(T, V); fm = C.make_fmap(cnt); log('頻度表')
        cls = C.learn_classes(T, V, log=log)
        np.savez(W + '/aux.npz', cnt=cnt, fm=fm, cls=cls)
    ORD = [1, 2, 3, 4, 5]; MINC = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}; rd = W + '/runs'
    runs = E.spill(W + '/train.bin', nt, rd, ORD, cfg.get('chunk', 8_000_000), log)
    for n in ORD:
        E.merge(runs[n], W + '/final_o%d' % n, MINC[n], 20_000_000, log)
        for a, b in runs[n]: os.remove(a); os.remove(b)
    shutil.rmtree(rd, ignore_errors=True)
    log('完了 最終表 %.2f GB' % (sum(os.path.getsize(f) for f in glob.glob(W + '/final_o*')) / 1e9))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
