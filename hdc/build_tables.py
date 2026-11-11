"""作業場所に sub.bin と n-gram の表 (次数 1〜5) を作る (cache_poe.py 等の下準備)。  使い方: python3 build_tables.py <W> <WORK> <S>"""
import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
W, WORK, S = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(WORK, exist_ok=True)
Ttr = np.memmap(W + '/train.bin', dtype=np.uint16, mode='r'); nb = 40; bl = S // nb; starts = np.linspace(8, len(Ttr) - bl - 8, nb).astype(np.int64)
if not os.path.exists(WORK + '/sub.bin'): np.concatenate([np.zeros(8, np.uint16)] + [np.asarray(Ttr[s:s + bl]) for s in starts]).tofile(WORK + '/sub.bin')
n_tok = os.path.getsize(WORK + '/sub.bin') // 2 - 8
for n in (1, 2, 3, 4, 5):
    if os.path.exists(WORK + '/final_o%d_k.bin' % n): continue
    runs = E.spill(WORK + '/sub.bin', n_tok, WORK + '/runs', [n], 8_000_000, lambda s: None); E.merge(runs[n], WORK + '/final_o%d' % n, 1 if n <= 2 else 2, 20_000_000, lambda s: None)
    for f in glob.glob(WORK + '/runs/*'): os.remove(f)
print('done', n_tok)
