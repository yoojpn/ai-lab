import sys, pickle, json, numpy as np, itertools
sys.path.insert(0,'.'); sys.path.insert(0,'..')
import align_copy as A, chat2 as C
from chat3 import Chat3
from tokenizers import Tokenizer
tk = Tokenizer.from_file('/tmp/tokhf/tok/tok.json'); al = pickle.load(open('/tmp/al/al_jaen.pkl','rb')); A.D = len(al.S)
W24 = '/tmp/bo24b'; Em = np.load(W24+'/E.npy'); lam = np.load(W24+'/combo_lam.npy')[:9]; uni = np.bincount(np.fromfile(W24+'/sub.bin',dtype=np.uint16)[8:],minlength=32768).astype(float)+1
db = C.local_dist(W24, None); ch = Chat3(db, Em, uni, lam, np.load(W24+'/cls64.npy'), tk, al, None)
T = json.load(open('icl_tasks.json')); T = [t for i, t in enumerate(T) if i % 3 == 0]; V = 32768
def fabs(ids, he):
    j, sc = al.cand_scores(np.array(ids), he); F = np.zeros(V); np.add.at(F, np.array(ids)[j], np.exp(sc)); return np.log1p(F)
CFG = [(0, 0), (0.3, 0), (0, 0.5), (0, 1.0), (0.3, 0.5), (0.3, 1.0)]
res = {c: {} for c in CFG}
for t in T:
    ids = [0] + tk.encode(t['prompt']).ids; he = len(ids); a = t['answer'].strip()
    for cfg in CFG:
        g = []
        for _ in range(6):
            c = ids + g; P = db(c).astype(np.float64); s = np.zeros(V)
            if cfg[0]: s += cfg[0] * ch.score(c)
            if cfg[1]: s += cfg[1] * fabs(c, he)
            e = P * np.exp(s - s.max()); e[0] = 0; g.append(int(np.argmax(e)))
        ok = tk.decode(g).strip().startswith(a); r = res[cfg].setdefault(t['task'], [0, 0]); r[0] += ok; r[1] += 1
for cfg in CFG: print('doc %.1f align_abs %.1f' % cfg, {k: '%d/%d' % tuple(v) for k, v in res[cfg].items()}, flush=True)
