import numpy as np, sys

from offline_path import *
d=np.load('/tmp/path10m.npz'); Rtr,ytr,Rv,yv,ptr_c,pv=[d[k] for k in ('Rtr','ytr','Rv','yv','ptr_c','pv')]
TRAIN='/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad/dl/tok/train.bin'
Ttr=np.memmap(TRAIN,dtype=np.uint16,mode='r'); BL=2000; nbk=10000000//BL; rng=np.random.default_rng(5); st=np.sort(rng.integers(5000,len(Ttr)-BL-10,nbk))
Tt=np.concatenate([np.asarray(Ttr[q-4100:q+BL]).astype(np.int64) for q in st]); Tv=np.fromfile(VAL,dtype=np.uint16).astype(np.int64)
# 注意: 保存時の ytr/ptr_c は 0 の語を除いた後。 同じ並びで使う
def lv_path(j):
    return Counts(keyof([Rtr[:,l] for l in range(j)]),ytr).lookup(keyof([Rv[:,l] for l in range(j)]),yv)
def lv_tok(o):
    return Counts(keyof([Tt[ptr_c-k] for k in range(1,o+1)]),ytr).lookup(keyof([Tv[pv-k] for k in range(1,o+1)]),yv)
def lv_joint(j,o):
    return Counts(keyof([Rtr[:,l] for l in range(j)]+[Tt[ptr_c-k] for k in range(1,o+1)]),ytr).lookup(keyof([Rv[:,l] for l in range(j)]+[Tv[pv-k] for k in range(1,o+1)]),yv)
bits=lambda p:-np.log2(np.maximum(p,1e-12)).mean()
P=[lv_path(j) for j in (1,2,3,4)]; lp,_=tune(P,yv); pp=wb_chain(P,yv,lam=lp)
T=[lv_tok(o) for o in (1,2,3)]; lt,_=tune(T,yv); pt=wb_chain(T,yv,lam=lt)
print('path',bits(pp),'tok',bits(pt))
for w in (0.2,0.3,0.5): print('混合 w_path=%.1f'%w,bits(w*pp+(1-w)*pt))
# 結合した住所: 経路 j 個 + 直前 1 語 / 2 語 を 段として 足す (粗い -> 細かい)
chain=[lv_tok(1),lv_tok(2)]+[lv_joint(2,1),lv_joint(4,1),lv_joint(2,2),lv_joint(4,2)]
lc,ls=tune(chain,yv); print('結合の鎖 (語1,語2,経路2+語1,経路4+語1,経路2+語2,経路4+語2): 損失',ls,'lam',lc,'被覆',[round(float(c[3].mean()),3) for c in chain])
