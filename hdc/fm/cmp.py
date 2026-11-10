import sys, numpy as np
sys.path.insert(0,'/home/claude/ai-lab/hdc'); import engine as E
D='/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad/ia_w'; T=np.fromfile(D+'/sub.bin',dtype=np.uint16)
P=np.fromfile('pos.bin',dtype=np.uint32).astype(np.int64); out=np.fromfile('out.bin',dtype=np.int64).reshape(len(P),7,3); ORD=(1,2,3,4,5,6,8); y=T[P].astype(np.uint64)
for oi,n in enumerate(ORD):
    k=np.fromfile('%s/o%d_k.bin'%(D,n),dtype=np.uint64); c=np.fromfile('%s/o%d_c.bin'%(D,n),dtype=np.uint32); cs=np.concatenate([[0],np.cumsum(c.astype(np.int64))])
    h=E.ctx_hash(T,P,n-1,n); lo=np.searchsorted(k,h<<np.uint64(16)); hi=np.searchsorted(k,(h+np.uint64(1))<<np.uint64(16)); N=cs[hi]-cs[lo]; Tn=hi-lo
    key=(h<<np.uint64(16))|y; j=np.minimum(np.searchsorted(k,key),len(k)-1); cw=np.where(k[j]==key,c[j],0)
    f=out[:,oi,:]; print('次数 %d: N 一致 %.4f (差の最大 %d) | 異なる語数 一致 %.4f | 正解語の回数 一致 %.4f' % (n,(f[:,0]==N).mean(),np.abs(f[:,0]-N).max(),(f[:,1]==Tn).mean(),(f[:,2]==cw).mean()))
print('--- 不一致の原因調べ (次数 8) ---')
oi=6; n=8; k=np.fromfile('%s/o8_k.bin'%D,dtype=np.uint64); c=np.fromfile('%s/o8_c.bin'%D,dtype=np.uint32); cs=np.concatenate([[0],np.cumsum(c.astype(np.int64))])
h=E.ctx_hash(T,P,7,8); lo=np.searchsorted(k,h<<np.uint64(16)); hi=np.searchsorted(k,(h+np.uint64(1))<<np.uint64(16)); N=cs[hi]-cs[lo]; bad=np.flatnonzero(out[:,oi,0]!=N)
print('不一致', len(bad), '/', len(P)); 
for b in bad[:8]:
    p=P[b]; print('p=%d 文脈=%s 次=%d FM_N=%d 表_N=%d'%(p, T[p-7:p].tolist(), T[p], out[b,oi,0], N[b]))
print('不一致のうち 文脈に 0 を含む: %d, 次が 0: %d' % (sum((T[P[b]-7:P[b]]==0).any() for b in bad), sum(T[P[b]]==0 for b in bad)))
print('全体で 文脈に 0 を含む割合 %.3f' % np.mean([(T[p-7:p]==0).any() for p in P]))
