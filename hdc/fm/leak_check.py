import pickle, numpy as np, time
t0=time.time(); convs = pickle.load(open('/tmp/al/convs.pkl','rb')); tr=np.load('/tmp/al/tr_idx.npy'); te=np.load('/tmp/al/te_idx.npy'); print('loaded',time.time()-t0,flush=True)
def grams(S, n=8):
    S=S.astype(np.uint64); h=np.zeros(len(S)-n+1,np.uint64)
    for i in range(n): h=h*np.uint64(1000003)+S[i:len(S)-n+1+i]
    return h
trh=np.concatenate([grams(convs[i][0]) for i in tr]); print('cat',len(trh),time.time()-t0,flush=True); trh.sort(); print('sorted',time.time()-t0,flush=True)
fr=[]
for i in te:
    g=grams(convs[i][0]); j=np.clip(np.searchsorted(trh,g),0,len(trh)-1); fr.append((trh[j]==g).mean())
print('leak: test 8-gram in train(half): mean',np.mean(fr),'median',np.median(fr),'frac>0.5',np.mean(np.array(fr)>0.5),flush=True)
te_arr=np.load('/tmp/al/te_idx.npy'); fr=np.array(fr); clean=te_arr[fr<0.03]; print('clean test convs',len(clean),'of',len(te_arr), flush=True); np.save('/tmp/al/te_clean_idx.npy',clean)
