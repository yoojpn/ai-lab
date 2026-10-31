"""
G6c: データ量を変えたときの bit/語。 同じ3-gram(Witten-Bell)、 同じテストファイル(標準ライブラリの保持分)、 語彙は最大の学習集合から固定(2万)。
学習: 標準ライブラリの学習分 + インストール済みパッケージのコード (ローカル実験のみ。 公開モデルには使わない)。
使い方: python3 exp_g6_scale.py '{"max_tokens":25000000}'
"""
import sys, json, glob, hashlib, time, sysconfig, tokenize, io
import numpy as np
from collections import Counter
from exp_g5 import NGram
from exp_g6 import toks_of

def main(cfg):
    t0=time.time(); log=lambda s: print('[%5ds] %s'%(time.time()-t0,s),flush=True)
    std=sysconfig.get_paths()['stdlib']
    sf=sorted(f for f in glob.glob(std+'/**/*.py',recursive=True) if 'site-packages' not in f and 'lib2to3/tests/data' not in f)
    test=[];train=[]
    for f in sf:
        h=int(hashlib.md5(f.encode()).hexdigest(),16)%10
        (test if h==0 else train if h>1 else []).append(f)
    extra=sorted(glob.glob('/usr/local/lib/python3.13/dist-packages/**/*.py',recursive=True))
    rng=np.random.default_rng(0); rng.shuffle(extra); rng.shuffle(train)
    if cfg.get('domain')=='extra':
        test=extra[:150]; extra=extra[150:]; train=[]
    docs=[];n=0
    for f in train+extra:
        tk=toks_of(f)
        if not tk or len(tk)<50: continue
        docs.append(tk); n+=len(tk)
        if n>=cfg['max_tokens']: break
    tdocs=[tk for tk in (toks_of(f) for f in test) if tk and len(tk)>=50]
    log('学習 %d ファイル %d 語, テスト %d ファイル %d 語'%(len(docs),n,len(tdocs),sum(map(len,tdocs))))
    cnt=Counter(t for d in docs for t in d)
    vocab={t:k+1 for k,(t,_) in enumerate([x for x in cnt.most_common(20000) if x[1]>=2])}; V=len(vocab)+2
    enc=lambda ds:(np.array([vocab.get(t,0) for d in ds for t in d],np.int64), np.array([s for s,d in zip(np.cumsum([0]+[len(d) for d in ds[:-1]]),ds) for _ in d],np.int64))
    Tt,St=enc(tdocs); log('テストの未知語率 %.3f'%(Tt==0).mean())
    res=[]
    for target in [3e5,1e6,3e6,1e7,2.5e7]:
        if target>n*1.05: break
        k=0;c=0
        while c<target and k<len(docs): c+=len(docs[k]);k+=1
        T,S=enc(docs[:k]); lm=NGram(V+1).fit(T,S)
        P=lm.prob(Tt,St)[0]; b=float(-np.log2(P).mean()); known=Tt!=0; bk=float(-np.log2(P[known]).mean())
        res.append(dict(train_tokens=int(len(T)),bits=b,bits_known=bk)); log('学習 %9d 語: テスト %.4f bit/語 (未知語を除く %.4f)'%(len(T),b,bk))
    log('RESULT_G6C '+json.dumps(res))
if __name__=='__main__': main(json.loads(sys.argv[1]))
