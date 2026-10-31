"""
エンジンの検証と実測。 (1) 正しさ: chunk を変えても (=ランの本数・併合の範囲数が変わっても) 結果が一致する。 (2) 速度と容量。 (3) データ量 vs bit/バイト (xz/gzip と並べる)。
使い方: python3 run_engine.py '{"max_bytes":150000000}'
"""
import sys, os, json, glob, hashlib, time, sysconfig, gzip, lzma, shutil
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import engine as E

def main(cfg):
    t0=time.time(); log=lambda s: print('[%5ds] %s'%(time.time()-t0,s),flush=True)
    W=cfg.get('work','/tmp/eng'); shutil.rmtree(W,ignore_errors=True); os.makedirs(W)
    std=sysconfig.get_paths()['stdlib']
    sf=sorted(f for f in glob.glob(std+'/**/*.py',recursive=True) if 'site-packages' not in f and 'lib2to3/tests/data' not in f)
    extra=sorted(glob.glob('/usr/local/lib/python3.13/dist-packages/**/*.py',recursive=True))
    rng=np.random.default_rng(0); rng.shuffle(extra)
    test_std=[f for f in sf if int(hashlib.md5(f.encode()).hexdigest(),16)%10==0]; train_std=[f for f in sf if int(hashlib.md5(f.encode()).hexdigest(),16)%10>1]
    test_ext=extra[:150]; train=train_std+extra[150:]; rng.shuffle(train)
    def rd(f):
        try:
            b=open(f,'rb').read()
            return b.decode('utf-8') if len(b)<1_000_000 else None
        except Exception: return None
    docs=[];nb=0
    for f in train:
        s=rd(f)
        if s: docs.append(s); nb+=len(s.encode())
        if nb>=cfg['max_bytes']: break
    tests={'stdlib':[s for s in map(rd,test_std) if s],'pkgs':[s for s in map(rd,test_ext) if s]}
    log('学習 %d ファイル %.1f MB'%(len(docs),nb/1e6))
    tk=Tokenizer(models.BPE()); tk.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder=decoders.ByteLevel()
    tr=trainers.BpeTrainer(vocab_size=E.VOC,special_tokens=['<s>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False)
    tk.train_from_iterator(docs[:3000],tr); log('BPE 語彙 %d'%tk.get_vocab_size())
    def enc(ds):
        out=[np.zeros(8,np.uint16)]
        for a in range(0,len(ds),200):
            for e in tk.encode_batch(ds[a:a+200]):
                out.append(np.array(e.ids,np.uint16)); out.append(np.zeros(1,np.uint16))
        return np.concatenate(out)
    T=enc(docs); T.tofile(W+'/train.tok'); ntrain=len(T)-8
    log('学習の語数 %d (%.2f バイト/語) 符号化 %.0f 秒'%(ntrain,nb/ntrain,time.time()-t0))
    TT={k:enc(v) for k,v in tests.items()}; TB={k:sum(len(s.encode()) for s in v) for k,v in tests.items()}
    ruler={}
    for k,v in tests.items():
        b='\n'.join(v).encode(); ruler[k]=dict(gzip9=len(gzip.compress(b,9))*8/len(b),xz9=len(lzma.compress(b,preset=9))*8/len(b))
    log('定規 (bit/バイト): %s'%{k:{a:round(b,3) for a,b in v.items()} for k,v in ruler.items()})
    ORD=cfg.get('orders',[1,2,3,4]); MINC={1:1,2:1,3:1,4:2}
    def build(nt,chunk,tag):
        sub=W+'/sub_%s.tok'%tag; T[:nt+8].tofile(sub); rd_=W+'/runs_'+tag
        t1=time.time(); runs=E.spill(sub,nt,rd_,ORD,chunk,lambda s:None); ts=time.time()-t1
        spill_bytes=sum(os.path.getsize(p) for n in runs for a,b in runs[n] for p in (a,b))
        t2=time.time(); pf={}
        for n in ORD:
            pf[n]=W+'/final_%s_o%d'%(tag,n); E.merge(runs[n],pf[n],MINC[n],cfg.get('range_entries',20_000_000),lambda s:None)
        tm=time.time()-t2; fin=sum(os.path.getsize(pf[n]+s) for n in ORD for s in ('_k.bin','_c.bin'))
        shutil.rmtree(rd_); os.remove(sub)
        return pf,dict(spill_s=ts,merge_s=tm,spill_bytes=spill_bytes,final_bytes=fin,nruns=len(runs[ORD[0]]))
    def score(pf):
        m=E.Model(pf); res={}
        for k,tt in TT.items():
            lp=m.logprob(tt,8,len(tt)); res[k]=-lp.sum()/(TB[k]*1.0)
        return res
    # (1) 正しさ: chunk を変える
    nt=min(ntrain,cfg.get('check_tokens',6_000_000)); outs={}
    for chunk,rng_e in ((nt,10**9),(nt//7,10**9),(nt//13,300000)):
        pf,st=build(nt,chunk,'c%d'%chunk); s=score(pf); outs[chunk]=s; log('正しさ検査 chunk=%d ラン%d本 範囲あたり<=%s: bit/バイト %s | 最終表 %.1f MB'%(chunk,st['nruns'],rng_e,{k:round(v,6) for k,v in s.items()},st['final_bytes']/1e6))
    vals=list(outs.values()); same=all(abs(v[k]-vals[0][k])<1e-9 for v in vals for k in v); log('chunk を変えても一致: %s'%same)
    # (3) データ量
    curve=[]
    for target in [3e5,1e6,3e6,1e7,3e7,ntrain]:
        if target>ntrain*1.02: break
        n_=int(target); pf,st=build(n_,cfg.get('chunk',8_000_000),'s%d'%n_); s=score(pf)
        curve.append(dict(tokens=n_,**{k:float(v) for k,v in s.items()},spill_gb=st['spill_bytes']/1e9,final_gb=st['final_bytes']/1e9,spill_s=st['spill_s'],merge_s=st['merge_s']))
        log('学習 %10d 語: bit/バイト %s | ラン %.2f GB 最終 %.2f GB | 集計 %.1f 秒 + 併合 %.1f 秒 (%.2f 百万語/秒)'%(n_,{k:round(v,3) for k,v in s.items()},st['spill_bytes']/1e9,st['final_bytes']/1e9,st['spill_s'],st['merge_s'],n_/1e6/(st['spill_s']+st['merge_s'])))
    log('RESULT_ENGINE '+json.dumps(dict(cfg=cfg,ruler=ruler,same=bool(same),curve=curve,bytes_per_token=nb/ntrain)))
if __name__=='__main__': main(json.loads(sys.argv[1]))
