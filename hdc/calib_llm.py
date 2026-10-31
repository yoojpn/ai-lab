"""較正: 本物の小型 LLM を 同じテキストで動かして bits/byte を測り、 私たちの表(会話/Wikipedia)と並べる。
  LLM は 先頭に <|endoftext|> を付け、 文書全体を文脈にして 全語の負の対数尤度を足す。 私たちの表は <bos> を文脈に 各語を足す。 どちらも (総 bit) / (総 UTF-8 バイト)。
  会話: 区切りは LLM では改行、 私たちは <turn> 語 (その費用も含める)。 差は 区切り記号の費用ぶん (小さい) で 厳密な同一条件ではない ☆。
使い方: python3 calib_llm.py '{"models":["gpt2","HuggingFaceTB/SmolLM2-135M"],"ndlg":150,"nwiki":100}'"""
import sys, json, time
import numpy as np, torch
import engine as E
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


def ours(M, T, a, b):
    lp = M.logprob_at(T, np.arange(a, b)); return float(-lp.sum())


def main(cfg):
    t0 = time.time(); rng = np.random.default_rng(1); tk = Tokenizer.from_file('/tmp/tok/tok.json'); res = {}
    lam = lambda W: {int(k): v for k, v in json.load(open(W + '/lam.json')).items()}
    Md = E.Model({n: '/tmp/tok/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam('/tmp/tok'), kn_base=True)
    Mw = E.Model({n: '/tmp/wiki/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam=lam('/tmp/wiki'), kn_base=True)
    sets = {}
    Td = np.fromfile('/tmp/tok/test.bin', dtype=np.uint16); dd = np.load('/tmp/tok/test_docs.npy'); de = np.r_[dd[1:], len(Td)]; lg = np.load('/tmp/tok/test_lang.npy'); items = []
    for d in rng.permutation(len(dd)):
        if lg[d] != 0 or not (60 <= de[d] - dd[d] <= 400): continue
        x = Td[dd[d]:de[d]].tolist(); turns = []; cur = []
        for t in x[1:]:
            if t in (1, 2): turns.append(tk.decode(cur)); cur = []
            else: cur.append(t)
        items.append(dict(text='\n'.join(turns), bits=ours(Md, Td, dd[d] + 1, de[d]))); 
        if len(items) >= cfg['ndlg']: break
    sets['会話(英語)'] = items
    Tw = np.fromfile('/tmp/wiki/test.bin', dtype=np.uint16); wd = np.load('/tmp/wiki/test_docs.npy'); we = np.r_[wd[1:], len(Tw)]; wl = np.load('/tmp/wiki/test_lang.npy'); items = []
    for d in rng.permutation(len(wd)):
        if wl[d] != 0 or we[d] - wd[d] < 520: continue
        a = wd[d] + 1; b = a + 400; items.append(dict(text=tk.decode(Tw[a:b].tolist()), bits=ours(Mw, Tw, a, b)))
        if len(items) >= cfg['nwiki']: break
    sets['Wikipedia(英語)'] = items
    for nm, its in sets.items():
        nb = sum(len(i['text'].encode()) for i in its); res[nm] = dict(n_docs=len(its), bytes=nb, ours_bpb=sum(i['bits'] for i in its) / nb)
        print(nm, 'ours bits/byte %.4f (%d docs, %d bytes)' % (res[nm]['ours_bpb'], len(its), nb), flush=True)
    for mn in cfg['models']:
        tkz = AutoTokenizer.from_pretrained(mn); lm = AutoModelForCausalLM.from_pretrained(mn, torch_dtype=torch.float32).eval(); bos = tkz.bos_token_id if tkz.bos_token_id is not None else tkz.eos_token_id
        for nm, its in sets.items():
            tot = 0.0
            for i in its:
                ids = [bos] + tkz(i['text'])['input_ids'][:1000]; x = torch.tensor([ids])
                with torch.no_grad(): lg_ = lm(x).logits[0, :-1].float()
                tot += float(torch.nn.functional.cross_entropy(lg_, x[0, 1:], reduction='sum')) / np.log(2)
            res[nm][mn] = tot / res[nm]['bytes']; print(mn, nm, 'bits/byte %.4f   (差 ours-LLM = %.3f bit/byte, 比 %.2f 倍)' % (res[nm][mn], res[nm]['ours_bpb'] - res[nm][mn], res[nm]['ours_bpb'] / res[nm][mn]), flush=True)
    json.dump(res, open('/tmp/calib_llm.json', 'w'), ensure_ascii=False, indent=1); print('時間 %.0f 秒' % (time.time() - t0))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
