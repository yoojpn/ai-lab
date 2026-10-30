"""
DLG: 会話の小さな試験。 公開の人間作成の対話データ (DailyDialog: 日常会話 約13千対話, 学習 約170万語) で、
  A  wiki だけで数えた表 / B  対話だけで数えた表 / C  wiki + 対話 で数えた表
の3つを、 対話の検証窓で混ぜる重みを決め、 対話のテスト窓で bit/語 を測る。 あわせて、 同じ問いかけへの返事を生成して読めるようにする。
  ・ ターンの区切りは語 '|' (DailyDialog の __eou__)。 語彙 V=5000 は 小文字の正規表現トークン (wiki と対話の合算の頻度上位)
  ・ 窓: 同じ対話の中の直前 SEQ=64 語 (足りない分は未知語0で左詰め)。 対話をまたがない
  ・ 指標: 全体 bit/語、 ターン先頭の語 (直前が '|') の bit/語、 ターン内の語 の bit/語
  ・ 生成: 温度0.8, 上位40語から標本抽出 (乱数固定)。 返事を3ターンぶん
使い方: LADDER_CORPUS=wt.pkl python3 exp_dlg.py '{"N":16000000,"ntune":4000,"ntest":20000}'
"""
import os, sys, json, math, time, re, io, zipfile, urllib.request, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
import numpy as np, torch
import torch.nn.functional as Fn
import exp_cb as B
import exp_cb2 as C2
from exp_cb import V, SEQ, LN2, log

TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
SEP = '|'
BASE = 'https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/'
PROMPTS = ["hello , how are you doing today ?", "excuse me , could you tell me how to get to the station ?",
           "what time does the movie start ?", "i'm so tired . i didn't sleep well last night .",
           "do you want to have dinner with me tonight ?"]


def load_dialogs():
    out = {}
    for sp in ('train', 'validation', 'test'):
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(BASE + sp + '.zip').read()))
        nm = [n for n in z.namelist() if n.endswith('dialogues_%s.txt' % sp)][0]
        lines = z.read(nm).decode('utf8').splitlines()
        out[sp] = [TOK.findall(l.lower().replace('__eou__', ' | ')) for l in lines if l.strip()]
    return out


def windows(dl, vocab, n, seed):
    ids = [np.array([vocab.get(t, 0) for t in d], np.int64) for d in dl]
    pos = [(a, i) for a, d in enumerate(ids) for i in range(1, len(d))]
    rng = np.random.default_rng(seed); pick = rng.choice(len(pos), min(n, len(pos)), replace=False)
    X = np.zeros((len(pick), SEQ), np.int64); y = np.zeros(len(pick), np.int64)
    for r, p in enumerate(pick):
        a, i = pos[p]; c = ids[a][max(0, i - SEQ):i]; X[r, SEQ - len(c):] = c; y[r] = ids[a][i]
    return torch.from_numpy(X), torch.from_numpy(y)


def build_model(x, beta=1280.0):
    G = B.Model()
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    G.pu = torch.from_numpy(((cnt + 1) / (cnt.sum() + V)).astype(np.float32)); G.logpu = torch.log(G.pu)
    C = B.build_counts(x)
    G.C1 = C[0].clone(); G.rs1 = G.C1.sum(1); G.n1 = (G.C1 > 0).sum(1).float(); G.tri = B.build_tri(x)
    G.pmF = [B.pmi_from_counts(Ck, G.pu, beta) for Ck in C]
    ppmi = torch.clamp(G.pmF[0] + G.pmF[1], min=0)
    U, S, Vv = torch.svd_lowrank(ppmi, q=80, niter=3)
    E = U[:, :64] * S[:64].sqrt()[None, :]; E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    G.cls = B.kmeans(E, C2.K); del C, ppmi
    cls_np = G.cls.numpy(); G.sp = {}
    for nm, spec in C2.SPECS.items():
        G.sp[nm] = C2.build_sparse(x, cls_np, spec, C2.MINCNT.get(nm, 1))
    return G


def evaluate(name, x, Xva, yva, Xte, yte, itos, sepid):
    t0 = time.time(); G = build_model(x); log('[%s] 表の構築 %.0f 秒 (学習 %d 語)' % (name, time.time() - t0, len(x)))
    Fv, gv = C2.collect(Xva, G)
    bF = ['bF_%d' % k for k in range(8)]; ci = ['ci_%d_%d_%s' % c_ for c_ in C2.CIRC]
    allx = ['uni'] + bF + ['tri'] + ci + ['g4', 'g5', 's13', 's23', 's14', 'c123', 'wc', 'cc4', 'cache16', 'cache64']
    w, D, b = C2.fit(Fv, allx, yva, gate=gv, gated=True); del Fv
    log('[%s] 調整窓 %.4f bit/語' % (name, b))
    tot = 0.0; ce_all = []
    for a in range(0, len(yte), 250):
        Fc, gc = C2.all_feats(Xte[a:a + 250], G)
        with torch.no_grad():
            ce_all.append(Fn.cross_entropy(C2.logits(Fc, allx, w, gc, D), yte[a:a + 250], reduction='none') / LN2)
    ce = torch.cat(ce_all); first = Xte[:, -1] == sepid
    res = dict(name=name, words=int(len(x)), tune=b, test=ce.mean().item(), turn_first=ce[first].mean().item(), turn_inner=ce[~first].mean().item(),
               frac_first=first.float().mean().item())
    log('[%s] テスト %.4f bit/語 (ターン先頭 %.4f, ターン内 %.4f, 先頭の割合 %.3f)' % (name, res['test'], res['turn_first'], res['turn_inner'], res['frac_first']))
    torch.manual_seed(0); samples = []
    for pr in PROMPTS:
        toks = [vocab_get(itos, t) for t in TOK.findall(pr.lower())] + [sepid]
        seen_sep = 0; out = []
        for _ in range(120):
            c = toks[-SEQ:]; X1 = torch.zeros(1, SEQ, dtype=torch.long); X1[0, SEQ - len(c):] = torch.tensor(c)
            with torch.no_grad():
                Fc, gc = C2.all_feats(X1, G); z = C2.logits(Fc, allx, w, gc, D)[0]
            z[0] = -1e9; vals, idx = torch.topk(z / 0.8, 40); nxt = int(idx[torch.multinomial(torch.softmax(vals, 0), 1)])
            toks.append(nxt); out.append(itos[nxt])
            if nxt == sepid:
                seen_sep += 1
                if seen_sep >= 3:
                    break
        samples.append(dict(prompt=pr, reply=' '.join(out)))
        log('[%s] 問い: %s\n    返事: %s' % (name, pr, ' '.join(out)))
    res['samples'] = samples; return res


def vocab_get(itos, t):
    return STOI.get(t, 0)


STOI = {}


def main(cfg):
    global STOI
    torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(0)
    N = cfg['N']; ntune = cfg.get('ntune', 4000); ntest = cfg.get('ntest', 20000)
    corp = pickle.load(open(os.environ['LADDER_CORPUS'], 'rb'))
    wtok = []
    for m in corp[2]:
        wtok.extend(m)
        if len(wtok) >= N:
            break
    wtok = wtok[:N]
    dl = load_dialogs(); log('対話 学習 %d / 検証 %d / テスト %d 件' % (len(dl['train']), len(dl['validation']), len(dl['test'])))
    cnt = Counter(wtok); cnt.update(t for d in dl['train'] for t in d); cnt.pop(SEP, None)
    top = [w for w, _ in cnt.most_common(V - 2)]
    itos = ['<unk>', SEP] + top; STOI = {w: i for i, w in enumerate(itos)}; sepid = 1
    xw = np.array([STOI.get(t, 0) for t in wtok], np.int64)
    xd = np.array([STOI.get(t, 0) for d in dl['train'] for t in d], np.int64)
    log('wiki %d 語, 対話 %d 語, 対話の未知語率 %.3f' % (len(xw), len(xd), float((xd == 0).mean())))
    Xva, yva = windows(dl['validation'], STOI, ntune, 1); Xte, yte = windows(dl['test'], STOI, ntest, 2)
    log('調整窓 %d, テスト窓 %d' % (len(yva), len(yte)))
    out = []
    for name, x in (('B_dialog_only', xd), ('A_wiki_only', xw), ('C_wiki_plus_dialog', np.concatenate([xw, xd]))):
        out.append(evaluate(name, x, Xva, yva, Xte, yte, itos, sepid))
    log('--- 結果 (対話テスト窓, bit/語) ---')
    for r in out:
        log('%-20s 学習 %9d 語: 全体 %.4f, ターン先頭 %.4f, ターン内 %.4f' % (r['name'], r['words'], r['test'], r['turn_first'], r['turn_inner']))
    log('RESULT_DLG ' + json.dumps(dict(cfg=cfg, res=out)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
