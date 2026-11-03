"""保存した重みで 質問に答えさせる (CPUで動く)。
使い方: python3 chat_net.py 重み.pt '質問' ['質問2' ...]      (重みは HF の Youichi1/ai-lab-data の models/ にある)
  環境変数 TEMP (既定 0.7) と TOPK (既定 40) と NMAX (既定 150)
"""
import sys, os, json, numpy as np, torch
import exp_ctx as X
from tokenizers import Tokenizer


def load(path):
    ck = torch.load(path, map_location='cpu'); cfg = ck['cfg']; st = ck['state']
    X.DEV = torch.device('cpu'); X.SPARSE = False; X.CV = bool(int(cfg.get('ctxv', 0))); X.BF16 = bool(int(cfg.get('bf16', 0)))
    X.MC = cfg.get('mc', X.MC); X.W2 = cfg.get('w2', X.W2); c = cfg.get('c', 8)
    net = X.Net(cfg.get('d', 128), c, cfg['L'], cfg.get('n', 256), cfg.get('kk', 4), cfg.get('k', 4), list(cfg.get('feats', [])), st['rar'].numpy())
    net.load_state_dict(st); net.eval(); return net, cfg


def answer(net, cfg, tk, q, temp=0.7, topk=40, nmax=150):
    c = cfg.get('c', 8); feats = list(cfg.get('feats', [])); offs8 = np.arange(1, c + 1); o1 = [1 + (np.arange(X.NS) * w) // X.NS for w in X.W1]; offs2 = np.arange(c + 1, X.W2 + 1)
    T_ = np.array([0] * c + tk.encode('### 質問\n%s\n### 回答\n' % q.strip()).ids, dtype=np.int64); n0 = len(T_)
    with torch.no_grad():
        for i_ in range(nmax):
            inp = X.make_inputs(T_, np.array([len(T_)]), np.array([c]), feats, offs8, o1, offs2); pr = net.dist(inp)[0] ** (1 / temp); pr[0] = 0; tv, ti = pr.topk(topk)
            pr = torch.zeros_like(pr).scatter_(0, ti, tv); pr = pr / pr.sum(); T_ = np.r_[T_, int(torch.multinomial(pr, 1))]
            if i_ % 8 == 7 and '###' in tk.decode(T_[n0:].tolist()): break
    return tk.decode(T_[n0:].tolist()).split('###')[0].strip()


if __name__ == '__main__':
    net, cfg = load(sys.argv[1]); tk = Tokenizer.from_file('/tmp/tok/tok.json')
    for q in sys.argv[2:]:
        print('Q:', q); print('A:', answer(net, cfg, tk, q, float(os.environ.get('TEMP', 0.7)), int(os.environ.get('TOPK', 40)), int(os.environ.get('NMAX', 150)))); print()
