"""FM-index (複数分割) で 1 語ずつ生成する会話。 数え上げのみ (勾配なし・網なし)。
使い方: python3 fm_chat.py SHARDLIST [LAMBDA_JSON] -- 発話1 発話2 ...    (発話を引数で与える / 引数がなければ 標準入力で 対話)
各発話の後ろには 語 2 (発話の終わり) を置く。 文脈の末尾から 次数 1..24 の 回数を 分割をまたいで足し、 Witten-Bell で混ぜた 確率から サンプルする。"""
import sys, os, json, subprocess
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.environ.get('FMENG', HERE + '/fmeng'); V = 32768
ORD = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
TK = Tokenizer.from_file(os.environ.get('TOKJSON', '/tmp/tokhf/tok/tok.json'))


class FM:
    def __init__(self, shardlist, lam=None):
        self.p = subprocess.Popen([ENG, shardlist, ','.join(map(str, ORD)), 'serve'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lam = lam or [4.0] * len(ORD)

    def dist(self, ctx):
        c = [int(x) for x in ctx[-23:]]; self.p.stdin.write('D %d %s\n' % (len(c), ' '.join(map(str, c)))); self.p.stdin.flush()
        p = np.full(V, 1.0 / V)
        for oi in range(len(ORD)):
            parts = self.p.stdout.readline().split(); N = int(parts[2]); T = int(parts[3]); m = int(parts[4])
            if N > 0:
                l = self.lam[oi]; d = N + l * T; p *= (l * T) / d
                if m:
                    ws = np.array(parts[5::2], dtype=np.int64); cs = np.array(parts[6::2], dtype=np.float64); p[ws] += cs / d
        assert self.p.stdout.readline().strip() == 'E'
        return p / p.sum()


def blocked(gen, n):
    """直前の n-1 語 + 次の語 が すでに 出ているなら その 次の語を 禁止 (同じ繰り返しを避ける)"""
    out = set()
    if len(gen) >= n - 1:
        tail = tuple(gen[-(n - 1):])
        for i in range(len(gen) - n + 1):
            if tuple(gen[i:i + n - 1]) == tail: out.add(gen[i + n - 1])
    return out


def reply(fm, hist, temp=0.7, topk=40, maxlen=80, seed=0):
    rng = np.random.default_rng(seed); ctx = [0]
    for u in hist: ctx += TK.encode(u).ids + [2]
    gen = []
    for step in range(maxlen):
        P = fm.dist(ctx + gen); P[0] = 0
        if step < 2: P[2] = 0
        for b in blocked(gen, 3): P[b] = 0
        idx = np.argsort(-P)[:topk]; q = P[idx] ** (1 / temp); q /= q.sum(); x = int(rng.choice(idx, p=q))
        if x == 2: break
        gen.append(x)
    return TK.decode(gen)


if __name__ == '__main__':
    args = sys.argv[1:]; sl = args[0]; rest = args[1:]; lam = None
    if rest and rest[0].endswith('.json'): lam = json.load(open(rest[0])).get('lam'); rest = rest[1:]
    if rest and rest[0] == '--': rest = rest[1:]
    fm = FM(sl, lam)
    if rest:
        for q in rest:
            print('あなた: %s' % q)
            for s in range(3): print('   返事 %d: %s' % (s + 1, reply(fm, [q], seed=s)))
    else:
        hist = []
        for line in sys.stdin:
            q = line.strip()
            if not q: continue
            r = reply(fm, hist + [q]); print('AI: ' + r, flush=True); hist += [q, r]
