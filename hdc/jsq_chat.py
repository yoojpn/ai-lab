"""質問に答える会話の試作 (数え上げのみ): 知識 = JSQuAD の本文 (日本語 Wikipedia の段落), 検索 = 文字 2-gram の TF-IDF, 答え = jsq_count の始まり/終わりの鍵の数え上げ。
使い方: python3 jsq_chat.py '質問1' '質問2' ...   (初回は 学習に 5 分、 以後は保存した表を使う)"""
import sys, os, pickle, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0]] + ['20000', '1', '8'] + sys.argv[1:]         # jsq_count の引数 (N_TRAIN N_VAL) を固定
import jsq_count as J

QS = sys.argv[4:]; PK = '/tmp/claude-0/-home-claude/3f8e15fc-9ff3-58a6-9cd3-0f9e59c9962a/scratchpad/jsq/models.pkl'
if os.path.exists(PK): ms = pickle.load(open(PK, 'rb'))
else: ms = J.train(20000); pickle.dump(ms, open(PK, 'wb'))
docs = list(dict.fromkeys(list(J.tr['context']) + list(J.va['context'])))
bg = lambda s: [s[i:i + 2] for i in range(len(s) - 1)]
df = collections.Counter(); tfs = []
for d in docs:
    c = collections.Counter(bg(d)); tfs.append(c); df.update(c.keys())
N = len(docs); idf = {k: math.log(N / (1 + v)) for k, v in df.items()}


def retrieve(q, k=3):
    qb = collections.Counter(bg(q)); sc = np.zeros(N)
    for i, c in enumerate(tfs):
        s = 0.0
        for g, n in qb.items():
            if g in c: s += idf.get(g, 0) * (1 + math.log(c[g]))
        sc[i] = s / (1 + 0.15 * math.log(len(docs[i])))
    top = np.argsort(-sc)[:k]; return [(int(i), float(sc[i])) for i in top]


def ask(q):
    best = None
    for i, rs in retrieve(q):
        a, s = J.predict(ms, docs[i], q); tot = s + 0.3 * rs
        if best is None or tot > best[0]: best = (tot, a, docs[i])
    return best[1], best[2]


if __name__ == '__main__':
    for q in QS:
        a, src = ask(q); print('Q: %s\nA: %s   (根拠: %s…)\n' % (q, a, src[:60].replace('\n', ' ')))
