"""文脈内学習 (ICL) の 小さな 検査集: 重みや 知識でなく 『文脈から 規則を 取る』 力を 測る。 出力: icl_tasks.json = [{task, prompt, answer}]
 kv: 擬似の 対応表 (誘導), plural: 例から 複数形, add: 2 桁の 足し算 (例 2 つ), seq: 等差数列, copy_up: 例から 大文字化"""
import json, random
R = random.Random(0)
W = "apple river stone cloud bread chair window garden silver pencil mirror bottle ladder pocket candle button carpet basket violin harbor cotton jacket hammer island pepper rocket saddle tunnel velvet wallet yellow zipper anchor bridge cactus dragon engine falcon glider hollow ivory jungle kettle lemon marble nectar orange pillow quartz ribbon sunset thunder umbrella vessel walnut".split()
T = []
for i in range(60):
    ks = R.sample(W, 4); vs = R.sample(W, 4); q = R.randrange(4)
    T.append({'task': 'kv', 'prompt': ''.join('%s : %s\n' % (k, v) for k, v in zip(ks, vs)) + '%s :' % ks[q], 'answer': ' ' + vs[q]})
PL = ["cat cats", "dog dogs", "tree trees", "book books", "car cars", "bird birds", "house houses", "road roads", "ship ships", "lake lakes", "hand hands", "door doors"]
for i in range(30):
    ex = R.sample(PL, 3); w = R.choice(W); T.append({'task': 'plural', 'prompt': ''.join(e.split()[0] + ' -> ' + e.split()[1] + '\n' for e in ex) + w + ' ->', 'answer': ' ' + w + 's'})
for i in range(60):
    a = [(R.randrange(10, 90), R.randrange(10, 90)) for _ in range(3)]; x, y = a[2]
    T.append({'task': 'add', 'prompt': ''.join('%d + %d = %d\n' % (p, q, p + q) for p, q in a[:2]) + '%d + %d =' % (x, y), 'answer': ' %d' % (x + y)})
for i in range(40):
    s = R.randrange(1, 20); d = R.randrange(2, 9); T.append({'task': 'seq', 'prompt': ' '.join(str(s + d * j) for j in range(5)) + ' ' , 'answer': str(s + d * 5)})
for i in range(30):
    ex = R.sample(W, 3); w = R.choice([x for x in W if x not in ex]); T.append({'task': 'copy_up', 'prompt': ''.join('%s => %s\n' % (e, e.upper()) for e in ex) + w + ' =>', 'answer': ' ' + w.upper()})
json.dump(T, open('/home/claude/ai-lab/hdc/fm/icl_tasks.json', 'w'), ensure_ascii=False); print(len(T))
