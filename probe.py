"""学習した状態を調べる: (1) 取り置き文書の bit/byte、(2) 続きの生成、(3) 会話ごっこ。
使い方: python3 probe.py <state.pkl> [--chat]"""
import json, lzma, os, pickle, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import forager
main_mod = sys.modules['__main__']
for n in ('State', 'Domain', 'PPM'):
    setattr(main_mod, n, getattr(forager, n))
GROUP = {'English': ['en.wikipedia.org/wiki/Photo', 'wiki/Roman', 'wiki/Jazz', 'wiki/Black', 'simple.', 'Python_(', 'Einstein', 'Pancake'],
         }


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dist(model, ctx):
    p = np.zeros(256); rem = 1.0; excl = set()
    for k in range(min(len(ctx), model.order), -1, -1):
        e = model.tabs[k].get(ctx[len(ctx) - k:] if k else b'')
        if e is None:
            continue
        cnt = e[0]
        tot = d = 0
        for s, c in cnt.items():
            if s not in excl:
                tot += c; d += 1
        if d == 0:
            continue
        for s, c in cnt.items():
            if s not in excl:
                p[s] += rem * c / (tot + d)
        rem *= d / (tot + d)
        excl.update(cnt.keys())
    free = [s for s in range(256) if s not in excl]
    for s in free:
        p[s] += rem / len(free)
    return p / p.sum()


def generate(model, prompt, n=240, temp=0.8, seed=0):
    rng = np.random.default_rng(seed)
    ctx = bytearray(prompt.encode('utf-8'))
    out = bytearray()
    for _ in range(n):
        p = dist(model, bytes(ctx[-model.order:]))
        p = p ** (1.0 / temp); p /= p.sum()
        b = int(rng.choice(256, p=p))
        ctx.append(b); out.append(b)
    return out.decode('utf-8', 'replace')


def heldout_bpc(model):
    docs = [d for d in json.load(open(os.path.join(HERE, 'heldout.json'), encoding='utf-8')) if len(d['text']) > 500]
    rows = []
    for d in docs:
        data = d['text'].encode('utf-8')[:20000]
        bpc = model.code(data, False) / len(data)
        lz = len(lzma.compress(data, preset=6)) * 8 / len(data)
        rows.append((d['url'].split('//')[1][:48], bpc, lz))
    return rows


if __name__ == '__main__':
    st = load(sys.argv[1])
    m = st.model
    print('状態: 読んだページ=%d 場所=%d 文脈数=%d 経過=%.0f分' % (st.pages, len(st.domains), m.n_contexts(), st.elapsed / 60))
    rows = heldout_bpc(m)
    print('\n取り置き文書の bit/byte (小さいほど良い。lzma=汎用圧縮の基準)')
    for u, b, l in rows:
        print('  %-50s 学習した予測器=%.2f  lzma=%.2f' % (u, b, l))
    print('  平均: 予測器=%.3f  lzma=%.3f' % (sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows)))
    prompts = ['The history of', 'Photosynthesis is the process', 'In Python, a function is']
    print('\n--- 続きの生成 ---')
    for pr in prompts:
        print('[%s]' % pr, generate(m, pr, 200).replace('\n', ' ')[:260])
    if '--chat' in sys.argv:
        print('\n--- 会話ごっこ(この予測器は会話用には学習していません) ---')
        for q in ['Hello! How are you today?', 'What is the capital of France?', '日本の首都はどこですか?']:
            print('User:', q); print('Bot :', generate(m, 'User: %s\nAssistant:' % q, 200, 0.7).replace('\n', ' ')[:260])
