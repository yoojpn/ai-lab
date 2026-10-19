"""WikiText-103(Wikipedia由来の標準データ)から、実験用のコーパスpickleを作る。
使い方: python3 prep_wikitext.py <訓練に使う語数> <出力pkl>
形式: {0: テスト用の記事(語のリスト)の列, 1: 検証用, 2: 訓練用}  ← exp_scale.build_corpus と同じ
"""
import pickle, re, sys, urllib.request
import pandas as pd

BASE = 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/'
FILES = {0: ['test-00000-of-00001.parquet'], 1: ['validation-00000-of-00001.parquet'],
         2: ['train-00000-of-00002.parquet', 'train-00001-of-00002.parquet']}
TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
TITLE = re.compile(r'^ = [^=].* = \n$')


def articles(lines):
    cur = []
    for ln in lines:
        if TITLE.match(ln) and cur:
            yield ''.join(cur)
            cur = []
        cur.append(ln)
    if cur:
        yield ''.join(cur)


def main():
    need = int(sys.argv[1]); out = sys.argv[2]
    corp = {}
    for split, files in FILES.items():
        mods, total = [], 0
        for f in files:
            urllib.request.urlretrieve(BASE + f, '/tmp/' + f)
            lines = pd.read_parquet('/tmp/' + f)['text'].tolist()
            for art in articles(lines):
                toks = TOK.findall(art.lower().replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.'))
                if len(toks) >= 200:
                    mods.append(toks); total += len(toks)
                if split == 2 and total >= need * 1.15:
                    break
            if split == 2 and total >= need * 1.15:
                break
        corp[split] = mods
        print('split', split, 'articles', len(mods), 'tokens', total, flush=True)
    pickle.dump(corp, open(out, 'wb'))


if __name__ == '__main__':
    main()
