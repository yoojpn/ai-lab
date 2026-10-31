"""
会話の取り出し (公開の著作権切れの小説から)。  Gutenberg Dialogue Dataset と同じ考え方:
  1. 段落ごとに、 引用符 (言語ごと) の中身を 発言として取り出す (地の文は捨てる)
  2. 発言を含む段落が続く間を 1 つの会話とする。 地の文が 150 文字 (日本語は 100 文字) を超えたら会話を区切る
  3. 100 語を超える発言は捨てて会話を区切る。 発言が 2 つ未満の会話は捨てる
入力: HF の manu/project_gutenberg (en,de,fr,es,it,nl,pt,sv,zh,pl), globis-university/aozorabunko-clean (ja)
出力: jsonl.gz  {"l":言語, "b":本のID, "t":[発言,...]}   を HF の保管先へ
使い方: python3 collect_dialog.py '{"source":"gutenberg","idx":0,"n":20,"out":"/tmp/dlg","upload":"Youichi1/ai-lab-data"}'
        python3 collect_dialog.py '{"source":"aozora","out":"/tmp/dlg","upload":"Youichi1/ai-lab-data"}'
"""
import os, re, sys, gzip, json, time

START = re.compile(r'\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n', re.I)
END = re.compile(r'\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG', re.I)
QUOTES = {
    'default': [re.compile(r'“([^”]{1,800})”'), re.compile(r'"([^"]{1,800})"')],
    'de': [re.compile(r'„([^“”]{1,800})[“”]'), re.compile(r'»([^«]{1,800})«'), re.compile(r'“([^”]{1,800})”'), re.compile(r'"([^"]{1,800})"')],
    'fr': [re.compile(r'«\s?([^»]{1,800})\s?»')],
    'es': [re.compile(r'«\s?([^»]{1,800})\s?»'), re.compile(r'“([^”]{1,800})”')],
    'it': [re.compile(r'«\s?([^»]{1,800})\s?»'), re.compile(r'“([^”]{1,800})”'), re.compile(r'"([^"]{1,800})"')],
    'pt': [re.compile(r'«\s?([^»]{1,800})\s?»'), re.compile(r'“([^”]{1,800})”')],
    'zh': [re.compile(r'“([^”]{1,300})”'), re.compile(r'「([^」]{1,300})」')],
    'ja': [re.compile(r'「([^」]{1,300})」'), re.compile(r'『([^』]{1,300})』')],
}
DASH = re.compile(r'^\s*[—–―-]{1,2}\s*(.+)$')
DASH_LANGS = {'fr', 'es', 'pt', 'it'}


def strip_gutenberg(t):
    m = START.search(t)
    if m:
        t = t[m.end():]
    m = END.search(t)
    if m:
        t = t[:m.start()]
    return t


def paragraphs(t, lang):
    if lang == 'ja':
        return [l.strip('　 \t') for l in t.split('\n') if l.strip('　 \t')]
    return [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', t) if p.strip()]


def turn_of(p, lang):
    pats = QUOTES.get(lang, QUOTES['default'])
    for pat in pats:
        sp = pat.findall(p)
        if sp:
            return ' '.join(s.strip() for s in sp)
    if lang in DASH_LANGS:
        m = DASH.match(p)
        if m:
            return m.group(1).strip()
    return None


def extract(text, lang, nar_limit=None, max_words=100):
    nar_limit = nar_limit or (100 if lang in ('ja', 'zh') else 150)
    out = []; cur = []; nar = 0
    def close():
        nonlocal cur, nar
        if len(cur) >= 2:
            out.append(cur)
        cur = []; nar = 0
    for p in paragraphs(text, lang):
        t = turn_of(p, lang)
        if t:
            n = len(t) if lang in ('ja', 'zh') else len(t.split())
            if n > (max_words * 2 if lang in ('ja', 'zh') else max_words) or n == 0:
                close(); continue
            cur.append(t); nar = 0
        else:
            nar += len(p)
            if nar > nar_limit:
                close()
    close()
    return out


def gutenberg(cfg, log):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    api = HfApi(); fs = sorted(s for s in api.list_repo_files('manu/project_gutenberg', repo_type='dataset') if s.endswith('.parquet'))
    mine = fs[cfg['idx']::cfg['n']] if 'files' not in cfg else cfg['files']
    log('担当ファイル %d / %d: %s' % (len(mine), len(fs), mine[:3]))
    out = '%s/gutenberg_%02d.jsonl.gz' % (cfg['out'], cfg.get('idx', 0)); os.makedirs(cfg['out'], exist_ok=True)
    st = dict(books=0, dialogs=0, turns=0, words=0)
    with gzip.open(out, 'wt', encoding='utf8') as w:
        for f in mine:
            lang = os.path.basename(f).split('-')[0]
            p = hf_hub_download('manu/project_gutenberg', f, repo_type='dataset'); pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=64):
                for bid, text in zip(batch.column('id').to_pylist(), batch.column('text').to_pylist()):
                    st['books'] += 1
                    for d in extract(strip_gutenberg(text), lang):
                        w.write(json.dumps({'l': lang, 'b': bid, 't': d}, ensure_ascii=False) + '\n')
                        st['dialogs'] += 1; st['turns'] += len(d); st['words'] += sum(len(x) if lang in ('ja', 'zh') else len(x.split()) for x in d)
            os.remove(p)
            log('%s: %s' % (f, st))
    return out, st


def aozora(cfg, log):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('globis-university/aozorabunko-clean', 'aozorabunko-dedupe-clean.jsonl.gz', repo_type='dataset')
    out = '%s/aozora_00.jsonl.gz' % cfg['out']; os.makedirs(cfg['out'], exist_ok=True); st = dict(books=0, dialogs=0, turns=0, chars=0)
    with gzip.open(p, 'rt', encoding='utf8') as f, gzip.open(out, 'wt', encoding='utf8') as w:
        for line in f:
            r = json.loads(line); st['books'] += 1; bid = r['meta'].get('作品ID', '')
            for d in extract(r['text'], 'ja'):
                w.write(json.dumps({'l': 'ja', 'b': bid, 't': d}, ensure_ascii=False) + '\n'); st['dialogs'] += 1; st['turns'] += len(d); st['chars'] += sum(len(x) for x in d)
    log('aozora: %s' % st)
    return out, st


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    out, st = (gutenberg if cfg['source'] == 'gutenberg' else aozora)(cfg, log)
    if cfg.get('upload'):
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ['HF_TOKEN']); api.create_repo(cfg['upload'], repo_type='dataset', private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=out, path_in_repo='dialog/' + os.path.basename(out), repo_id=cfg['upload'], repo_type='dataset', commit_message='add ' + os.path.basename(out))
        log('uploaded dialog/' + os.path.basename(out))
    print('RESULT_DIALOG ' + json.dumps(dict(cfg={k: v for k, v in cfg.items() if k != 'upload'}, st=st)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
