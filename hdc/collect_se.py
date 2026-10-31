"""
StackExchange (archive.org のダンプ, CC BY-SA) の Q&A を 会話形式に直す。
  サイトごとの 7z から Posts.xml だけを取り出し (流しながら処理, ディスクに XML を置かない)、 先頭から cap_mb MB 読んだところで打ち切る。
  質問 (点数>=1) と その回答 (点数>=1, 点数の高い順に最大2つ) を 1 つのスレッドにして jsonl.gz へ。 HTML は文章に直す (コードは残す)。
  出力: {"s":サイト,"id":質問ID,"t":[{"r":"q","x":...},{"r":"a","x":...}]}   帰属表示のため サイトと ID を残す。
使い方: python3 collect_se.py '{"sites":["japanese.stackexchange.com"],"cap_mb":120,"out":"/tmp/se","upload":"Youichi1/ai-lab-data"}'
"""
import os, re, sys, gzip, json, time, html, urllib.request, xml.etree.ElementTree as ET
import py7zr
from py7zr.io import Py7zIO, WriterFactory

BASE = 'https://archive.org/download/stackexchange/'
UA = {'User-Agent': 'ai-lab-research/0.1'}
CODE = re.compile(r'<pre[^>]*><code>(.*?)</code></pre>', re.S)
TAG = re.compile(r'<[^>]+>')


def clean(h):
    h = CODE.sub(lambda m: '\n' + html.unescape(TAG.sub('', m.group(1))) + '\n', h)
    h = re.sub(r'</(p|li|h\d|blockquote|div)>', '\n', h); h = re.sub(r'<br\s*/?>', '\n', h)
    h = html.unescape(TAG.sub('', h))
    return re.sub(r'\n{3,}', '\n\n', h).strip()


class Stop(Exception):
    pass


class Sink(Py7zIO):
    def __init__(self, cap, qs, ans):
        self.cap = cap; self.n = 0; self.qs = qs; self.ans = ans; self.p = ET.XMLPullParser(['end'])
    def write(self, s):
        self.n += len(s); self.p.feed(s)
        for ev, e in self.p.read_events():
            if e.tag != 'row': continue
            a = e.attrib
            try:
                sc = int(a.get('Score', '0'))
            except ValueError:
                sc = 0
            if sc >= 1:
                if a.get('PostTypeId') == '1':
                    self.qs[a['Id']] = (sc, a.get('Title', ''), a.get('Body', ''))
                elif a.get('PostTypeId') == '2':
                    self.ans.setdefault(a.get('ParentId'), []).append((sc, a.get('Body', '')))
            e.clear()
        if self.n >= self.cap:
            raise Stop()
        return len(s)
    def read(self, size=None): return b''
    def seek(self, offset, whence=0): return 0
    def flush(self): pass
    def size(self): return self.n


class Fac(WriterFactory):
    def __init__(self, cap, qs, ans): self.cap = cap; self.qs = qs; self.ans = ans
    def create(self, filename): return Sink(self.cap, self.qs, self.ans)


def do_site(site, cap_mb, out_dir, log):
    path = '/tmp/se_%s.7z' % site
    for k in range(3):
        try:
            urllib.request.urlretrieve(urllib.request.Request(BASE + site + '.7z', headers=UA).full_url, path) if False else None
            with urllib.request.urlopen(urllib.request.Request(BASE + site + '.7z', headers=UA), timeout=60) as r, open(path, 'wb') as f:
                while True:
                    c = r.read(1 << 22)
                    if not c: break
                    f.write(c)
            break
        except Exception as ex:
            log('download retry %s %s' % (site, ex)); time.sleep(3)
    qs = {}; ans = {}
    try:
        with py7zr.SevenZipFile(path) as z:
            z.extract(targets=['Posts.xml'], factory=Fac(int(cap_mb * 1e6), qs, ans))
    except Stop:
        pass
    os.remove(path)
    out = '%s/%s.jsonl.gz' % (out_dir, site); n = 0; chars = 0
    with gzip.open(out, 'wt', encoding='utf8') as w:
        for qid, (sc, title, body) in qs.items():
            a = sorted(ans.get(qid, []), reverse=True)[:2]
            if not a: continue
            turns = [{'r': 'q', 'x': (title + '\n' + clean(body)).strip()}] + [{'r': 'a', 'x': clean(b)} for _, b in a]
            w.write(json.dumps({'s': site, 'id': qid, 't': turns}, ensure_ascii=False) + '\n'); n += 1; chars += sum(len(t['x']) for t in turns)
    log('%s: スレッド %d, 文字 %.1f M' % (site, n, chars / 1e6))
    return out, dict(site=site, threads=n, chars=chars)


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    os.makedirs(cfg['out'], exist_ok=True); stats = []
    if 'sites' not in cfg:                                      # 自動選択: メタを読み、 本体サイト(metaなし, stackoverflow.com本体を除く)を大きい順に top 件、 idx::n を担当
        meta = json.load(urllib.request.urlopen(urllib.request.Request('https://archive.org/metadata/stackexchange', headers=UA), timeout=120))
        fs = sorted([(int(f['size']), f['name'][:-3]) for f in meta['files'] if f['name'].endswith('.7z') and not f['name'].startswith('stackoverflow.com-') and '.meta.' not in f['name'] and not f['name'].startswith('meta.')], reverse=True)
        names = [n for _, n in fs[:cfg.get('top', 120)]]
        for must in ('ja.stackoverflow.com', 'japanese.stackexchange.com', 'chinese.stackexchange.com'):
            if must not in names and any(n == must for _, n in fs): names.append(must)
        cfg['sites'] = names[cfg['idx']::cfg['n']]; log('担当 %d サイト: %s' % (len(cfg['sites']), cfg['sites']))
    for site in cfg['sites']:
        try:
            out, st = do_site(site, cfg.get('cap_mb', 120), cfg['out'], log); stats.append(st)
            if cfg.get('upload'):
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ['HF_TOKEN']); api.create_repo(cfg['upload'], repo_type='dataset', private=True, exist_ok=True)
                api.upload_file(path_or_fileobj=out, path_in_repo='qa/' + os.path.basename(out), repo_id=cfg['upload'], repo_type='dataset', commit_message='add qa/' + os.path.basename(out)); os.remove(out)
        except Exception as ex:
            log('失敗 %s: %r' % (site, ex))
    print('RESULT_SE ' + json.dumps(stats))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
