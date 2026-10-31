"""
GitHub から 許諾の緩い人手のコードを集める (Actions の中で動かす。 GITHUB_TOKEN を使う。 セッションのプロキシでは動かない)。
  1. 検索 API: language:<L> license:<SPDX> stars:<範囲> を 星の範囲ごとに引く (1 クエリ最大 1000 件)。 シャードは (言語 x 許諾 x 星の範囲) の組を idx::n で分担。
  2. 各リポジトリの tarball を取り、 対象の拡張子だけを取り出す (500KB 以下、 1 行 1000 字未満、 重複排除、 vendor/third_party/min は除く)。
  3. gzip シャード (doc を \\x00 区切り) を HF の code/gh_<言語>/ へ、 1 ジョブ 1 コミットで。
使い方: python3 collect_gh.py '{"langs":["C","C++"],"idx":0,"n":10,"out":"/tmp/gh","upload":"Youichi1/ai-lab-data","per_cell":300}'
"""
import os, re, io, sys, gzip, json, time, tarfile, hashlib, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

TOK = os.environ.get('GITHUB_TOKEN', '')
H = {'Authorization': 'Bearer ' + TOK, 'Accept': 'application/vnd.github+json', 'User-Agent': 'ai-lab-research'}
EXT = {'C': ('.c', '.h'), 'C++': ('.cc', '.cpp', '.cxx', '.hpp', '.hh', '.h', '.c'), 'Go': ('.go',), 'Java': ('.java',), 'C#': ('.cs',), 'PHP': ('.php',), 'TypeScript': ('.ts', '.tsx'), 'JavaScript': ('.js', '.mjs'), 'Rust': ('.rs',), 'Ruby': ('.rb',), 'Shell': ('.sh',), 'Kotlin': ('.kt',), 'Swift': ('.swift',)}
LIC = ['mit', 'apache-2.0', 'bsd-3-clause', 'bsd-2-clause', 'isc', 'unlicense']
STARS = ['>=20000', '10000..19999', '5000..9999', '3000..4999', '2000..2999', '1200..1999', '800..1199', '500..799', '350..499', '250..349', '180..249', '120..179', '80..119', '50..79', '30..49', '20..29']
SKIP = ('/vendor/', '/third_party/', '/thirdparty/', '/node_modules/', '/external/', '/deps/', '/test/data/', '.min.js', '.pb.go', '.pb.cc', '.pb.h', '_generated', '/generated/')


def api(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60) as r:
                return r.read(), r.headers
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                w = int(e.headers.get('Retry-After') or 0) or max(1, int(e.headers.get('X-RateLimit-Reset', time.time() + 30)) - int(time.time()))
                time.sleep(min(90, w + 1)); continue
            if e.code in (404, 451, 409): return None, None
            time.sleep(2 + 3 * k)
        except Exception:
            time.sleep(2 + 3 * k)
    return None, None


def search(lang, lic, stars, pages=10):
    out = []
    q = urllib.parse.quote('language:"%s" license:%s stars:%s fork:false archived:false' % (lang, lic, stars))
    for p in range(1, pages + 1):
        b, _ = api('https://api.github.com/search/repositories?q=%s&sort=stars&order=desc&per_page=100&page=%d' % (q, p))
        time.sleep(2.2)                                              # 検索は毎分 30 回まで
        if not b: break
        items = json.loads(b).get('items', [])
        out += [(i['full_name'], i['size']) for i in items]
        if len(items) < 100: break
    return out


def files_of(blob, ext):
    out = []
    t = tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz')
    for m in t.getmembers():
        if not (m.isfile() and m.name.endswith(ext) and 40 <= m.size <= 500_000): continue
        n = '/' + m.name
        if any(s in n for s in SKIP): continue
        d = t.extractfile(m).read()
        try: s = d.decode('utf-8')
        except Exception: continue
        if '\x00' in s: continue
        ls = s.split('\n')
        if len(s) / max(1, len(ls)) >= 120 or max(map(len, ls)) >= 1000: continue
        out.append(d)
    return out


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    cells = [(l, lic, st) for l in cfg['langs'] for lic in LIC for st in STARS]
    mine = cells[cfg['idx']::cfg['n']]; log('担当セル %d / %d' % (len(mine), len(cells)))
    repos = {}
    for (l, lic, st) in mine:
        for name, size in search(l, lic, st, cfg.get('pages', 10))[:cfg.get('per_cell', 1000)]:
            if size <= cfg.get('max_kb', 150000): repos.setdefault(name, l)
    log('リポジトリ %d' % len(repos))
    os.makedirs(cfg['out'], exist_ok=True); stt = {}
    def work(item):
        name, l = item
        b, _ = api('https://api.github.com/repos/%s/tarball' % name)
        if not b: return l, None
        try: return l, files_of(b, EXT[l])
        except Exception: return l, None
    fh = {}; seen = set(); done = 0
    with ThreadPoolExecutor(cfg.get('threads', 6)) as ex:
        for (l, fs) in ex.map(work, list(repos.items())):
            done += 1; s = stt.setdefault(l, dict(repos=0, files=0, bytes=0, fail=0))
            if fs is None: s['fail'] += 1; continue
            s['repos'] += 1
            if l not in fh: fh[l] = gzip.open('%s/gh_%s_%02d.bin.gz' % (cfg['out'], l.replace('+', 'p').replace('#', 's'), cfg['idx']), 'wb', compresslevel=3)
            for d in fs:
                h = hashlib.sha1(d).digest()
                if h in seen: continue
                seen.add(h); fh[l].write(d + b'\x00'); s['files'] += 1; s['bytes'] += len(d)
            if done % 100 == 0: log('%d/%d %s' % (done, len(repos), stt))
    for f in fh.values(): f.close()
    log('完了 %s' % stt)
    if cfg.get('upload') and fh:
        from huggingface_hub import HfApi
        api_ = HfApi(token=os.environ['HF_TOKEN']); api_.create_repo(cfg['upload'], repo_type='dataset', private=True, exist_ok=True)
        for k in range(8):
            try:
                api_.upload_folder(folder_path=cfg['out'], path_in_repo='code/gh', repo_id=cfg['upload'], repo_type='dataset', allow_patterns=['*.bin.gz'], commit_message='add code/gh shard %d' % cfg['idx']); log('uploaded'); break
            except Exception as ex:
                log('upload 再試行 %d: %r' % (k, str(ex)[:120])); time.sleep(60 * (k + 1))
    print('RESULT_GH ' + json.dumps(dict(idx=cfg['idx'], st=stt)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
