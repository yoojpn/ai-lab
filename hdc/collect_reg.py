"""
npm / crates.io / Go / RubyGems から 許諾の緩い人手のコードを集める (PyPI と同じ型)。
  ランキングと許諾は ecosyste.ms (packages.ecosyste.ms) から: ダウンロード数の降順、 normalized_licenses がすべて許諾の緩いもの。
  本体は各レジストリの配布物: npm tgz / crates .crate / Go proxy の zip / gem。
  取り出す拡張子: npm .js .mjs .cjs .ts .tsx .jsx (min.js・.d.ts・1行が長いものは除外) / crates .rs / go .go / ruby .rb。 1 ファイル <= 500KB。
  出力: gzip のシャード (doc を \\x00 区切り)。 --upload で HF へ。
使い方: python3 collect_reg.py '{"reg":"npm","top":6000,"idx":0,"n":20,"out":"/tmp/reg","upload":"Youichi1/ai-lab-data"}'
"""
import os, re, io, sys, gzip, json, time, zipfile, tarfile, hashlib, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

UA = {'User-Agent': 'ai-lab-research/0.1 (youichi0929@outlook.jp)'}
REG = {'npm': 'npmjs.org', 'crates': 'crates.io', 'go': 'proxy.golang.org', 'ruby': 'rubygems.org'}
EXT = {'npm': ('.js', '.mjs', '.cjs', '.ts', '.tsx', '.jsx'), 'crates': ('.rs',), 'go': ('.go',), 'ruby': ('.rb',)}
OK = {'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc', 'unlicense', '0bsd', 'zlib', 'cc0-1.0', 'bsd', 'python-2.0', 'blueoak-1.0.0', 'mit-0', 'bsl-1.0'}


def get(url, limit=80_000_000, tries=3, deadline=120):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                t_end = time.time() + deadline; ch = []; n = 0
                while True:
                    c = r.read(1 << 20)
                    if not c: break
                    ch.append(c); n += len(c)
                    if n > limit or time.time() > t_end: return None
                return b''.join(ch)
        except Exception:
            time.sleep(1 + 2 * k)
    return None


def listing(reg, page):
    b = get('https://packages.ecosyste.ms/api/v1/registries/%s/packages?per_page=100&page=%d&sort=downloads&order=desc' % (REG[reg], page), limit=300_000_000, deadline=240)
    return json.loads(b) if b else []


def permissive(p):
    ls = [l.lower() for l in (p.get('normalized_licenses') or [])]
    return bool(ls) and all(l in OK for l in ls)


def url_of(reg, p):
    n = p['name']; v = p.get('latest_release_number')
    if not v: return None
    if reg == 'npm':
        base = n.split('/')[-1]; return 'https://registry.npmjs.org/%s/-/%s-%s.tgz' % (n, base, v)
    if reg == 'crates':
        return 'https://static.crates.io/crates/%s/%s-%s.crate' % (n, n, v)
    if reg == 'go':
        return 'https://proxy.golang.org/%s/@v/%s.zip' % (re.sub(r'[A-Z]', lambda m: '!' + m.group(0).lower(), n), v if v.startswith('v') else 'v' + v)
    if reg == 'ruby':
        return 'https://rubygems.org/downloads/%s-%s.gem' % (n, v)


def files_of(reg, blob, url):
    out = []
    ext = EXT[reg]
    def ok(name, data):
        if not name.endswith(ext) or len(data) > 500_000 or len(data) < 40: return False
        if name.endswith(('.min.js', '.d.ts', '.d.mts', '.d.cts')) or '/node_modules/' in name: return False
        try: s = data.decode('utf-8')
        except Exception: return False
        if '\x00' in s: return False
        ls = s.split('\n'); return (len(s) / max(1, len(ls))) < 120 and max(map(len, ls)) < 1000
    if reg in ('go',):
        z = zipfile.ZipFile(io.BytesIO(blob))
        for n in z.namelist():
            if n.endswith(ext) and z.getinfo(n).file_size <= 500_000:
                d = z.read(n)
                if ok(n, d): out.append(d)
    else:
        if reg == 'ruby':
            t = tarfile.open(fileobj=io.BytesIO(blob)); m = t.extractfile('data.tar.gz'); t = tarfile.open(fileobj=io.BytesIO(m.read()), mode='r:gz')
        else:
            t = tarfile.open(fileobj=io.BytesIO(blob), mode='r:*')
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(ext) and m.size <= 500_000:
                d = t.extractfile(m).read()
                if ok(m.name, d): out.append(d)
    return out


def main(cfg):
    reg = cfg['reg']; t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    pages = [p for p in range(1, (cfg['top'] + 99) // 100 + 1) if p % cfg['n'] == cfg['idx'] % cfg['n']] if 'pages' not in cfg else cfg['pages']
    pk = []
    for pg in pages:
        L = listing(reg, pg); log('ページ %d: %d 件' % (pg, len(L))); pk += [p for p in L if permissive(p)]
        time.sleep(1)
    log('許諾OK %d' % len(pk))
    os.makedirs(cfg['out'], exist_ok=True); seen = set(); st = dict(pk=0, files=0, bytes=0, fail=0)
    out = '%s/%s_%02d.bin.gz' % (cfg['out'], reg, cfg['idx']); fh = gzip.open(out, 'wb', compresslevel=3)
    def work(p):
        u = url_of(reg, p)
        if not u: return []
        b = get(u)
        if b is None: return None
        try: return files_of(reg, b, u)
        except Exception: return None
    with ThreadPoolExecutor(cfg.get('threads', 8)) as ex:
        for p, fs in zip(pk, ex.map(work, pk)):
            if fs is None: st['fail'] += 1; continue
            st['pk'] += 1
            for d in fs:
                h = hashlib.sha1(d).digest()
                if h in seen: continue
                seen.add(h); fh.write(d + b'\x00'); st['files'] += 1; st['bytes'] += len(d)
            if st['pk'] % 200 == 0: log(str(st))
    fh.close(); log('完了 %s' % st)
    if cfg.get('upload'):
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ['HF_TOKEN']); api.create_repo(cfg['upload'], repo_type='dataset', private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=out, path_in_repo='code/%s/%s' % (reg, os.path.basename(out)), repo_id=cfg['upload'], repo_type='dataset', commit_message='add code/%s' % reg)
    print('RESULT_REG ' + json.dumps(dict(reg=reg, idx=cfg['idx'], st=st)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
