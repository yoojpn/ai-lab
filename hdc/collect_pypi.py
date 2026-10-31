"""
PyPI のパッケージから、 許諾の緩い (MIT/BSD/Apache/ISC/PSF/Unlicense) 人手のPythonコードを集める。
  - 上位リスト (ダウンロード数) から標本または範囲を取り、 メタデータの license / classifiers で判定 (GPL系・不明は除外)
  - 純Pythonのwheelを優先、 なければ sdist。 .py (<=1MB) を取り出し、 内容のハッシュで重複を除く
  - 出力: シャード (docを \x00 区切りの utf-8) と 統計 jsonl
使い方: python3 collect_pypi.py '{"sample":300,"out":"/tmp/pypi"}'   /  '{"start":0,"end":2000,"out":"/tmp/pypi"}'
"""
import sys, os, re, io, gzip, json, time, zipfile, tarfile, hashlib, random, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

OK = re.compile(r'\b(MIT|BSD|Apache|ISC|Python Software Foundation|PSF|Unlicense|Public Domain|CC0|Zlib)\b', re.I)
BAD = re.compile(r'GPL|AGPL|LGPL|GNU|Copyleft|Proprietary|Commercial|All rights reserved', re.I)
UA = {'User-Agent': 'ai-lab-research/0.1 (youichi0929@outlook.jp)'}


def get(url, limit=None, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                b = r.read(limit + 1 if limit else -1)
                if limit and len(b) > limit:
                    return None
                return b
        except Exception:
            time.sleep(1 + k)
    return None


def permissive(info):
    txt = ' '.join([info.get('license') or '', info.get('license_expression') or ''] + [c for c in info.get('classifiers', []) if c.startswith('License')])
    if len(txt) > 3000:
        txt = txt[:300]                                    # 本文が入っている場合は先頭だけ
    if BAD.search(txt) and not re.search(r'MIT|BSD|Apache', txt):
        return False, txt[:80]
    if BAD.search(txt) and re.search(r'GPL|GNU', txt):
        return False, txt[:80]
    return bool(OK.search(txt)), txt[:80]


def pick(urls):
    w = [u for u in urls if u['packagetype'] == 'bdist_wheel' and u['filename'].endswith('none-any.whl')]
    if w:
        return w[0]
    s = [u for u in urls if u['packagetype'] == 'sdist']
    if s:
        return s[0]
    w = [u for u in urls if u['packagetype'] == 'bdist_wheel']
    return w[0] if w else None


def pyfiles(blob, fname):
    out = []
    if fname.endswith('.whl') or fname.endswith('.zip'):
        z = zipfile.ZipFile(io.BytesIO(blob))
        for n in z.namelist():
            if n.endswith('.py'):
                i = z.getinfo(n)
                if i.file_size <= 1_000_000:
                    out.append(z.read(n))
    else:
        t = tarfile.open(fileobj=io.BytesIO(blob), mode='r:*')
        for m in t.getmembers():
            if m.isfile() and m.name.endswith('.py') and m.size <= 1_000_000:
                out.append(t.extractfile(m).read())
    return out


def main(cfg):
    rows = json.load(open('/tmp/top.json'))['rows']; names = [r['project'] for r in rows]
    if 'sample' in cfg:
        random.seed(0); idx = sorted(random.sample(range(len(names)), cfg['sample']))
    else:
        idx = list(range(cfg['start'], min(cfg['end'], len(names))))
    out = cfg['out']; os.makedirs(out, exist_ok=True)
    seen = set(); lock = threading.Lock(); shard = [0]; shard_bytes = [0]; fh = [gzip.open('%s/shard_%04d.bin.gz' % (out, 0), 'wb', compresslevel=3)]
    stats = open(out + '/stats.jsonl', 'a'); t0 = time.time(); agg = dict(n=0, perm=0, nodl=0, bytes=0, files=0, dl=0, dup=0)

    def work(i):
        name = names[i]; st = dict(rank=i, name=name)
        b = get('https://pypi.org/pypi/%s/json' % name)
        if not b:
            st['err'] = 'meta'; return st
        d = json.loads(b); ok, lic = permissive(d['info']); st['license'] = lic; st['perm'] = ok
        if not ok:
            return st
        u = pick(d['urls'])
        if not u:
            st['err'] = 'nofile'; return st
        blob = get(u['url'], limit=100_000_000)
        if blob is None:
            st['err'] = 'dl'; return st
        st['dl'] = len(blob)
        try:
            fs = pyfiles(blob, u['filename'])
        except Exception:
            st['err'] = 'extract'; return st
        nb = 0; nf = 0; dup = 0; buf = []
        for f in fs:
            try:
                s = f.decode('utf-8')
            except Exception:
                continue
            if '\x00' in s:
                continue
            h = hashlib.sha1(s.encode()).digest()
            with lock:
                if h in seen:
                    dup += 1; continue
                seen.add(h)
            buf.append(s.encode()); nb += len(s.encode()); nf += 1
        with lock:
            for s in buf:
                fh[0].write(s + b'\x00'); shard_bytes[0] += len(s) + 1
            if shard_bytes[0] > 500_000_000:
                fh[0].close(); shard[0] += 1; shard_bytes[0] = 0; fh[0] = gzip.open('%s/shard_%04d.bin.gz' % (out, shard[0]), 'wb', compresslevel=3)
        st.update(bytes=nb, files=nf, dup=dup); return st
    with ThreadPoolExecutor(cfg.get('threads', 16)) as ex:
        for k, st in enumerate(ex.map(work, idx)):
            stats.write(json.dumps(st) + '\n'); agg['n'] += 1
            agg['perm'] += bool(st.get('perm')); agg['bytes'] += st.get('bytes', 0); agg['files'] += st.get('files', 0); agg['dl'] += st.get('dl', 0); agg['dup'] += st.get('dup', 0); agg['nodl'] += 1 if st.get('err') else 0
            if (k + 1) % 100 == 0:
                print('[%4ds] %d 件: %s' % (time.time() - t0, k + 1, agg), flush=True)
    fh[0].close(); stats.close()
    el = time.time() - t0
    print('完了 %d 件 %.0f 秒: %s | 許諾OK %.1f%% | 許諾OKの平均 %.2f MB/パッケージ | 取得 %.2f MB/s' % (agg['n'], el, agg, 100 * agg['perm'] / agg['n'], agg['bytes'] / max(1, agg['perm']) / 1e6, agg['dl'] / el / 1e6))
    json.dump(dict(agg=agg, secs=el, nnames=len(names)), open(out + '/summary.json', 'w'))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
