#!/usr/bin/env python3
"""
自律採食エージェント v0(初日版)

ネットを自分で辿り、「圧縮がどれだけ進みそうか」を頼りに読む場所を選んで、
言語の統計(バイト単位のPPM予測器)を育てる。勾配は使わない。
人間が与えるのは、種(最初のURL)とガードレールだけ。

■ ガードレール(Guard クラス。エージェントの学習側からは変更できない)
  - robots.txt を守る(取得できない・拒否されたら読まない。Crawl-delay も尊重)
  - ドメインごとに最低 min_delay 秒の間隔、1ドメインあたりの最大ページ数
  - 公開ページ(http/https, GET のみ)。ログイン・Cookie・認証情報は使わない
  - 社内/ローカルのアドレス(プライベートIP等)宛ては、リダイレクト先も含めて遮断(SSRF対策)
  - text/html, text/plain のみ。1ページの最大サイズ制限
  - <meta name="robots" content="noindex/nofollow"> を尊重
  - 保存するのは統計(文脈の数え上げ)と、取得URLの記録のみ。本文は保存しない
  - 停止スイッチ: 作業ディレクトリに STOP ファイルを置くと止まる
  - 身元の明示: User-Agent に連絡先(--contact)を必ず入れる

■ 学習の仕組み(これまでの実験の部品)
  - 採食: ドメインを腕とみなし、「確認用サンプル(取り置き)の符号長がどれだけ減ったか」=
          圧縮進捗の大きいドメインを優先する(ノイズ・既習・重複は進捗が出ないので自然に離れる)
  - 安い仕分け: 最初のページを自己圧縮してみて、ほとんど縮まないものは捨てる
  - 驚きゲート(任意): 予測が当たったバイトでは更新を省く
  - 睡眠: 記憶(文脈の表)が予算を超えたら、1回しか見ていない高次の文脈を捨てる
  - 重複除去: 内容のハッシュで同一ページを捨てる
"""
import argparse
import collections
import hashlib
import html.parser
import ipaddress
import json
import lzma
import math
import os
import pickle
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser

UA_TOKEN = 'ForagerResearchBot'
OK_CTYPES = ('text/html', 'text/plain', 'application/xhtml+xml')
SKIP_EXT = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.ico', '.pdf', '.zip', '.gz',
            '.tar', '.tgz', '.bz2', '.xz', '.7z', '.rar', '.mp3', '.mp4', '.avi', '.mov',
            '.webm', '.woff', '.woff2', '.ttf', '.exe', '.dmg', '.iso', '.bin', '.apk',
            '.css', '.js', '.json', '.xml')


# =====================================================================
# ガードレール
# =====================================================================
class Guard:
    def __init__(self, contact, allow_local=False, deny=(), max_bytes=300_000, timeout=10.0):
        self.ua = '%s/0.1 (+%s)' % (UA_TOKEN, contact)
        self.allow_local = allow_local
        self.deny = tuple(d.strip().lower() for d in deny if d.strip())
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.robots = {}
        guard = self

        class Redirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                ok, _ = guard.url_ok(newurl)
                if not ok or not guard.robots_ok(newurl):
                    raise urllib.error.URLError('blocked redirect')
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        self.opener = urllib.request.build_opener(Redirect)

    def url_ok(self, url):
        if len(url) > 400:
            return False, 'long'
        p = urllib.parse.urlsplit(url)
        if p.scheme not in ('http', 'https'):
            return False, 'scheme'
        host = (p.hostname or '').lower()
        if not host:
            return False, 'nohost'
        if '@' in p.netloc:
            return False, 'userinfo'
        try:
            port = p.port
        except ValueError:
            return False, 'port'
        if not self.allow_local and port not in (None, 80, 443):
            return False, 'port'
        if any(host == d or host.endswith('.' + d) for d in self.deny):
            return False, 'deny'
        try:
            infos = socket.getaddrinfo(host, port or (443 if p.scheme == 'https' else 80),
                                       proto=socket.IPPROTO_TCP)
        except OSError:
            return False, 'dns'
        if not self.allow_local:
            for info in infos:
                ip = ipaddress.ip_address(info[4][0].split('%')[0])
                if not ip.is_global:
                    return False, 'private-ip'
        return True, ''

    def robots_for(self, url):
        p = urllib.parse.urlsplit(url)
        key = p.scheme + '://' + p.netloc
        ent = self.robots.get(key)
        now = time.time()
        if ent and now < ent[1]:
            return ent[0]
        r = self.fetch(key + '/robots.txt', check_robots=False, max_bytes=500_000, any_ctype=True)
        if r.get('status') == 200 and r.get('body') is not None:
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(r['body'].decode('utf-8', 'replace').splitlines())
            rp.modified()
            res = rp
        elif r.get('status') in (401, 403):
            res = 'DENY'
        elif r.get('status') and 400 <= r['status'] < 500:
            res = 'ALLOW'
        else:
            res = 'DENY'
        self.robots[key] = (res, now + (600 if res == 'DENY' else 3600))
        return res

    def robots_ok(self, url):
        res = self.robots_for(url)
        if res == 'ALLOW':
            return True
        if res == 'DENY':
            return False
        return res.can_fetch(UA_TOKEN, url)

    def crawl_delay(self, url):
        res = self.robots_for(url)
        if isinstance(res, urllib.robotparser.RobotFileParser):
            d = res.crawl_delay(UA_TOKEN)
            return float(d) if d else 0.0
        return 0.0

    def fetch(self, url, check_robots=True, max_bytes=None, any_ctype=False):
        max_bytes = max_bytes or self.max_bytes
        ok, why = self.url_ok(url)
        if not ok:
            return dict(status=None, error='blocked:' + why)
        if check_robots and not self.robots_ok(url):
            return dict(status=None, error='blocked:robots')
        req = urllib.request.Request(url, headers={
            'User-Agent': self.ua,
            'Accept': 'text/html,text/plain;q=0.9',
            'Accept-Language': 'en,ja;q=0.8'})
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                ctype = resp.headers.get_content_type()
                if not any_ctype and ctype not in OK_CTYPES:
                    return dict(status=resp.status, error='ctype:' + ctype)
                body = resp.read(max_bytes + 1)[:max_bytes]
                return dict(status=resp.status, url=resp.geturl(), ctype=ctype, body=body,
                            charset=resp.headers.get_content_charset(), error=None)
        except urllib.error.HTTPError as e:
            return dict(status=e.code, error='http')
        except Exception as e:  # noqa: BLE001
            return dict(status=None, error=type(e).__name__)


# =====================================================================
# ページの解析
# =====================================================================
BLOCK = {'p', 'div', 'br', 'li', 'tr', 'pre', 'section', 'article', 'table', 'ul', 'ol',
         'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'nav', 'footer', 'header'}
SKIP = {'script', 'style', 'noscript', 'template', 'svg', 'head'}


BOILER_TAGS = {'nav', 'aside', 'footer', 'header', 'form', 'select', 'button', 'dialog'}
BOILER_RE = re.compile(r'navbox|sidebar|interlanguage|mw-panel|vector-(menu|toc|page|main|dropdown|user)|'
                       r'editsection|catlinks|printfooter|navigation|menu|breadcrumb|cookie|banner|'
                       r'\btoc\b|reference|reflist|hatnote|mw-jump|noprint|metadata|language-selector')
VOID = {'br', 'img', 'input', 'link', 'meta', 'hr', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr'}


class Page(html.parser.HTMLParser):
    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.text = []
        self.links = []
        self.skip = 0
        self.bp = None
        self.noindex = False
        self.nofollow = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in SKIP:
            self.skip += 1
        if self.bp is not None:
            if tag == self.bp[0] and tag not in VOID:
                self.bp[1] += 1
        elif tag not in VOID and tag not in ('html', 'body', 'main', 'article') and (
                tag in BOILER_TAGS or
                BOILER_RE.search(('%s %s %s' % (a.get('class') or '', a.get('id') or '', a.get('role') or '')).lower())):
            self.bp = [tag, 1]
        if tag == 'meta' and (a.get('name') or '').lower() == 'robots':
            c = (a.get('content') or '').lower()
            self.noindex = self.noindex or 'noindex' in c
            self.nofollow = self.nofollow or 'nofollow' in c
        elif tag == 'a' and a.get('href') and 'nofollow' not in (a.get('rel') or '').lower():
            self.links.append(urllib.parse.urljoin(self.base, a['href']))
        if tag in BLOCK:
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in SKIP and self.skip:
            self.skip -= 1
        if self.bp is not None and tag == self.bp[0]:
            self.bp[1] -= 1
            if self.bp[1] <= 0:
                self.bp = None
        if tag in BLOCK:
            self.text.append('\n')

    def handle_data(self, d):
        if not self.skip and self.bp is None:
            self.text.append(d)


def extract(base, body, charset, ctype):
    text = body.decode(charset or 'utf-8', 'replace')
    if ctype == 'text/plain':
        return text.strip(), [], False, False
    p = Page(base)
    try:
        p.feed(text)
        p.close()
    except Exception:  # noqa: BLE001
        pass
    t = ''.join(p.text)
    t = re.sub(r'[ \t\r\f\v]+', ' ', t)
    t = re.sub(r'\n[ \n]*\n+', '\n\n', t).strip()
    return t, p.links, p.noindex, p.nofollow


def norm_url(u):
    u, _ = urllib.parse.urldefrag(u)
    p = urllib.parse.urlsplit(u)
    if p.scheme not in ('http', 'https') or not p.netloc:
        return None
    path = p.path or '/'
    if path.lower().endswith(SKIP_EXT):
        return None
    netloc = p.netloc.lower()
    try:
        netloc.encode('ascii')
    except UnicodeEncodeError:
        try:
            netloc = netloc.encode('idna').decode('ascii')
        except UnicodeError:
            return None
    path = urllib.parse.quote(path, safe="/%:@!$&'()*+,;=~-._")
    query = urllib.parse.quote(p.query, safe="=&%+:/?@!$'()*,;~-._")
    if (p.scheme == 'http' and netloc.endswith(':80')) or \
            (p.scheme == 'https' and netloc.endswith(':443')):
        netloc = netloc.rsplit(':', 1)[0]
    return urllib.parse.urlunsplit((p.scheme, netloc, path, query, ''))


# =====================================================================
# 予測器(バイト単位のPPM-C。数え上げだけで、勾配は使わない)
# =====================================================================
class PPM:
    def __init__(self, order=5):
        self.order = order
        self.tabs = [dict() for _ in range(order + 1)]

    def n_contexts(self):
        return sum(len(t) for t in self.tabs)

    def code(self, data, learn, gate=0.0):
        """data の符号長(ビット)を返す。learn=True なら学習する。
        gate>0 なら、予測が当たった(符号長が gate ビット未満の)バイトでは更新を省く"""
        order = self.order
        tabs = self.tabs
        log2 = math.log2
        bits = 0.0
        for i in range(len(data)):
            s = data[i]
            top = i if i < order else order
            p = 1.0
            hit = False
            for k in range(top, -1, -1):
                e = tabs[k].get(data[i - k:i])
                if e is None:
                    continue
                cnt, tot = e
                c = cnt.get(s)
                d = len(cnt)
                if c is not None:
                    p *= c / (tot + d)
                    hit = True
                    break
                p *= d / (tot + d)
            if not hit:
                p *= 1.0 / 256
            sb = -log2(p)
            bits += sb
            if learn and sb >= gate:
                for k in range(top + 1):
                    key = data[i - k:i]
                    e = tabs[k].get(key)
                    if e is None:
                        tabs[k][key] = [{s: 1}, 1]
                    else:
                        cnt = e[0]
                        cnt[s] = cnt.get(s, 0) + 1
                        e[1] += 1
        return bits

    def bpc(self, data):
        return self.code(data, False) / max(1, len(data))

    def prune(self, budget):
        """睡眠: 記憶が予算を超えたら、見た回数の少ない高次の文脈から忘れる"""
        removed = 0
        for thr in (1, 2, 3, 5):
            if self.n_contexts() <= budget:
                break
            for k in range(self.order, 1, -1):
                tab = self.tabs[k]
                dead = [key for key, e in tab.items() if e[1] <= thr]
                for key in dead:
                    del tab[key]
                removed += len(dead)
                if self.n_contexts() <= budget:
                    break
        return removed


def self_bpc(data, order=3):
    """安い仕分け用: その場で自己圧縮してみる(前半で学び、後半の符号長を測る)"""
    m = PPM(order)
    half = len(data) // 2
    m.code(data[:half], True)
    return m.bpc(data[half:])


# =====================================================================
# エージェントの状態
# =====================================================================
class Domain:
    def __init__(self, host):
        self.host = host
        self.frontier = collections.deque()
        self.probe = None
        self.hist = []
        self.nvis = 0
        self.tries = 0
        self.pages = 0
        self.dups = 0
        self.empty = 0
        self.errors = 0
        self.junk = False
        self.junk_reason = ''
        self.next_ok = 0.0


class State:
    def __init__(self, order):
        self.model = PPM(order)
        self.domains = {}
        self.seen = set()
        self.hashes = set()
        self.pages = 0
        self.fetches = 0
        self.bytes = 0
        self.on_bits = 0.0
        self.on_bytes = 0
        self.ema = None
        self.probe_order = []
        self.elapsed = 0.0


def enqueue(st, url, args):
    u = norm_url(url)
    if u is None or u in st.seen:
        return
    host = urllib.parse.urlsplit(u).netloc
    d = st.domains.get(host)
    if d is None:
        if len(st.domains) >= args.max_domains:
            return
        d = Domain(host)
        st.domains[host] = d
    if d.junk or len(d.frontier) >= args.frontier_cap:
        return
    st.seen.add(u)
    d.frontier.append(u)


def group_of(host):
    """同じ運営の場所をまとめる(例: wikipedia.org と wikiquote.org は 'wikimedia')"""
    parts = host.split(':')[0].split('.')
    reg = '.'.join(parts[-2:]) if len(parts) >= 2 else host
    return 'wikimedia' if reg.startswith('wik') else reg


def lp_of(d, w):
    if d.probe is None or d.nvis < w or len(d.hist) < w + 1:
        return None
    return (d.hist[-1 - w] - d.hist[-1]) / w


def pick_domain(st, args, rng):
    now = time.time()
    ready, wait = [], None
    for d in st.domains.values():
        if d.junk or not d.frontier or d.pages >= args.max_pages_per_domain:
            continue
        if args.max_pages_per_group and getattr(st, 'group_pages', {}).get(group_of(d.host), 0) >= args.max_pages_per_group:
            continue
        if d.next_ok > now:
            wait = d.next_ok if wait is None else min(wait, d.next_ok)
            continue
        ready.append(d)
    if not ready:
        return None, wait
    if args.policy == 'random':
        return rng.choice(ready), None
    lps = [max(lp_of(d, args.window), 0.0) for d in st.domains.values()
           if lp_of(d, args.window) is not None]
    prior = max(0.02, 2.0 * (sum(lps) / len(lps))) if lps else 0.5
    best, best_s = None, None
    for d in ready:
        lp = lp_of(d, args.window)
        if d.nvis == 0:
            s = prior
        elif lp is None:
            s = prior * 1.2
        else:
            s = max(lp, 0.0) + args.beta / math.sqrt(1 + d.nvis)
        s += 1e-6 * rng.random()
        if best_s is None or s > best_s:
            best, best_s = d, s
    return best, None


def mark_junk(d, why):
    d.junk = True
    d.junk_reason = why
    d.frontier.clear()


def visit(st, d, args, guard, rng, urls_log):
    url = d.frontier.popleft()
    r = guard.fetch(url)
    st.fetches += 1
    d.next_ok = time.time() + max(args.min_delay, guard.crawl_delay(url))
    if urls_log:
        urls_log.write('%d\t%s\t%s\t%s\n' % (time.time(), r.get('status'), r.get('error') or '-', url))
        urls_log.flush()
    if r.get('status') in (429, 503):
        d.next_ok = time.time() + 180          # 混み合っている: しばらく待つ(礼儀)
    if r.get('error') or r.get('status') != 200:
        d.errors += 1
        if d.errors >= 5 and d.pages == 0:
            mark_junk(d, 'errors')
        return
    text, links, noindex, nofollow = extract(r['url'], r['body'], r['charset'], r['ctype'])
    if not nofollow:
        for l in links:
            enqueue(st, l, args)
    if noindex:
        return
    data = text.encode('utf-8', 'ignore')
    d.tries += 1
    if len(data) < 300:
        d.empty += 1
        if d.empty >= 3 and d.pages == 0:
            mark_junk(d, 'empty')
        return
    h = hashlib.blake2b(data[:4096] + len(data).to_bytes(4, 'big'), digest_size=8).digest()
    if h in st.hashes:
        d.dups += 1
        if d.dups >= 8 and d.dups > d.pages:
            mark_junk(d, 'duplicates')
        return
    st.hashes.add(h)
    d.pages += 1
    st.pages += 1
    if not hasattr(st, 'group_pages'):
        st.group_pages = {}
    g = group_of(d.host)
    st.group_pages[g] = st.group_pages.get(g, 0) + 1
    st.bytes += len(data)
    model = st.model
    if d.probe is None and d.nvis == 0 and len(data) >= 1500:
        if args.prescreen and self_bpc(data[:4000]) >= args.junk_bpc:
            mark_junk(d, 'incompressible')
            return
        mid = len(data) // 2
        d.probe = data[mid - 700: mid + 700]
        train = data[:mid - 700] + b'\n' + data[mid + 700:]
        d.hist.append(model.bpc(d.probe))
        bits = model.code(train, True, args.gate)
        n = len(train)
        d.hist.append(model.bpc(d.probe))
        d.nvis = 1
        st.probe_order.append(d.host)
    else:
        bits = model.code(data, True, args.gate)
        n = len(data)
        if d.probe is not None:
            d.hist.append(model.bpc(d.probe))
            d.nvis += 1
        elif d.tries >= 4:
            mark_junk(d, 'too-short')
    st.on_bits += bits
    st.on_bytes += n
    b = bits / n
    st.ema = b if st.ema is None else 0.97 * st.ema + 0.03 * b


# =====================================================================
# 記録・保存
# =====================================================================
def metrics_row(st, args):
    hosts = st.probe_order[:30]
    probes = [st.domains[h].probe for h in hosts if st.domains[h].probe]
    pm = lz = None
    if probes:
        n = sum(len(p) for p in probes)
        pm = sum(st.model.code(p, False) for p in probes) / n
        cat = b'\n'.join(probes)
        lz = len(lzma.compress(cat, preset=6)) * 8 / len(cat)
    top = sorted(st.domains.values(), key=lambda d: -d.pages)[:5]
    return {
        't': round(st.elapsed), 'pages': st.pages, 'fetches': st.fetches,
        'kb_read': round(st.bytes / 1024), 'domains': len(st.domains),
        'probed': sum(1 for d in st.domains.values() if d.probe),
        'junk': sum(1 for d in st.domains.values() if d.junk),
        'contexts': st.model.n_contexts(),
        'online_bpc_ema': None if st.ema is None else round(st.ema, 3),
        'probe_bpc': None if pm is None else round(pm, 3),
        'lzma_bpc': None if lz is None else round(lz, 3),
        'top_domains': [(d.host, d.pages) for d in top]}


def save_state(st, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(st, f, protocol=5)
    os.replace(tmp, path)


def make_parser():
    ap = argparse.ArgumentParser(description='自律採食エージェント v0')
    ap.add_argument('--contact', default='', help='User-Agentに入れる連絡先URL/メール(必須)')
    ap.add_argument('--seed-url', action='append', default=[], help='種のURL(複数可)')
    ap.add_argument('--seeds', default='', help='種のURLを1行ずつ書いたファイル')
    ap.add_argument('--max-pages-per-group', type=int, default=2500, help='同じ運営の場所(例: Wikimedia全体)から読むページ数の上限。0で無制限')
    ap.add_argument('--deny-file', default='', help='拒否するドメインを1行ずつ書いたファイル')
    ap.add_argument('--state', default='state.pkl')
    ap.add_argument('--metrics', default='metrics.jsonl')
    ap.add_argument('--urls-log', default='urls.log')
    ap.add_argument('--time-budget-min', type=float, default=20.0)
    ap.add_argument('--max-pages', type=int, default=10**9)
    ap.add_argument('--max-fetches', type=int, default=10**9)
    ap.add_argument('--min-delay', type=float, default=3.0)
    ap.add_argument('--max-pages-per-domain', type=int, default=100)
    ap.add_argument('--max-domains', type=int, default=20000)
    ap.add_argument('--frontier-cap', type=int, default=300)
    ap.add_argument('--budget-contexts', type=int, default=1_500_000)
    ap.add_argument('--order', type=int, default=5)
    ap.add_argument('--policy', choices=['progress', 'random'], default='progress')
    ap.add_argument('--no-prescreen', dest='prescreen', action='store_false')
    ap.add_argument('--junk-bpc', type=float, default=6.0)
    ap.add_argument('--gate', type=float, default=0.0)
    ap.add_argument('--window', type=int, default=3)
    ap.add_argument('--beta', type=float, default=0.05)
    ap.add_argument('--ckpt-sec', type=float, default=600.0)
    ap.add_argument('--metrics-sec', type=float, default=300.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--allow-local', action='store_true', help='試験用: ローカルアドレスを許可')
    ap.add_argument('--quiet', action='store_true')
    return ap


def main(argv=None):
    args = make_parser().parse_args(argv)
    if not args.contact and not args.allow_local:
        sys.exit('--contact(連絡先のURLかメール)が必要です。身元を明示せずに巡回しません。')
    deny = []
    if args.deny_file and os.path.exists(args.deny_file):
        deny = open(args.deny_file, encoding='utf-8').read().split()
    guard = Guard(args.contact or 'local-test', allow_local=args.allow_local, deny=deny)
    if args.state and os.path.exists(args.state):
        with open(args.state, 'rb') as f:
            st = pickle.load(f)
        for d in st.domains.values():
            d.next_ok = 0.0
            d.frontier = collections.deque(u for u in (norm_url(x) for x in d.frontier) if u)
        if not args.quiet:
            print('再開: pages=%d domains=%d' % (st.pages, len(st.domains)), flush=True)
    else:
        st = State(args.order)
    seeds = list(args.seed_url)
    if args.seeds and os.path.exists(args.seeds):
        for line in open(args.seeds, encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#'):
                seeds.append(line)
    for s in seeds:
        enqueue(st, s, args)
    rng = random.Random(args.seed * 1000003 + st.pages)
    urls_log = open(args.urls_log, 'a', encoding='utf-8') if args.urls_log else None
    t_start = time.time()
    deadline = t_start + args.time_budget_min * 60
    base_elapsed = st.elapsed
    last_ckpt = last_met = time.time()
    pages0 = st.pages
    fetches0 = st.fetches
    while (time.time() < deadline and (st.pages - pages0) < args.max_pages
           and (st.fetches - fetches0) < args.max_fetches):
        if os.path.exists('STOP'):
            print('STOP ファイルを検出: 停止します', flush=True)
            break
        d, wait = pick_domain(st, args, rng)
        if d is None:
            if wait is None:
                print('辿れる場所がなくなりました', flush=True)
                break
            time.sleep(min(max(wait - time.time(), 0.0), 1.0))
            continue
        visit(st, d, args, guard, rng, urls_log)
        st.elapsed = base_elapsed + (time.time() - t_start)
        now = time.time()
        if args.metrics and now - last_met >= args.metrics_sec:
            row = metrics_row(st, args)
            with open(args.metrics, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            if not args.quiet:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            last_met = now
        if args.state and now - last_ckpt >= args.ckpt_sec:
            st.model.prune(args.budget_contexts)
            save_state(st, args.state)
            last_ckpt = now
    st.elapsed = base_elapsed + (time.time() - t_start)
    st.model.prune(args.budget_contexts)
    if args.state:
        save_state(st, args.state)
    if args.metrics:
        with open(args.metrics, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics_row(st, args), ensure_ascii=False) + '\n')
    if urls_log:
        urls_log.close()
    return st


if __name__ == '__main__':
    main()
