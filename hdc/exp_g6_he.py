"""
G6b: HumanEval (164問) に、 3-gram の数え上げ (リテラルも語として保持) で 貪欲に関数本体を生成して、 実行して通るかを測る。
  学習: Python 標準ライブラリ (許諾の緩い人手のコード)。 語単位 3-gram のバックオフ (最頻の次語)。
  評価: 生成した本体を プロンプトにつないで実行し、 HumanEval の check を通れば成功 (pass@1, 貪欲 1 本)。
使い方: python3 exp_g6_he.py '{}'
"""
import sys, json, io, glob, tokenize, sysconfig, gzip, subprocess, tempfile, os
from collections import Counter, defaultdict

SPEC = {tokenize.NEWLINE: 'NEWLINE', tokenize.NL: 'NL', tokenize.INDENT: 'INDENT', tokenize.DEDENT: 'DEDENT'}


def toks(src):
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.ENDMARKER or t.type == tokenize.COMMENT:
                continue
            out.append(SPEC.get(t.type, t.string))
    except Exception:
        pass
    return out


def detok(ts):
    ind = 0; lines = ['']; 
    for t in ts:
        if t == 'INDENT':
            ind += 1
        elif t == 'DEDENT':
            ind = max(0, ind - 1)
        elif t in ('NEWLINE', 'NL'):
            lines.append('    ' * ind)
        else:
            lines[-1] += (' ' if lines[-1].strip() else '') + t
    return '\n'.join(lines)


def main(cfg):
    p = sysconfig.get_paths()['stdlib']; tri = defaultdict(Counter); bi = defaultdict(Counter); uni = Counter()
    for f in sorted(glob.glob(p + '/**/*.py', recursive=True)):
        if 'site-packages' in f or '/test' in f:
            continue
        try:
            ts = toks(open(f, encoding='utf-8').read())
        except Exception:
            continue
        ts = ['<s>', '<s>'] + ts
        for a, b, c in zip(ts, ts[1:], ts[2:]):
            tri[(a, b)][c] += 1; bi[b][c] += 1; uni[c] += 1
    print('学習: 3-gram 文脈 %d' % len(tri), flush=True)
    import urllib.request
    data = [json.loads(l) for l in gzip.open('/tmp/he.gz', 'rt')]
    ok = 0; res = []
    for k, pr in enumerate(data):
        ptoks = toks(pr['prompt'])
        while ptoks and ptoks[-1] in ('NEWLINE', 'NL', 'DEDENT'):
            ptoks.pop()
        ctx = ptoks[-2:] + ['NEWLINE', 'INDENT']; gen = []; depth = 1
        pn = set(ptoks)
        for _ in range(150):
            a, b = (['<s>', '<s>'] + ptoks + ['NEWLINE', 'INDENT'] + gen)[-2:]
            c = tri.get((a, b)) or bi.get(b) or uni
            # プロンプトに出る語を少しだけ優先 (コピー): 候補の上位5つの中でプロンプトにあるもの
            top = [w for w, _ in c.most_common(5)]
            w = next((x for x in top if x in pn or x in 'NEWLINE INDENT DEDENT ( ) : , = return'), top[0])
            gen.append(w)
            if w == 'INDENT':
                depth += 1
            if w == 'DEDENT':
                depth -= 1
                if depth <= 0:
                    break
        code = pr['prompt'].rstrip('\n') + '\n' + detok(['INDENT'] + gen[:-1] if False else gen) + '\n'
        body = detok(gen)
        full = pr['prompt'] + '\n' + '\n'.join('    ' + l.strip() if l.strip() else l for l in body.split('\n')) + '\n\n' + pr['test'] + '\ncheck(%s)\n' % pr['entry_point']
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
            fh.write(full); name = fh.name
        try:
            r = subprocess.run([sys.executable, name], capture_output=True, timeout=5); passed = r.returncode == 0
        except Exception:
            passed = False
        os.unlink(name); ok += passed; res.append(passed)
    print('HumanEval pass@1 (貪欲): %d / %d' % (ok, len(data)))
    print('RESULT_G6B ' + json.dumps(dict(passed=ok, total=len(data))))


if __name__ == '__main__':
    main(json.loads(sys.argv[1] if len(sys.argv) > 1 else '{}'))
