"""文章(WikiText)とコード(Pythonのソースそのもの)を混ぜた訓練コーパスを作る。
使い方: python3 make_mix.py <wt.pkl> <out.pkl> <コードの割合r> <訓練の総語数T>
 出力: {0:文章テスト, 1:文章検証, 2:混合訓練(文書をランダム順), 3:コードのテスト}   ※exp_y の extra_tests:[3] でコードのテストも測れる
 コード = 標準ライブラリ+インストール済みの .py 全文(docstring以外のコードも含む)。小文字化して [a-z]+|数字|記号 に分割(exp_scaleと同じ)。"""
import os, sys, re, ast, hashlib, pickle, sysconfig
import numpy as np
TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
wt_path, out_path, r, T = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
wt = pickle.load(open(wt_path, 'rb'))
need_code = int(r * T); need_text = T - need_code
def take(docs, n):
    out = []; tot = 0
    for d in docs:
        if tot >= n: break
        out.append(d[:n - tot]); tot += len(out[-1])
    return out
text_train = take(wt[2], need_text)
code_test = []; code_train = []
if True:
    roots = [sysconfig.get_paths()['stdlib'], sysconfig.get_paths()['purelib']]; files = []
    for rt in roots:
        for dp, dn, fn in os.walk(rt):
            dn.sort()
            for f in sorted(fn):
                if f.endswith('.py'): files.append(os.path.join(dp, f))
    files.sort(key=lambda p: hashlib.md5(p.encode()).hexdigest()); seen = set(); nte = ntr = 0
    for p in files:
        if ntr >= need_code and nte >= 1_500_000: break
        h = int(hashlib.md5(p.encode()).hexdigest(), 16) % 10
        try: txt = open(p, encoding='utf-8').read()
        except Exception: continue
        k = hashlib.md5(txt.encode()).hexdigest()
        if k in seen or len(txt) < 500: continue
        seen.add(k); toks = TOK.findall(txt.lower())
        if len(toks) < 200: continue
        if h == 0 and nte < 1_500_000: code_test.append(toks); nte += len(toks)
        elif h >= 2 and ntr < need_code: code_train.append(toks); ntr += len(toks)
    code_train = take(code_train, need_code)
mixed = text_train + code_train
rng = np.random.default_rng(0); order = rng.permutation(len(mixed)); mixed = [mixed[i] for i in order]
if r == 0:   # コードのテストだけは用意しておく(r>0と同じ方法)
    pass
print('text_train', sum(len(d) for d in text_train), 'code_train', sum(len(d) for d in code_train), 'code_test', sum(len(d) for d in code_test), flush=True)
pickle.dump({0: wt[0], 1: wt[1], 2: mixed, 3: code_test}, open(out_path, 'wb'))
