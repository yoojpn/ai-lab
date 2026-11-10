"""100億語の索引 (FM-index) を 分割ごとに作って Hugging Face に保存する。 Colab / 手元の両方で動く。
流れ (1 分割 = 約 SHARD_TOK 語): 各データ源から 割り当て分を取る -> トークン化 -> 重複除去 (32 語の窓の標本ハッシュ) -> 分割ファイル -> 接尾辞配列 -> BWT -> xz 圧縮 -> HF に保存。
環境変数: HF_TOKEN, TINY=1 (試運転: 小さい量 / HF に上げない), WORK (作業フォルダ), TOTAL_TOK, SHARD_TOK, TIME_BUDGET_H"""
import os, sys, time, json, hashlib, threading, queue, subprocess, shutil, random, io, lzma
import numpy as np
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1'); os.environ.setdefault('RAYON_NUM_THREADS', '2')

TINY = os.environ.get('TINY') == '1'
TOKEN = os.environ.get('HF_TOKEN', '')
REPO = 'Youichi1/ai-lab-data'
WORK = os.environ.get('WORK', '/content/fm' if os.path.isdir('/content') else '/tmp/fmwork')
TOTAL = int(float(os.environ.get('TOTAL_TOK', 20e6 if TINY else 10e9)))
SHARD = int(float(os.environ.get('SHARD_TOK', 10e6 if TINY else 400e6)))
BUDGET = float(os.environ.get('TIME_BUDGET_H', 10)) * 3600
NROUND = max(1, TOTAL // SHARD)
os.makedirs(WORK, exist_ok=True)
T0 = time.time(); LOGF = WORK + '/build_log.txt'


def lg(s):
    line = '[%6ds] %s' % (time.time() - T0, s); print(line, flush=True)
    with open(LOGF, 'a') as f: f.write(line + '\n')


from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
import pyarrow.parquet as pq
api = HfApi(token=TOKEN); fsys = HfFileSystem(token=TOKEN)


from huggingface_hub import CommitOperationAdd


def hf_up_many(pairs, tries=40):
    """複数のファイルを 1 コミットで 保存 (HF は 1 時間に 128 コミットまで)。 制限に当たったら 待って やり直す"""
    if TINY: return True
    pairs = [(l, r) for l, r in pairs if os.path.exists(l)]
    for k in range(tries):
        try:
            api.create_commit(repo_id=REPO, repo_type='dataset', operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for l, r in pairs], commit_message='fm build ' + ','.join(r for _, r in pairs)[:100]); return True
        except Exception as e:
            msg = str(e)
            if k < 3 or k % 5 == 0: lg('upload 失敗 %s (%d): %s' % (pairs[0][1], k, msg[-160:].replace('\n', ' ')))
            time.sleep(min(150, 30 * (k + 1)))
    return False


def hf_up(local, remote, tries=40):
    return hf_up_many([(local, remote)], tries)


def hf_files(repo, prefix):
    return sorted(f for f in api.list_repo_files(repo, repo_type='dataset') if f.startswith(prefix))


# ---------------------------------------------------------------- データ源
def prefetch(gen, depth=3):
    q = queue.Queue(depth); DONE = object()

    def run():
        try:
            for x in gen: q.put(x)
        except Exception as e: lg('読み込みの例外: %s' % str(e)[:200])
        q.put(DONE)
    threading.Thread(target=run, daemon=True).start()

    def out():
        while True:
            try: x = q.get(timeout=900)
            except queue.Empty: lg('読み込みが 15 分 止まった: この源を 尽きたことにする'); return
            if x is DONE: return
            yield x
    return out()


def dl(repo, path):
    """ファイルを まるごと ダウンロード (範囲読みより 30 倍ほど速い)。 戻り値: 手元のパス"""
    for k in range(5):
        try: return hf_hub_download(repo, path, repo_type='dataset', token=TOKEN, local_dir=WORK + '/dl/' + repo.replace('/', '_'))
        except Exception as e: lg('ダウンロード失敗 %s %s (%d)' % (repo, path, k)); time.sleep(10 * (k + 1))
    raise RuntimeError('download failed ' + path)


def pq_rows(repo, files, cols, rg_step, seed, skip=0):
    """ファイルを 1 つずつ ダウンロードして 行グループを (rg_step おきに) 読み、 使い終わったら 消す"""
    rng = random.Random(seed); files = list(files); rng.shuffle(files)
    for f in files:
        p = None
        try:
            p = dl(repo, f); pf = pq.ParquetFile(p); n = pf.num_row_groups; off = rng.randrange(rg_step)
            for rg in range(off, n, rg_step):
                for r in pf.read_row_group(rg, columns=cols).to_pylist(): yield r
        except Exception as e: lg('ファイル失敗 %s: %s' % (f, str(e)[:120]))
        finally:
            if p and os.path.exists(p): os.remove(p)


def src_web(repo, prefix, textcol='text', rg_step=8, minchar=80, extra=None, seed=1):
    files = [f for f in hf_files(repo, prefix) if f.endswith('.parquet')]
    cols = [textcol] + (extra or [])
    for r in pq_rows(repo, files, cols, rg_step, seed):
        t = r.get(textcol)
        if t and len(t) >= minchar: yield [t[:60000]], False


def src_ja(seed=1):
    for r in pq_rows('HuggingFaceFW/fineweb-2', [f for f in hf_files('HuggingFaceFW/fineweb-2', 'data/jpn_Jpan/train') if f.endswith('.parquet')], ['text', 'language_score'], 3, seed):
        if r['text'] and len(r['text']) >= 80 and (r.get('language_score') or 0) >= 0.8: yield [r['text'][:60000]], False


def src_wiki(cfg, seed=1):
    files = [f for f in hf_files('wikimedia/wikipedia', cfg + '/') if f.endswith('.parquet')]
    for r in pq_rows('wikimedia/wikipedia', files, ['title', 'text'], 2 if cfg.endswith('.en') else 1, seed):
        if r['text'] and len(r['text']) >= 100: yield [(r['title'] + '\n' + r['text'])[:60000]], False


PERM = {'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc', 'unlicense', 'cc0-1.0'}


def src_code(seed=1):
    files = [f for f in hf_files('codeparrot/github-code-clean', 'data/') if f.endswith('.parquet')]
    for r in pq_rows('codeparrot/github-code-clean', files, ['code', 'license', 'size'], 3, seed):
        if r['license'] in PERM and 300 < (r['size'] or 0) < 100000 and r['code']: yield [r['code']], False


def src_se(seed=1):
    files = [f for f in hf_files('HuggingFaceH4/stack-exchange-preferences', 'data/') if f.endswith('.parquet') and '.meta.' not in f]
    random.Random(seed).shuffle(files)
    for f in files:
        try:
            p = dl('HuggingFaceH4/stack-exchange-preferences', f); t = pq.read_table(p, columns=['question', 'answers']).to_pylist(); os.remove(p)
        except Exception as e: lg('SE 失敗 %s %s' % (f, str(e)[:100])); continue
        for r in t:
            ans = r.get('answers') or []
            if not ans or not r.get('question'): continue
            b = max(ans, key=lambda a: a.get('pm_score') or 0)
            if b.get('text'): yield [r['question'][:8000], b['text'][:8000]], True


def jsonl_lines(repo, path):
    p = dl(repo, path)
    try:
        with open(p, encoding='utf-8') as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    try: yield json.loads(ln)
                    except Exception: pass
    finally:
        if os.path.exists(p): os.remove(p)


def to_utts(d):
    if 'conversations' in d: return [c.get('value', '') for c in d['conversations'] if c.get('value')]
    if 'instruction' in d:
        a = d['instruction'] + (('\n' + d['input']) if d.get('input') else ''); return [a, d.get('output', '')] if d.get('output') else []
    if 'text' in d: return [d['text']]
    return []


def src_jsonl_chat(repo, path):
    for d in jsonl_lines(repo, path):
        u = to_utts(d)
        if len(u) >= 2: yield [x[:8000] for x in u], True


def src_oasst1(seed=1):
    rows = []
    for f in hf_files('OpenAssistant/oasst1', 'data/'):
        if f.endswith('.parquet'):
            p = dl('OpenAssistant/oasst1', f); rows += pq.read_table(p, columns=['message_id', 'parent_id', 'text', 'deleted']).to_pylist(); os.remove(p)
    by = {r['message_id']: r for r in rows if not r['deleted']}; ch = set(r['parent_id'] for r in by.values() if r['parent_id'])
    leaves = [m for m in by if m not in ch]; random.Random(seed).shuffle(leaves)
    for m in leaves:
        u = []; x = by[m]
        while x:
            u.append(x['text'][:8000]); x = by.get(x['parent_id'])
        if len(u) >= 2: yield u[::-1], True


def src_aux(name):
    p = hf_hub_download(REPO, 'aux/' + name, repo_type='dataset', token=TOKEN, local_dir=WORK + '/aux')
    for ln in open(p):
        if ln.strip():
            d = json.loads(ln); yield [x[:4000] for x in d], True


def make_sources():
    S = {}
    S['ja_web'] = lambda: src_ja(); S['ja_wiki'] = lambda: src_wiki('20231101.ja'); S['en_edu'] = lambda: src_web('HuggingFaceFW/fineweb-edu', 'sample/10BT', rg_step=6, minchar=200)
    S['en_wiki'] = lambda: src_wiki('20231101.en'); S['code'] = lambda: src_code()
    S['math_owm'] = lambda: src_web('open-web-math/open-web-math', 'data/', rg_step=4, minchar=200); S['math_fine'] = lambda: src_web('HuggingFaceTB/finemath', 'finemath-4plus', rg_step=4, minchar=200)
    S['dlg_se'] = lambda: src_se(); S['dlg_izumi'] = lambda: src_jsonl_chat('izumi-lab/llm-japanese-dataset', 'data-cc-by-sa.jsonl')
    S['dlg_ja_oasst'] = lambda: iter_chain(src_jsonl_chat('llm-jp/oasst1-21k-ja', 'oasst1-21k-ja.jsonl'), src_jsonl_chat('llm-jp/oasst2-33k-ja', 'oasst2-33k-ja.jsonl'), src_jsonl_chat('llm-jp/databricks-dolly-15k-ja', 'databricks-dolly-15k-ja.jsonl'))
    S['dlg_oasst1'] = lambda: src_oasst1(); S['dlg_rpc'] = lambda: src_aux('rpc_train.jsonl')
    return S


def iter_chain(*gs):
    for g in gs:
        for x in g: yield x


# 全体の割り当て (語)。 足りない分は ja_web に回す
QUOTA = {'ja_web': 3.4e9, 'ja_wiki': 0.6e9, 'en_edu': 2.0e9, 'en_wiki': 0.5e9, 'code': 1.5e9, 'math_owm': 0.7e9, 'math_fine': 0.5e9, 'dlg_se': 0.3e9, 'dlg_izumi': 0.25e9, 'dlg_ja_oasst': 0.1e9, 'dlg_oasst1': 0.03e9, 'dlg_rpc': 0.03e9}
QSUM = sum(QUOTA.values()); QUOTA = {k: v / QSUM * TOTAL for k, v in QUOTA.items()}

# ---------------------------------------------------------------- トークン化 / 重複除去
from tokenizers import Tokenizer
tkp = hf_hub_download(REPO, 'tok/tok.json', repo_type='dataset', token=TOKEN, local_dir=WORK + '/tokhf'); TK = Tokenizer.from_file(tkp)
M1 = np.uint64(0x9E3779B97F4A7C15); M2 = np.uint64(0xBF58476D1CE4E5B9); M3 = np.uint64(0x94D049BB133111EB); M4 = np.uint64(0x2545F4914F6CDD1D); WIN = 32
G = np.empty(0, np.uint64)          # 見た窓の鍵 (整列済み) + 文書全体のハッシュ
P = []                              # この分割の中で 追加した鍵 (整列済みの塊の列)


def mix(a):
    a = (a + np.uint64(1)) * M1; a ^= a >> np.uint64(29); a *= M2; a ^= a >> np.uint64(32); a *= M3; a ^= a >> np.uint64(29); return a


def keys_of(arrs):
    """各文書 (uint16 配列) の 窓の標本鍵と 文書ハッシュ。 戻り値: 鍵, 文書番号の配列 (窓), 文書ごとの鍵 (文書ハッシュ)"""
    lens = np.array([len(a) for a in arrs]); cat = np.concatenate(arrs).astype(np.uint64); g = mix(cat); C = np.concatenate([[np.uint64(0)], np.cumsum(g, dtype=np.uint64)])
    doc = np.repeat(np.arange(len(arrs)), lens); N = len(cat); dh = np.zeros(len(arrs), np.uint64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    for i, a in enumerate(arrs): dh[i] = np.uint64(int.from_bytes(hashlib.blake2b(a.tobytes(), digest_size=8).digest(), 'little'))
    if N >= WIN:
        H = C[WIN:] - C[:-WIN]; ok = doc[:N - WIN + 1] == doc[WIN - 1:]; samp = ((H * M4) >> np.uint64(58)) == 0; m = ok & samp
        wk = H[m]; wd = doc[:N - WIN + 1][m]
    else: wk = np.empty(0, np.uint64); wd = np.empty(0, np.int64)
    return wk, wd, dh


def isin_chunks(chunks, x):
    r = np.zeros(len(x), bool)
    for c in chunks: r |= isin_sorted(c, x)
    return r


def isin_sorted(S, x):
    if len(S) == 0: return np.zeros(len(x), bool)
    j = np.minimum(np.searchsorted(S, x), len(S) - 1); return S[j] == x


def dedup_batch(arrs, add=True):
    """重複の文書を落とす。 戻り値: 残す文書のフラグ。 add=True なら 残した文書の鍵を P に足す"""
    global P
    wk, wd, dh = keys_of(arrs); n = len(arrs); keep = np.ones(n, bool)
    dupdoc = isin_sorted(G, dh) | isin_chunks(P, dh); u, first = np.unique(dh, return_index=True); fo = np.full(n, True); fo[:] = False; fo[first] = True; dupdoc |= ~fo   # 同じ batch の 2 番目以降
    keep &= ~dupdoc
    if len(wk):
        dw = isin_sorted(G, wk) | isin_chunks(P, wk); o = np.lexsort((wd, wk)); ks = wk[o]; ds = wd[o]; firstmask = np.ones(len(ks), bool); firstmask[1:] = ks[1:] != ks[:-1]
        owner = np.empty(len(wk), np.int64); gi = np.cumsum(firstmask) - 1; owner_sorted = ds[firstmask][gi]; owner[o] = owner_sorted; dw |= owner < wd
        ns = np.bincount(wd, minlength=n); nd = np.bincount(wd[dw], minlength=n); keep &= ~((ns >= 2) & (nd >= 0.5 * np.maximum(ns, 1)))
    if add:
        mk = keep[wd] if len(wk) else np.zeros(0, bool); newk = np.concatenate([dh[keep], wk[mk]]) if len(wk) else dh[keep]
        P.append(np.unique(newk))
        if len(P) > 40: P[:] = [np.unique(np.concatenate(P))]
    return keep


def tokenize(docs):
    """docs: [(utts, is_dlg)] -> 文書ごとの uint16 配列 (対話は 発話ごとに 2 を足す)"""
    flat = []; idx = []
    for utts, d in docs:
        idx.append((len(flat), len(utts), d)); flat += utts
    enc = TK.encode_batch(flat); out = []
    for s, k, d in idx:
        if d:
            parts = []
            for e in enc[s:s + k]: parts.append(np.array(e.ids, np.uint16)); parts.append(np.array([2], np.uint16))
            out.append(np.concatenate(parts))
        else: out.append(np.array(enc[s].ids, np.uint16))
    return out


# ---------------------------------------------------------------- 検証用 / 混入防止
VAL = {}; VALCAP = int(3e6 if not TINY else 2e5)


def blocklist_init():
    """評価データの窓を 先に 鍵の集合に入れる (学習に 重なる文書が 入らないように)"""
    global G
    arrs = []
    try:
        p = hf_hub_download(REPO, 'tok/val.bin', repo_type='dataset', token=TOKEN, local_dir=WORK + '/tokhf'); v = np.fromfile(p, dtype=np.uint16); arrs.append(v)
        VAL['prev_val'] = v[:VALCAP]
    except Exception as e: lg('tok/val.bin なし %s' % str(e)[:100])
    for nm in ('jsq_val_contexts.jsonl', 'rpc_test.jsonl'):
        p = hf_hub_download(REPO, 'aux/' + nm, repo_type='dataset', token=TOKEN, local_dir=WORK + '/aux') if not TINY or True else None
        txt = []
        for ln in open(p):
            if ln.strip():
                d = json.loads(ln); txt += (d if isinstance(d, list) else [d])
        for a in range(0, len(txt), 500): arrs += [np.array(e.ids, np.uint16) for e in TK.encode_batch([t[:20000] for t in txt[a:a + 500]])]
        if nm.startswith('rpc'): VAL['dlg_rpc_test'] = np.concatenate([np.concatenate([np.array(e.ids, np.uint16), [2]]) for e in TK.encode_batch(txt)]).astype(np.uint16)
    arrs = [a for a in arrs if len(a) >= 1]; ks = []
    for a in range(0, len(arrs), 2000):
        ch = arrs[a:a + 2000]; wk, wd, dh = keys_of(ch); ks += [wk, dh]
    G = np.unique(np.concatenate(ks)); lg('混入防止の鍵 %d' % len(G))


def is_val(arr, src):
    h = int.from_bytes(hashlib.blake2b(arr[:64].tobytes(), digest_size=8).digest(), 'little')
    return h % 2000 == 0 and VALN.get(src, 0) < VALCAP


VALN = {}; DEFICIT = 0.0

# ---------------------------------------------------------------- 構築
SRC = make_sources(); GENS = {}; CONSUMED = {k: 0 for k in SRC}; EXH = set()


NPULL = {k: 0 for k in SRC}; SKIP = {}


def get_gen(name):
    if name not in GENS:
        g = prefetch(SRC[name]())
        for _ in range(SKIP.get(name, 0)):
            try: next(g)
            except StopIteration: EXH.add(name); break
        NPULL[name] = SKIP.get(name, 0); GENS[name] = g
    return GENS[name]


import multiprocessing as mp
CTX = mp.get_context('fork'); QS = {}; PROCS = {}; LASTGOT = {}


def worker(name, q, skip):
    """データ源ごとに 1 プロセス: 読む -> トークン化 -> キューに入れる (parent が 重複除去と 組み立て)"""
    try:
        g = prefetch(SRC[name]()); n = 0
        for _ in range(skip):
            try: next(g)
            except StopIteration: break
        while True:
            batch = []
            for _ in range(256):
                try: batch.append(next(g))
                except StopIteration: break
            if not batch: break
            arrs = tokenize(batch); q.put((len(batch), [a[:200000] for a in arrs]))
    except Exception as e: lg('worker %s 例外: %s' % (name, str(e)[:200]))
    q.put(None)


def start_workers(skip):
    for name in QUOTA:
        QS[name] = CTX.Queue(6); p = CTX.Process(target=worker, args=(name, QS[name], skip.get(name, 0)), daemon=True); p.start(); PROCS[name] = p; NPULL[name] = skip.get(name, 0); LASTGOT[name] = time.time()


def fill_round(r, quota_round, stats):
    """全データ源の キューを 順繰りに 引いて 割り当て分まで 集める (読み込みと トークン化は 別プロセスで 並列)"""
    global DEFICIT
    names = list(QUOTA); per = {n: 0 for n in names}; parts = {n: [] for n in names}; need = {n: quota_round[n] + (DEFICIT if n == 'ja_web' else 0) for n in names}
    from queue import Empty
    while True:
        active = [n for n in names if per[n] < need[n] and n not in EXH]
        if not active: break
        idle = True
        for n in active:
            try: item = QS[n].get_nowait()
            except Empty:
                if time.time() - LASTGOT[n] > 900: lg('読み込みが 15 分 止まった: %s を 尽きたことにする' % n); EXH.add(n)
                continue
            idle = False; LASTGOT[n] = time.time()
            if item is None: EXH.add(n); continue
            cnt, arrs = item; NPULL[n] += cnt; ok = np.array([len(a) >= 8 for a in arrs])
            if not ok.any(): continue
            arrs = [a for a, k in zip(arrs, ok) if k]; keep = dedup_batch(arrs); stats['drop_' + n] = stats.get('drop_' + n, 0) + int((~keep).sum())
            for a, k in zip(arrs, keep):
                if not k: continue
                if is_val(a, n):
                    VAL.setdefault(n, []); VAL[n] = (VAL[n] if isinstance(VAL[n], list) else [VAL[n]]) + [a]; VALN[n] = VALN.get(n, 0) + len(a); continue
                parts[n].append(a); parts[n].append(np.zeros(1, np.uint16)); per[n] += len(a) + 1
            heartbeat(r, per)
        if idle: time.sleep(0.05)
    out = []; newdef = 0.0
    for n in names:
        out += parts[n]; stats['tok_' + n] = stats.get('tok_' + n, 0) + per[n]; newdef += max(0, need[n] - per[n])
    DEFICIT = newdef
    return out, per


import faulthandler
_stack = open(WORK + '/stack.txt', 'w'); faulthandler.dump_traceback_later(300, repeat=True, file=_stack)
HB = {'t': time.time(), 'up': time.time()}


def heartbeat(r, per):
    if time.time() - HB['t'] > 30:
        HB['t'] = time.time(); lg('  分割 %d 途中: %s | 引いた文書 %s' % (r, {k: int(v / 1e6) for k, v in per.items() if v}, {k: v for k, v in NPULL.items() if v}))
        if time.time() - HB['up'] > 600 and not TINY:
            HB['up'] = time.time(); threading.Thread(target=hf_up, args=(LOGF, 'results/fm_build_log.txt', 3)).start()


MKB = WORK + '/mkbwt'


def ensure_tools():
    if os.path.exists(MKB): return
    d = WORK + '/libsais'
    if not os.path.isdir(d): subprocess.check_call(['git', 'clone', '-q', '--depth', '1', 'https://github.com/IlyaGrebnov/libsais.git', d])
    src = hf_hub_download(REPO, 'fm/code/mkbwt.cpp', repo_type='dataset', token=TOKEN, local_dir=WORK + '/code') if not TINY else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mkbwt.cpp')
    subprocess.check_call(['gcc', '-O2', '-c', d + '/src/libsais.c', '-I' + d + '/include', '-o', WORK + '/libsais.o'])
    shutil.copy(src, WORK + '/mkbwt.cpp'); subprocess.check_call(['g++', '-O2', '-I' + WORK, WORK + '/mkbwt.cpp', WORK + '/libsais.o', '-o', MKB])
    # mkbwt.cpp は "libsais/include/libsais.h" を参照するので 作業フォルダに 合わせる
    return


def postprocess(r, tokfile, meta_extra):
    """接尾辞配列 -> BWT -> xz -> HF"""
    try:
        t = time.time(); bwt = '%s/s%02d.bwt' % (WORK, r); meta = '%s/s%02d.meta.json' % (WORK, r)
        subprocess.check_call([MKB, tokfile, bwt, meta]); lg('分割 %d: 接尾辞配列+BWT %.0fs' % (r, time.time() - t)); t = time.time()
        xzf = '%s/shard_%02d.bwt.xz' % (WORK, r)
        with open(xzf, 'wb') as fo: subprocess.check_call(['xz', '-1', '-T2', '-c', bwt], stdout=fo)
        m = json.load(open(meta)); m.update(meta_extra); m['bwt_xz_bytes'] = os.path.getsize(xzf); json.dump(m, open(meta, 'w'))
        lg('分割 %d: xz %.0fs, %.3f B/語' % (r, time.time() - t, m['bwt_xz_bytes'] / m['n']))
        ok = hf_up_many([(xzf, 'fm/shard_%02d.bwt.xz' % r), (meta, 'fm/shard_%02d.meta.json' % r)])
        if ok:
            for f in (tokfile, bwt, xzf):
                if not TINY or f != xzf: os.path.exists(f) and os.remove(f)
            lg('分割 %d: 保存 済み' % r)
        else: lg('分割 %d: 保存 失敗 (ファイルは 残す)' % r)
    except Exception as e: lg('分割 %d: postprocess 例外 %s' % (r, str(e)[:300]))


def save_state(r, stats, final=False):
    st = {'rounds_done': r + 1, 'stats': stats, 'time': time.time() - T0, 'npull': NPULL, 'deficit': DEFICIT}
    open(WORK + '/state.json', 'w').write(json.dumps(st)); pairs = [(WORK + '/state.json', 'fm/state.json'), (LOGF, 'results/fm_build_log.txt')]
    if final or (r + 1) % 6 == 0:
        for k, v in VAL.items():
            a = np.concatenate(v) if isinstance(v, list) else v; a.astype(np.uint16).tofile(WORK + '/val_%s.bin' % k); pairs.append((WORK + '/val_%s.bin' % k, 'fm/val/val_%s.bin' % k))
    threading.Thread(target=hf_up_many, args=(pairs,)).start()


def main():
    global G, P
    lg('開始: TINY=%s 合計 %.2e 語, 分割 %.2e 語 x %d, 予算 %.1fh' % (TINY, TOTAL, SHARD, NROUND, BUDGET / 3600))
    global SKIP, DEFICIT
    blocklist_init(); ensure_tools(); stats = {}; threads = []
    if not TINY:
        try:
            have = api.list_repo_files(REPO, repo_type='dataset')
            if 'fm/state.json' in have and 'fm/dedup_keys.npy' in have:
                p = hf_hub_download(REPO, 'fm/state.json', repo_type='dataset', token=TOKEN, local_dir=WORK + '/resume'); st = json.load(open(p)); SKIP = st.get('npull', {}); DEFICIT = st.get('deficit', 0.0); stats = st.get('stats', {})
                p = hf_hub_download(REPO, 'fm/dedup_keys.npy', repo_type='dataset', token=TOKEN, local_dir=WORK + '/resume'); G = np.union1d(G, np.load(p)); lg('再開: 分割 %d まで済み, 鍵 %d, 飛ばす %s' % (st['rounds_done'], len(G), SKIP))
        except Exception as e: lg('再開の読み込み失敗 %s' % str(e)[:150])
    start_workers(SKIP)
    for r in range(NROUND):
        if time.time() - T0 > BUDGET: lg('時間の予算を超えたので 終了'); break
        if not TINY:
            try:
                have = api.list_repo_files(REPO, repo_type='dataset')
                if 'fm/shard_%02d.meta.json' % r in have: lg('分割 %d は 保存済み (飛ばす)' % r); continue
            except Exception: pass
        t = time.time(); qr = {k: v / NROUND for k, v in QUOTA.items()}
        parts, per = fill_round(r, qr, stats); 
        if P: G = np.union1d(G, np.unique(np.concatenate(P)))
        P = []
        if not parts: lg('データが 尽きた'); break
        arr = np.concatenate([np.zeros(1, np.uint16)] + parts); tf = '%s/s%02d.tok' % (WORK, r); arr.tofile(tf)
        lg('分割 %d: %d 語 (%.0fs) 内訳 %s 鍵 %d' % (r, len(arr), time.time() - t, {k: int(v / 1e6) for k, v in per.items()}, len(G)))
        if (r + 1) % 4 == 0 and not TINY:
            np.save(WORK + '/dedup_keys.npy', G); threading.Thread(target=hf_up, args=(WORK + '/dedup_keys.npy', 'fm/dedup_keys.npy')).start()
        th = threading.Thread(target=postprocess, args=(r, tf, {'round': r, 'per_source': per})); th.start(); threads.append(th); save_state(r, stats)
        if len(threads) > 1: threads[-2].join()
        del parts, arr
    for th in threads: th.join()
    # 最後: 重複の鍵を 保存 (再開用)
    if not TINY:
        np.save(WORK + '/dedup_keys.npy', G); hf_up(WORK + '/dedup_keys.npy', 'fm/dedup_keys.npy')
    save_state(NROUND - 1, stats, final=True); time.sleep(5)
    # 保存に 失敗して 残っている 分割を もう一度 上げる
    for f in sorted(os.listdir(WORK)):
        if f.startswith('shard_') and f.endswith('.bwt.xz'):
            r = int(f[6:8]); mj = '%s/s%02d.meta.json' % (WORK, r)
            if os.path.exists(mj) and hf_up_many([(WORK + '/' + f, 'fm/' + f), (mj, 'fm/shard_%02d.meta.json' % r)]): lg('分割 %d: 再保存 済み' % r)
    lg('終わり')


if __name__ == '__main__':
    main()
