import re, time, numpy as np
from fm_chat import TK, ORD, blocked
CACHE = {}
import sys
OUT = [None]
def P_(m):
    f = OUT[0] or sys.__stdout__
    print(m, file=f, flush=True)
def dist(ctx):
    k = tuple(int(x) for x in ctx[-23:])
    if k not in CACHE:
        if len(CACHE) > 20000: CACHE.clear()
        CACHE[k] = fm.dist(list(k)).astype(np.float32)
    return CACHE[k]
# (質問, [許す答え])
QA = [
 ('日本の首都はどこですか？', ['東京']), ('富士山の高さは？', ['3776', '3,776', '3776']), ('東京タワーの高さは？', ['333']),
 ('アメリカの首都は？', ['ワシントン']), ('フランスの首都は？', ['パリ']), ('日本で一番高い山は？', ['富士山']),
 ('日本の初代内閣総理大臣は？', ['伊藤博文']), ('水の化学式は？', ['H2O', 'H₂O']), ('1年は何日？', ['365']),
 ('太陽系で一番大きい惑星は？', ['木星']), ('夏目漱石の代表作は？', ['吾輩は猫', '坊っちゃん', '坊ちゃん', 'こころ']), ('日本の通貨は？', ['円']),
 ('地球から一番近い恒星は？', ['太陽']), ('光の速さは？', ['30万', '299', '3.0']), ('中国の首都は？', ['北京']),
 ('韓国の首都は？', ['ソウル']), ('イギリスの首都は？', ['ロンドン']), ('琵琶湖は何県にある？', ['滋賀']),
 ('日本で一番面積の大きい都道府県は？', ['北海道']), ('源氏物語の作者は？', ['紫式部']), ('江戸幕府を開いたのは誰？', ['徳川家康', '家康']),
 ('世界で一番大きい海は？', ['太平洋']), ('人体で一番大きい臓器は？', ['皮膚']), ('水が沸騰する温度は？', ['100']),
 ('2020年に延期された夏季オリンピックの開催都市は？', ['東京']), ('北海道の道庁所在地は？', ['札幌']), ('大阪の有名な食べ物は？', ['たこ焼', 'お好み焼']),
 ('ドイツの首都は？', ['ベルリン']), ('地球の衛星の名前は？', ['月']), ('東京都庁があるのは何区？', ['新宿']),
 ('一週間は何日？', ['7', '七']), ('日本で一番長い川は？', ['信濃川']), ('織田信長が討たれた事件は？', ['本能寺']),
 ('日本の国鳥は？', ['キジ', '雉']), ('日本初のノーベル賞受賞者は？', ['湯川']), ('イタリアの首都は？', ['ローマ']),
 ('オーストラリアの首都は？', ['キャンベラ']), ('カナダの首都は？', ['オタワ']), ('「吾輩は猫である」の作者は？', ['夏目漱石', '漱石']),
 ('万有引力の法則を発見したのは誰？', ['ニュートン']), ('相対性理論を提唱したのは誰？', ['アインシュタイン']),
]
def to_cloze(q):
    s = q.strip(); s = re.sub(r'[？?！!。]+$', '', s)
    s = re.sub(r'(は|が)?(どこ|何県|何区|何日|何|誰|だれ|いつ|どれ)(にある|にいる|ですか|でしょうか|か)?$', lambda m: m.group(1) or '', s)
    s = re.sub(r'(ですか|でしょうか)$', '', s)
    return s
def txt(ids): return TK.decode(ids)
def stopped(g): t = txt(g); return ('。' in t) or ('\n' in t)
def greedy(ctx, maxlen=24, dlg=False, temp=0.0, seed=0, ban=True):
    rng = np.random.default_rng(seed); gen = []
    for step in range(maxlen):
        P = dist(ctx + gen).copy(); P[0] = 0
        if dlg and step < 2: P[2] = 0
        if not dlg: P[2] = 0
        if ban:
            for b in blocked(gen, 3): P[b] = 0
        if temp <= 0: x = int(P.argmax())
        else:
            idx = np.argsort(-P)[:40]; q = P[idx] ** (1 / temp); q = q / q.sum(); x = int(rng.choice(idx, p=q))
        if dlg and x == 2: break
        gen.append(x)
        if stopped(gen) and not dlg: break
    return txt(gen)
def beam(ctx, width=5, depth=10, topk=8):
    H = [(0.0, [])]
    for step in range(depth):
        C = []
        for lp, g in H:
            if g and stopped(g): C.append((lp, g)); continue
            P = dist(ctx + g).copy(); P[0] = 0; P[2] = 0
            idx = np.argsort(-P)[:topk]
            for x in idx: C.append((lp + float(np.log(P[x] + 1e-12)), g + [int(x)]))
        H = sorted(C, key=lambda a: -a[0])[:width]
        if all(g and stopped(g) for _, g in H): break
    return [(lp, txt(g)) for lp, g in H]
def ok(ans, acc): a = ans[:60]; return any(x in a for x in acc)
def evaluate(fn, name, qa=QA):
    t = time.time(); hit = 0; rows = []
    for q, acc in qa:
        a = fn(q); h = ok(a, acc); hit += h; rows.append((q, a.replace('\n', ' / ')[:50], h))
    P_('%s: %d/%d (%.0fs)' % (name, hit, len(qa), time.time() - t)); return rows
def show(rows): 
    for q, a, h in rows: P_(('○' if h else '×') + ' ' + q + ' → ' + a)
