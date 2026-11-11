"""FM-index 上の 質問応答 (数え上げのみ)。 質問を 文頭 (「X は」) に直して 続きを ビーム探索で出す (事実の質問)。 あいさつなどは 対話方式 (fm_chat.reply)。
使い方: python3 fm_qa.py SHARDLIST LAMBDA_JSON -- 質問1 質問2 ...      (LAMBDA_JSON は fm_eval_result.json の val_ja_web の lam を {"lam": [...]} にしたもの)
カーネルなどで使うなら: from fm_qa import QA_Engine; e = QA_Engine(fm); e.ask('日本の首都は？')"""
import re, sys, os, json
import numpy as np
import fm_chat
from fm_chat import TK, blocked

QM = TK.encode('？').ids[0]
_BANSTR = ['？', '?', '、', '。', '，', '．', '.', ',', '\n', '（', '(', '「', '」', '　', ' ', 'どこ', 'どれ', 'いつ', '誰', 'だれ', '何', 'こちら', 'とは', 'です', 'ですか', 'って', 'も', 'に', 'を', 'が', 'の', 'は', 'と', 'で', 'や']
BAN1 = np.zeros(32768, bool)
for _s in _BANSTR:
    _ids = TK.encode(_s).ids
    if len(_ids) == 1: BAN1[_ids[0]] = True
for _i in range(32768):
    _t = TK.decode([_i]).strip()
    if _t == '' or _t[0] in '？?、。,.!！」）)・-—…:：;；#|/\\*': BAN1[_i] = True


def to_cloze(q):
    """「日本の首都はどこですか？」 -> 「日本の首都は」"""
    s = re.sub(r'[？?！!。]+$', '', q.strip())
    s = re.sub(r'(は|が)?(どこ|何県|何区|何日|何|誰|だれ|いつ|どれ)(にある|にいる|ですか|でしょうか|か)?$', lambda m: m.group(1) or '', s)
    return re.sub(r'(ですか|でしょうか)$', '', s)


def is_question(q):
    """事実を聞く質問か (対話方式との 切り替え)"""
    q = q.strip(); c = to_cloze(q)
    return bool(re.search(r'[？?]$|(ですか|でしょうか|教えて)$', q)) and c.endswith(('は', 'が')) and len(c) >= 4 and not re.search(r'(好き|趣味|名前|元気|調子|おすすめ|どう|疲れ)', q)


class QA_Engine:
    def __init__(self, fm, scale=0.3):
        self.fm = fm; self.cache = {}; fm.lam = [l * scale for l in fm.lam]

    def dist(self, ctx):
        k = tuple(int(x) for x in ctx[-23:])
        if k not in self.cache:
            if len(self.cache) > 20000: self.cache.clear()
            self.cache[k] = self.fm.dist(list(k)).astype(np.float32)
        return self.cache[k]

    @staticmethod
    def stopped(g): t = TK.decode(g); return ('。' in t) or ('\n' in t)

    def factual(self, q, width=6, depth=14, topk=10, norm=0.5):
        ctx = TK.encode(to_cloze(q)).ids; H = [(0.0, [])]
        for step in range(depth):
            C = []
            for lp, g in H:
                if g and self.stopped(g): C.append((lp, g)); continue
                P = self.dist(ctx + g).copy(); P[0] = 0; P[2] = 0; P[QM] = 0
                if step == 0: P[BAN1] = 0
                for b in blocked(g, 3): P[b] = 0
                P = P / P.sum()
                for x in np.argsort(-P)[:topk]: C.append((lp + float(np.log(P[x] + 1e-12)), g + [int(x)]))
            H = sorted(C, key=lambda a: -(a[0] / (len(a[1]) ** norm)))[:width]
            if all(g and self.stopped(g) for _, g in H): break
        a = TK.decode(H[0][1]); a = re.split(r'ですが|であるが|であり|だが|[。\n、]', a)[0].strip()   # 最初の 1 節で切る
        return a

    def chat(self, hist, seed=0):
        old = self.fm.lam; self.fm.lam = [l / 0.3 for l in old]      # 対話は もとの lam
        try: return fm_chat.reply(self.fm, hist, seed=seed)
        finally: self.fm.lam = old

    def ask(self, q, hist=None, seed=0):
        return self.factual(q) if is_question(q) else self.chat((hist or []) + [q], seed)


if __name__ == '__main__':
    a = sys.argv[1:]; sl = a[0]; lam = json.load(open(a[1]))['lam']; qs = a[3:] if len(a) > 2 and a[2] == '--' else []
    e = QA_Engine(fm_chat.FM(sl, lam))
    for q in qs: print('あなた: %s\nAI: %s' % (q, e.ask(q)))
