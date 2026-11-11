"""会話の長続き版: 索引の確率 P(w) (直前 23 語だけ見る) に、 会話の全体 (履歴) から 作る 特徴を 対数で足す。 数え上げのみ (勾配なし)。
  P'(w) = P(w) exp(s(w)) / Z,   s(w) = sum_f l[f][頻度の段(w)] * 特徴_f(w)
  特徴: 履歴に出た回数 (全体 / 直近 48 語 / 指数減衰) の log1p、 履歴の埋め込みの平均との近さ (全体 / 直近 100 語)、 帰納 (今の直前 1〜2 語と同じ並びが 履歴にあれば その続きの語)。
重み l は hdc/combo.py が 24M 語の索引で 数えた値 (combo_lam.npy の 先頭 9 行)。 100 億語の索引では 別に 数え直せる (fit_lam)。
使い方 (ローカルの 小さい索引で 動作確認): python3 chat2.py local <WORK>    /  Colab: from chat2 import Chat2, ..."""
import sys, os, numpy as np
V = 32768
FEATS = ['far', 'near', 'exp', 'simfar', 'simnear', 'ind1', 'ind2', 'indc2', 'indc3']


class Chat2:
    def __init__(self, dist, Em, uni, lam, CL64=None, tk=None):
        """dist(ctx) -> P[V] (索引の確率); Em: 語の埋め込み [V,48]; uni: 語の出現頻度 [V]; lam: [9,4] の重み; CL64: 語のクラス [V]"""
        self.dist, self.Em, self.lam, self.tk = dist, Em, np.asarray(lam), tk
        u = uni / uni.sum(); self.FB = np.digitize(u, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]); self.NB4 = np.array([0, 0, 1, 1, 2, 3])[self.FB]
        idf = np.log(1.0 / np.maximum(u, 1e-9)).astype(np.float32); self.idf = idf / idf.max(); self.CL = CL64

    def score(self, H):
        """履歴 H (語の列) から 各語への 加点 s(w) [V]"""
        H = np.asarray(H[-3000:], dtype=np.int64); s = np.zeros(V, np.float32); lam = self.lam[:, self.NB4]          # [9, V]
        if len(H) == 0: return s
        far = np.log1p(np.bincount(H, minlength=V)); near = np.log1p(np.bincount(H[-48:], minlength=V))
        ex = np.log1p(np.bincount(H, weights=0.5 ** ((len(H) - 1 - np.arange(len(H))) / 200.0), minlength=V))
        s += lam[0] * far + lam[1] * near + lam[2] * ex
        for f, seg in ((3, H), (4, H[-100:])):
            v = (self.Em[seg] * self.idf[seg][:, None]).sum(0); nv = np.linalg.norm(v)
            if nv > 0: s += lam[f] * (self.Em @ (v / nv))
        n = len(H)
        if n >= 4:
            t1 = H[3:]; m1 = H[2:-1] == H[-1]; m2 = m1 & (H[1:-2] == H[-2]); ms = [m1, m2]
            if self.CL is not None:
                cs = self.CL[H].astype(np.int64); c2 = (cs[2:-1] == cs[-1]) & (cs[1:-2] == cs[-2]); ms += [c2, c2 & (cs[:-3] == cs[-3])]
            for k, m in enumerate(ms):
                if m.any(): s += lam[5 + k] * np.log1p(np.bincount(t1[m], minlength=V))
        return s

    def next_dist(self, ctx, hist):
        P = self.dist(ctx).astype(np.float64); s = self.score(hist); e = P * np.exp(s - s.max()); return e / e.sum()

    def reply(self, utts, temp=0.7, topk=40, maxlen=80, seed=0, plain=False):
        """utts: 発話のリスト (文字列)。 文脈 = [0] + 各発話 + [2]。 plain=True なら 特徴を 足さない (比較用)"""
        rng = np.random.default_rng(seed); ctx = [0]
        for u in utts: ctx += self.tk.encode(u).ids + [2]
        gen = []
        for step in range(maxlen):
            full = ctx + gen
            P = self.dist(full) if plain else self.next_dist(full, full)
            P = np.asarray(P, np.float64).copy(); P[0] = 0
            if step < 2: P[2] = 0
            for i in range(len(gen) - 2):                                    # 同じ 3 語の並びの繰り返しを禁止
                if gen[i:i + 2] == gen[-2:]: P[gen[i + 2]] = 0
            idx = np.argsort(-P)[:topk]; q = P[idx] ** (1 / temp); q /= q.sum(); x = int(rng.choice(idx, p=q))
            if x == 2: break
            gen.append(x)
        return self.tk.decode(gen)


def local_dist(WORK, W):
    """動作確認用: ローカルの 小さい索引 (engine.Model) で 全語彙の確率を作る。 hdc/classlm2.py の dense_pa と同じ計算"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import engine as E
    M = E.Model({n: WORK + '/final_o%d' % n for n in (1, 2, 3, 4, 5)}, smooth='wb', lam={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 3.0}, kn_base=False)
    def dist(ctx):
        c = np.zeros(64, np.uint16); c[-min(len(ctx), 64):] = ctx[-64:]; pad = np.concatenate([c, np.zeros(1, np.uint16)]); ii = np.array([64]); P = M.uni_p.copy()
        for n in M.orders:
            if n == 1: continue
            h = E.ctx_hash(pad, ii, n - 1, n)[0]; t = M.t[n]; j = min(np.searchsorted(t['uc'], h), len(t['uc']) - 1)
            if t['uc'][j] != h: continue
            N = float(t['N'][j]); T = float(t['T'][j]); X = float(t['X'][j]); lo = np.searchsorted(t['k'], h << np.uint64(16)); hi = np.searchsorted(t['k'], (h + np.uint64(1)) << np.uint64(16))
            ws = (np.asarray(t['k'][lo:hi]) & np.uint64(0xFFFF)).astype(np.int64); cs = np.asarray(t['c'][lo:hi]).astype(np.float64); lm = M.lam[n]
            cnt = np.zeros(V); cnt[ws] = cs; P = (cnt + lm * (T + X) * P) / max(N - X + lm * (T + X), 1e-9)
        return P
    return dist


if __name__ == '__main__':
    from tokenizers import Tokenizer
    WORK = sys.argv[2]; tk = Tokenizer.from_file('/tmp/tokhf/tok/tok.json'); Em = np.load(WORK + '/E.npy'); lam = np.load(WORK + '/combo_lam.npy')
    uni = np.bincount(np.fromfile(WORK + '/sub.bin', dtype=np.uint16)[8:], minlength=V).astype(float) + 1; CL = np.load(WORK + '/cls64.npy')
    ch = Chat2(local_dist(WORK, None), Em, uni, lam, CL, tk)
    for utts in (['私の名前は田中太郎です。好きな食べ物はカレーです。', 'こんにちは。', '私の名前を覚えていますか？'], ['My name is Michael Faraday and I study electricity.', 'Nice to meet you!', 'What is my name?']):
        for plain in (True, False):
            print(('特徴なし' if plain else '特徴あり'), '|', utts[-1], '->', ch.reply(utts, plain=plain, seed=1)[:160].replace('\n', ' '))
