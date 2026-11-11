BANSTR = ['？', '?', '、', '。', '，', '．', '.', ',', '\n', '（', '(', '「', '」', '　', ' ', 'どこ', 'どれ', 'いつ', '誰', 'だれ', '何', 'こちら', 'とは', 'です', 'ですか', 'って', 'も', 'に', 'を', 'が', 'の', 'は', 'と', 'で', 'や']
BAN1 = np.zeros(32768, bool)
for s in BANSTR:
    ids = TK.encode(s).ids
    if len(ids) == 1: BAN1[ids[0]] = True
for i in range(32768):
    t = TK.decode([i]).strip()
    if t == '' or t[0] in '？?、。,.!！」）)・-—…:：;；#|/\\*': BAN1[i] = True
def cloze_dec(q, maxlen=24, first_ban=True, ban_all_q=True):
    ctx = TK.encode(to_cloze(q)).ids; gen = []
    for step in range(maxlen):
        P = dist(ctx + gen).copy(); P[0] = 0; P[2] = 0
        if step == 0 and first_ban: P[BAN1] = 0
        if ban_all_q: P[TK.encode('？').ids[0]] = 0
        for b in blocked(gen, 3): P[b] = 0
        gen.append(int(P.argmax()))
        if stopped(gen): break
    return txt(gen)
