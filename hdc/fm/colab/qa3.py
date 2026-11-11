import threading, io, contextlib
def frame_dec(prefix_fn, maxlen=24):
    def f(q):
        ctx = TK.encode(prefix_fn(q)).ids; gen = []
        for step in range(maxlen):
            P = dist(ctx + gen).copy(); P[0] = 0; P[2] = 0
            if step == 0: P[BAN1] = 0
            P[TK.encode('？').ids[0]] = 0
            for b in blocked(gen, 3): P[b] = 0
            gen.append(int(P.argmax()))
            if stopped(gen): break
        return txt(gen)
    return f
FR = {
 'cloze': lambda q: to_cloze(q),
 'QA1': lambda q: '質問：' + q + '\n回答：',
 'QA2': lambda q: 'Q. ' + q + '\nA. ',
 'Q改行': lambda q: q + '\n',
 'wiki': lambda q: to_cloze(q) + ' ',
 'は、': lambda q: to_cloze(q) + '、',
}
def run_exp(path, jobs):
    fo = open(path, 'w'); OUT[0] = fo
    try:
        for name, fn, scale in jobs:
            fm.lam = [l * scale for l in BASE_LAM]; CACHE.clear()
            rows = evaluate(fn, '%s lam x%g' % (name, scale)); show(rows)
        P_('ALLDONE')
    except Exception as e: P_('ERR %r' % e)
    finally: OUT[0] = None; fo.close()
BASE_LAM = list(fm.lam)
