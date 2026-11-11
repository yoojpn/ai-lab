def ens_beam(prefix_fns, width=6, depth=14, topk=10, norm=0.5):
    def f(q):
        ctxs = [TK.encode(p(q)).ids for p in prefix_fns]; H = [(0.0, [])]
        for step in range(depth):
            C = []
            for lp, g in H:
                if g and stopped(g): C.append((lp, g)); continue
                L = 0
                for c in ctxs:
                    P = dist(c + g).copy(); P[0] = 0; P[2] = 0; P[QM] = 0
                    if step == 0: P[BAN1] = 0
                    P = P / P.sum(); L = L + np.log(P + 1e-9)
                L = L / len(ctxs)
                for b in blocked(g, 3): L[b] = -1e9
                for x in np.argsort(-L)[:topk]: C.append((lp + float(L[x]), g + [int(x)]))
            H = sorted(C, key=lambda a: -(a[0] / (len(a[1]) ** norm)))[:width]
            if all(g and stopped(g) for _, g in H): break
        return txt(H[0][1])
    return f
FR2 = {'q原文': lambda q: q, 'q+答え：': lambda q: q + '答え：', 'q+A.': lambda q: q + 'A.', 'q+回答': lambda q: q + '回答'}
