QM = TK.encode('？').ids[0]
def beam_dec(prefix_fn, width=6, depth=14, topk=10, norm=0.0):
    def f(q):
        ctx = TK.encode(prefix_fn(q)).ids; H = [(0.0, [])]
        for step in range(depth):
            C = []
            for lp, g in H:
                if g and stopped(g): C.append((lp, g)); continue
                P = dist(ctx + g).copy(); P[0] = 0; P[2] = 0; P[QM] = 0
                if step == 0: P[BAN1] = 0
                for b in blocked(g, 3): P[b] = 0
                P = P / P.sum()
                for x in np.argsort(-P)[:topk]: C.append((lp + float(np.log(P[x] + 1e-12)), g + [int(x)]))
            H = sorted(C, key=lambda a: -(a[0] / (len(a[1]) ** norm)))[:width]
            if all(g and stopped(g) for _, g in H): break
        return txt(H[0][1])
    return f
