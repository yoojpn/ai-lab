"""FM-index (複数分割) の bit/token を 検証データで測る。 Witten-Bell の後ろ向き平滑化 (次数ごとの lam は 前半で選び、 後半で測る)。
使い方: python3 fm_eval.py SHARDLIST VAL1.bin[,VAL2.bin...] [N_QUERY=30000] [ORDERS=1,2,3,4,5,6,8,10,12,16,20,24] [THREADS=8]
shardlist.txt の各行: <bwt.xz> <meta.json>"""
import sys, os, subprocess, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.environ.get('FMENG', HERE + '/fmeng')
SL = sys.argv[1]; VALS = sys.argv[2].split(','); NQ = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
ORD = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '1,2,3,4,5,6,8,10,12,16,20,24').split(',')]; TH = sys.argv[5] if len(sys.argv) > 5 else '8'
W = os.environ.get('WORK', '/tmp'); V = 32768


def queries(v, nq, seed=0):
    rng = np.random.default_rng(seed); p = rng.integers(8, len(v) - 1, int(nq * 1.1)); p = p[(v[p] != 0)][:nq]; p = np.sort(p)
    Q = np.full((len(p), 24), -1, np.int32)
    for i, q in enumerate(p):
        c = v[max(0, q - 23):q]; Q[i, 23 - len(c):23] = c; Q[i, 23] = v[q]
    return Q


def run(Q, tag):
    qf = '%s/q_%s.bin' % (W, tag); of = '%s/o_%s.bin' % (W, tag); Q.tofile(qf)
    r = subprocess.run([ENG, SL, ','.join(map(str, ORD)), 'batch', qf, of, TH], capture_output=True, text=True); sys.stderr.write(r.stderr[-400:])
    if r.returncode: raise SystemExit('fmeng 失敗')
    return np.fromfile(of, dtype=np.int64).reshape(len(Q), len(ORD), 3)


def wb(out, lam, uni=None):
    """out: (問, 次数, 3) -> 正解語の確率 (WB の後ろ向き)"""
    n = out.shape[0]; p = np.full(n, 1.0 / V)
    for oi in range(len(ORD)):
        N = out[:, oi, 0].astype(float); T = out[:, oi, 1].astype(float); c = out[:, oi, 2].astype(float); m = N > 0; l = lam[oi]
        p = np.where(m, (c + l * T * p) / np.maximum(N + l * T, 1e-9), p)
    return p


def tune(out):
    lam = [4.0] * len(ORD); grid = [0.3, 0.6, 1, 1.5, 2, 3, 4.5, 6, 9, 14, 22, 40]
    for it in range(2):
        for oi in range(len(ORD)):
            best = None
            for g in grid:
                l = list(lam); l[oi] = g; b = -np.log2(np.maximum(wb(out[0::2], l), 1e-12)).mean()
                if best is None or b < best[0]: best = (b, g)
            lam[oi] = best[1]
    return lam


if __name__ == '__main__':
    allres = {}
    for vf in VALS:
        v = np.fromfile(vf, dtype=np.uint16).astype(np.int32); tag = os.path.basename(vf).replace('.bin', '')
        if len(v) < 1000: print('%s: 短すぎ (%d 語)' % (tag, len(v))); continue
        Q = queries(v, NQ); t = time.time(); out = run(Q, tag); lam = tune(out)
        b2 = -np.log2(np.maximum(wb(out[1::2], lam), 1e-12)).mean(); b1 = -np.log2(np.maximum(wb(out[0::2], lam), 1e-12)).mean()
        cov = {ORD[oi]: float((out[:, oi, 0] > 0).mean()) for oi in range(len(ORD))}
        print('FMEVAL %-18s %7d 問: %.4f bit/token (奇数番の問で評価; 調整に使った偶数番 %.4f) lam=%s 被覆=%s [%.0fs]' % (tag, len(Q), b2, b1, lam, {k: round(x, 3) for k, x in cov.items()}, time.time() - t), flush=True)
        allres[tag] = {'bits': float(b2), 'n': int(len(Q)), 'lam': lam, 'coverage': cov}
    json.dump(allres, open(W + '/fm_eval_result.json', 'w'), ensure_ascii=False)
