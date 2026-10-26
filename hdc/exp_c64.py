"""候補の再ランク(出力を全語→少数の候補にする案)の前提: 索引が出す候補に正解が入る率(recall)。
 候補 = 数える模型(4-gram+バックオフ)の上位K語 ∪ 文書内で直近に出た語。  使い方: python3 exp_c64.py '{"N":16000000,"nq":3000}'
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lab
from lab import V, log
import exp_scale as es
from exp_g import Chain
from exp_h import chain_dist
es.CACHE = os.environ.get('LADDER_CORPUS', es.CACHE)


def run(cfg):
    t0 = time.time(); ctx = lab.load(cfg['N']); nq = cfg.get('nq', 3000)
    ch = Chain(ctx.Xtr, ctx.ytr, [tuple(range(1, k + 1)) for k in range(5)], 0.9)
    uni = np.bincount(ctx.ytr, minlength=V); utop = np.argsort(-uni)
    X, y = ctx.Xte[:nq], ctx.yte[:nq]; Ks = (8, 16, 64, 256)
    rec = {K: 0 for K in Ks}; recu = {K: 0 for K in Ks}; recd = {K: 0 for K in Ks}; size = {K: 0.0 for K in Ks}; recent_hit = 0
    for i in range(nq):
        D = chain_dist(ch, X[i]); order = np.argsort(-D)
        doc = ctx.te_i[ctx.mte[i]]; p = int(ctx.pte[i]); rw = set(doc[max(0, p - 300):p].tolist()); recent_hit += int(y[i] in rw)
        for K in Ks:
            top = set(order[:K].tolist())
            rec[K] += int(y[i] in top); recu[K] += int(y[i] in set(utop[:K].tolist()))
            un = top | rw; recd[K] += int(y[i] in un); size[K] += len(un)
    res = dict(cfg=cfg, recall_chain={K: rec[K] / nq for K in Ks}, recall_unigram={K: recu[K] / nq for K in Ks},
               recall_chain_plus_recent={K: recd[K] / nq for K in Ks}, avg_union_size={K: size[K] / nq for K in Ks},
               recent_300_hit=recent_hit / nq, sec=time.time() - t0)
    log('RESULT ' + json.dumps(res))
    for k in ('recall_unigram', 'recall_chain', 'recall_chain_plus_recent', 'avg_union_size'): log('%s %s' % (k, res[k]))
    log('直近300語に正解がある率 %.3f' % res['recent_300_hit'])


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
