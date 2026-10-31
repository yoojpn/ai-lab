"""spill 済みの runs ディレクトリから 併合だけをやり直す (途中で失敗した場合の再開)。 次数ごとに併合 -> そのランを消す。
使い方: python3 resume_merge.py '{"dir":"/tmp/wiki","orders":[2,3,4,5]}'"""
import sys, os, re, json, time, shutil
import engine as E


def main(cfg):
    t0 = time.time(); log = lambda s: print('[%5ds] %s' % (time.time() - t0, s), flush=True)
    W = cfg['dir']; rd = W + '/runs'; MINC = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}; MINC.update({int(k): v for k, v in cfg.get('minc', {}).items()})
    for n in cfg['orders']:
        rs = sorted(set(int(m.group(1)) for f in os.listdir(rd) for m in [re.match(r'o%d_r(\d+)_k\.npy$' % n, f)] if m))
        runs = [('%s/o%d_r%d_k.npy' % (rd, n, r), '%s/o%d_r%d_c.npy' % (rd, n, r)) for r in rs]
        E.merge(runs, W + '/final_o%d' % n, MINC[n], 20_000_000, log)
        for a, b in runs: os.remove(a); os.remove(b)
    shutil.rmtree(rd, ignore_errors=True); log('完了')


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
