"""使い方: LADDER_CORPUS=... python3 auto.py <施策番号> <訓練語数N>   → 最後に RESULT {json} を出す"""
import sys, json, time
import lab, items

if __name__ == '__main__':
    k, N = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    ctx = lab.load(N)
    lab.log('データ準備 %.0fs' % (time.time() - t0))
    res = items.ITEMS[k](ctx)
    res['item'] = k; res['N'] = N; res['sec'] = time.time() - t0
    print('RESULT ' + json.dumps(res, ensure_ascii=False, default=float), flush=True)
