"""μP風(隠れ層の学習率を 128/d 倍)の検証: 幅 d を変えたとき最適な学習率が動くか。標準(SP)と比べる。
使い方: python3 exp_mup.py '{"N":8000000,"d":128}'  → lrを4通り × (SP, muP風) を測り、最良lrを表示"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_y
from lab import log
cfg0 = json.loads(sys.argv[1]); out = {}
for mup in (0, 1):
    for lr in (4e-3, 8e-3, 1.6e-2, 3.2e-2):
        c = dict(cfg0, lr=lr, mup=mup, epochs=1)
        import io, contextlib
        try:
            exp_y.run(c)
        except Exception as e:
            log('失敗 %s' % e)
log('RESULT-MUP-DONE')
