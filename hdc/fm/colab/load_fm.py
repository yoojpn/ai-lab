import subprocess, sys, os, json, time, threading
subprocess.run('pkill -f chatd.py; pkill -f fmeng; sleep 2', shell=True)
sys.path.insert(0, '/content/ev/code/fm/code')
os.environ['TOKJSON'] = '/content/ev/tokhf/tok/tok.json'; os.environ['FMENG'] = '/content/ev/code/fm/code/fmeng'
import fm_chat
LAMS = json.load(open('/content/ev/res/results/fm_eval_result.json')); LOADED = [False]
def _ld():
    global fm
    fm = fm_chat.FM('/content/ev/list.txt', LAMS['val_ja_web']['lam']); fm.dist([0, 100]); LOADED[0] = True
threading.Thread(target=_ld, daemon=True).start(); print('loading')
