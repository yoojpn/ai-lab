"""Actions 用: HF の生の会話データ -> トークン化 (tok_dlg) -> n-gram 表 (dlg_build, 次数を 1 つずつ) -> HF の tok/ に置く。
使い方: HF_TOKEN=... python3 prep_tok.py '{"pattern":"dialog/*","dest":"tok"}'   (pattern を絞ると小さな試運転ができる)"""
import os, sys, json, glob, shutil
from huggingface_hub import HfApi, snapshot_download
import tok_dlg, dlg_build

cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
REPO = 'Youichi1/ai-lab-data'; tok = os.environ['HF_TOKEN']; api = HfApi(token=tok)
pats = cfg.get('patterns', ['dialog/*'])
snapshot_download(REPO, repo_type='dataset', allow_patterns=pats, local_dir='/tmp/hfraw', token=tok)
os.makedirs('/tmp/dlg', exist_ok=True)
for f in glob.glob('/tmp/hfraw/dialog/*.jsonl.gz'): shutil.copy(f, '/tmp/dlg/')
print('生データ', len(glob.glob('/tmp/dlg/*.jsonl.gz')), 'ファイル', flush=True)
shutil.rmtree('/tmp/hfraw', ignore_errors=True)
tok_dlg.main({'in': '/tmp/dlg', 'out': '/tmp/tok', 'vocab': 32768}); shutil.rmtree('/tmp/dlg', ignore_errors=True)
dlg_build.main({'dir': '/tmp/tok', 'serial': True})
for f in glob.glob('/tmp/tok/runs*'): shutil.rmtree(f, ignore_errors=True)
dest = cfg.get('dest', 'tok')
api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path='/tmp/tok', path_in_repo=dest, commit_message='tokenized dialogue + 5-gram tables (%s)' % dest)
print('アップロード完了 ->', dest)
