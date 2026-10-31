"""
Hugging Face のプライベートデータセットへフォルダを上げる。 トークンは環境変数 HF_TOKEN から (ファイルには書かない)。
使い方: HF_TOKEN=... python3 upload_hf.py '{"repo":"Youichi1/ai-lab-data","folder":"/tmp/pypi","path_in_repo":"code/pypi/part1"}'
"""
import os, sys, json
from huggingface_hub import HfApi
cfg = json.loads(sys.argv[1]); api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo(cfg['repo'], repo_type='dataset', private=True, exist_ok=True)
api.upload_folder(repo_id=cfg['repo'], repo_type='dataset', folder_path=cfg['folder'], path_in_repo=cfg['path_in_repo'], commit_message='add ' + cfg['path_in_repo'])
print('uploaded', cfg['folder'], '->', cfg['repo'], cfg['path_in_repo'])
