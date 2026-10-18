# ai-lab
勾配なしの学習(数え上げ型の予測器)で、ウェブを自律的に巡回して学ぶ実験。**試作段階。**

- `forager.py` … 巡回+学習。robots.txt遵守、アクセス間隔、私的IP遮断、生の文章は保存しない(統計のみ)。
- `probe.py` … 取り置き文書での予測の成績、生成、会話ごっこ。
- `merge_states.py` … 複数ワーカーの学習結果を足し合わせる。
- `.github/workflows/forage.yml` … 手動起動。`minutes`(巡回分数)と`workers`(並列数)を指定。

## 注意
- GitHub Actions の利用規約(GitHub-hosted runner は「ソフトウェアプロジェクトの開発・テストに関連する用途」)に照らして、長時間・多並列の巡回は灰色。まずは短い試運転から。
- 連絡先は未設定。長期運用の前にリポジトリ変数 `FORAGER_CONTACT` を設定すること。
- 緊急停止: Actionsの実行をキャンセル / 変数 `FORAGER_ENABLED=false`。
