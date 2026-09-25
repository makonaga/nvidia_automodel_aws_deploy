# notebooks

SageMaker Training Job を起動し、成果物を確認する Notebook です。SageMaker Studio の JupyterLab で実行します。

| ファイル                           | 内容                                                                                                       |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `01_launch_training_job.ipynb` | セッション設定、実行構成（インスタンスの選択からバッチサイズとチェックポイント prefix を導出）、プリフライト確認、データのアップロード、Estimator の定義、ジョブの実行、成果物の確認と学習曲線 |
| `02_verify_adapter_inference.ipynb` | 学習したアダプタを HF transformers + PEFT でロードし、キーの一致と生成結果を確認する検証ジョブの起動と結果表示（`install_guide/07_inference_and_merge.md` ステップ1） |
| `03_merge_and_deploy_vllm.ipynb` | アダプタをマージした HF 形式モデルを作るジョブ、AWS vLLM DLC でのエンドポイントデプロイ、Chat Completions での呼び出し、削除（同ステップ2） |

実行手順は `install_guide/03_training_job.md`、8 GPU や Spot への展開は `install_guide/04_scale_and_operations.md`、アダプタの推論検証は `install_guide/07_inference_and_merge.md` を参照してください。

Estimator の `source_dir='../src'` と `dependencies=['../configs']` は Notebook の置き場所からの相対パスなので、`notebooks/` 配下のまま開いてください。  
実行時に生成される `artifacts/` は Git の管理対象外です。
