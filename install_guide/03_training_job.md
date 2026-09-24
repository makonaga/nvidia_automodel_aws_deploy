# Training Job 実行ガイド

このドキュメントでは、SageMaker Studio の JupyterLab から `notebooks/01_launch_training_job.ipynb` を実行し、ECR に push したイメージで最初の SageMaker Training Job を実行する手順を説明します。  
初回は `ml.g5.2xlarge`（1×A10G 24 GB）で `Qwen/Qwen3.5-0.8B` と料理データ（245 件）を 3 エポック学習し、toolkit の torchrun 起動、チャネルの写像、チェックポイントの同期、`model.tar.gz` の出力を確認します。

> **対象リージョン**: `us-west-2`（オレゴン）  
> Studio ドメイン、ECR イメージ、Training Job は同一リージョンである必要があります。

## 検証状況

| 項目 | 内容 |
| --- | --- |
| 実施日 | 2026-09-24 |
| インスタンス | `ml.g5.2xlarge`（1×A10G 24 GB） |
| 結果 | 完走。課金 443 秒。起動とイメージ pull 4 分、モデル DL 16 秒、学習 3 分（3 エポック 15 ステップ）、成果物アップロード 30 秒 |
| loss | train 3.28 → 2.41、val 2.56 → 2.45 → 2.42 |

## 前提条件

| 項目 | 内容 |
| --- | --- |
| イメージ | `01_container_build.md` で `nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130` が ECR に push 済み |
| クォータ | Service Quotas > Amazon SageMaker > `ml.g5.2xlarge for training job usage` が 1 以上。0 なら申請する（承認に数時間〜1 日） |
| 実行ロール | Studio のユーザープロファイルに紐づく実行ロール（Notebook 内で `sagemaker.get_execution_role()` が返すもの）。`AmazonSageMakerFullAccess` 相当があれば、自アカウント ECR からの pull、既定バケットの読み書き、CloudWatch Logs が通る |
| ネットワーク | Training Job が Hugging Face Hub に到達できること（Estimator に VPC を指定しない）。Studio ドメインが VPC-only モードでも Training Job には影響しない |
| Notebook 実行環境 | SageMaker Studio の JupyterLab（SageMaker Distribution イメージ、`ml.t3.medium` で十分）。`sagemaker` SDK は同梱 |

課金はインスタンスの起動から終了までで、`ml.g5.2xlarge` は 1 時間 1.5 USD 前後です。

---

## ステップ1: Studio で JupyterLab を起動する

1. コンソール > Amazon SageMaker AI > Studio（`us-west-2`）を開き、ユーザープロファイルで **Open Studio** を選びます
2. 左メニューの **Applications > JupyterLab** から **Create JupyterLab space** を選びます（既存の space があればそれを使います）
   - Instance: `ml.t3.medium`（Notebook 自体は軽く、GPU は不要です）
   - Image: `SageMaker Distribution`（最新版）
   - Storage: 既定の 5 GB で足ります（成果物の `model.tar.gz` は数十 MB）
3. **Run space** の後に **Open JupyterLab** を選びます

## ステップ2: リポジトリを取得する

JupyterLab で **File > New > Terminal** を開き、次を実行します。

```bash
cd ~
git clone https://github.com/makonaga/nvidia_automodel_aws_deploy.git
```

- `sagemaker` SDK は SageMaker Distribution に同梱されているため、追加のインストールは不要です
- リポジトリが private の場合、`git clone` で GitHub のユーザー名と Personal Access Token（`repo` スコープ）を求められます。代わりにローカル PC で `git archive -o repo.zip HEAD` を作り、JupyterLab の Upload で持ち込んでも構いません
- Studio のホームディレクトリ（`/home/sagemaker-user`）は space を停止しても保持されます

## ステップ3: Notebook を開く

左のファイルブラウザで `nvidia_automodel_aws_deploy/notebooks/01_launch_training_job.ipynb` を開き、カーネルに **Python 3 (ipykernel)** を選びます。  
Estimator の `source_dir='../src'` と `dependencies=['../configs']` は Notebook の置き場所からの相対パスなので、必ず `notebooks/` 配下の Notebook をそのまま開いてください。

Notebook のセル構成は次のとおりです。

| セル | 内容 |
| --- | --- |
| セッション設定 | `role`、`region`、`bucket`、`image_uri` を取得する |
| 実行構成 | `target` でインスタンスを選ぶ。GPU 数、クォータ名、`global_batch_size`、`RUN_TAG` がここから導出される |
| 0. プリフライト確認 | ECR イメージ、クォータ、実行ロールを確認する |
| 1. 学習データを S3 へ | `train.jsonl` と `val.jsonl` をアップロードする |
| 2. ベースモデルの渡し方 | HF Hub の `model_id` か S3 の `model_s3` を選ぶ |
| 3. Estimator | `hyperparameters`、`metric_definitions`、`PyTorch` Estimator を定義する |
| 4. 実行 | `estimator.fit()` でジョブを起動し、ログを流す |
| 5. 成果物の確認 | `model.tar.gz` を取得して展開し、学習曲線を表示する |

## ステップ4: セッション設定セルを実行する

次を確認します。

- `region us-west-2` と表示される。違う場合は Studio ドメインのリージョンが違うので、`us-west-2` の Studio で開き直します
- `image_uri` が `<account-id>.dkr.ecr.us-west-2.amazonaws.com/nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130` になっている
- `bucket` は既定の `sagemaker-us-west-2-<account-id>`。既存のデータバケットを使う場合はここで書き換えます（実行ロールがそのバケットを読み書きできること）

先頭に出る `sagemaker.config INFO - Not applying SDK defaults from location: ...` は、SDK の既定設定ファイルが無いという情報表示で無害です。

## ステップ5: 実行構成セルとプリフライト確認セルを実行する

実行構成セルは `target = 'cheap'` のまま実行します。  
`ml.g5.2xlarge x1 | world_size=1 | local_batch=1 | global_batch=1 pack` と `RUN_TAG qwen35-0.8b-lora-mtp-cheap-v1` が表示されます。

`global_batch_size` は `local_batch × GPU 数 × instance_count × grad_accum` で導出されます。  
1 GPU で勾配蓄積をしたい場合は `grad_accum = 4` のように指定します。

続けてプリフライト確認セルを実行します。

- `ECR image : 0.6.0-pt2.10-py313-cu130 9.6 GB (compressed)` と出れば ECR 側は問題ありません
- `quota : ml.g5.2xlarge for training job usage = 1.0` 以上であること。`0.0` ならコンソールの Service Quotas から引き上げを申請し、承認まで待ちます
- Studio の実行ロールには通常 `servicequotas:ListServiceQuotas` が無く、`quota : 参照権限なし (AccessDeniedException)` と表示されます。ジョブ自体には影響しないので、コンソールの Service Quotas > AWS services > Amazon SageMaker で `ml.g5.2xlarge for training job usage` を検索し、Applied quota value が 1 以上であることを目視で確認して次へ進みます

## ステップ6: 学習データを S3 へアップロードする

セル「1. 学習データを S3 へ」を実行します。  
`train.jsonl` と `val.jsonl` が `s3://<bucket>/automodel/cooking_basics/{train,validation}/` にアップロードされ、S3 URI が 2 行表示されます。

自分のデータを使う場合の形式は `05_configuration.md` を参照してください。

## ステップ7: ベースモデルの渡し方を選ぶ

セル「2. ベースモデルの渡し方」は初回は既定のまま（`model_id = 'Qwen/Qwen3.5-0.8B'`、`model_s3 = None`）で実行し、HF Hub から直接取得します。

Training Job が HF Hub に到達できない環境や、S3 上の自前モデルを使う場合は `model_s3` に S3 の非圧縮 prefix または `model.tar.gz` を指定します。  
`train.py` が `model` チャネルを解決し、tar.gz はノードごとに 1 回だけ展開します。

## ステップ8: Estimator を定義する

セル「3. Estimator」をそのまま実行します。エラーが出なければ Estimator が作られるだけで、まだジョブは始まりません。  
初回は `hyperparameters` を変更する必要はありません（3 エポック、lr 1e-4、`LOWEST_VAL` を出力。バッチは実行構成セルの導出値）。

Estimator の要点は次のとおりです。

| 引数 | 値 | 意味 |
| --- | --- | --- |
| `image_uri` | ECR のイメージ | `01_container_build.md` で push したもの |
| `entry_point` / `source_dir` / `dependencies` | `train.py` / `../src` / `['../configs']` | `/opt/ml/code` に展開される |
| `distribution` | `{'torch_distributed': {'enabled': True}}` | toolkit が GPU 数に応じた `torchrun` で起動する |
| `checkpoint_s3_uri` | `s3://<bucket>/automodel/cooking_basics/checkpoints/<RUN_TAG>/` | `/opt/ml/checkpoints` と双方向同期。同じ prefix があると AutoModel が自動再開する |
| `metric_definitions` | `train:loss`、`eval:loss` など | AutoModel のログ形式に合わせた正規表現。CloudWatch のメトリクスとして記録される |
| `environment` | `HF_HOME`、`HF_TOKEN`、`NCCL_DEBUG`、`WANDB_MODE` | gated モデルを使うときだけ `HF_TOKEN` を環境変数から渡す |

## ステップ9: ジョブを実行し、ログを追う

セル「4. 実行」で `estimator.fit(...)` を呼ぶと、ジョブが作成され、CloudWatch のログが Notebook に流れます。  
ログの見どころを順に示します。

| 段階 | 目安 | 期待するログ |
| --- | --- | --- |
| 起動 | 0〜2 分 | `Starting - Starting the training job...` → `Downloading - Downloading input data` |
| イメージ pull | 3〜5 分 | `Training - Training image download completed. Training in progress.` ここまでログが止まって見えるのは正常です |
| toolkit | 直後 | `Invoking script with the following command:` に続いて `torchrun --nnodes 1 --nproc_per_node 1 train.py --config qwen3_5_cooking_lora.yaml --final_checkpoint LOWEST_VAL ...` |
| train.py | 直後 | `rank 0/1 \| config=... \| sagemaker=True`、`dataset ← /opt/ml/input/data/train/train.jsonl`、`model ← Qwen/Qwen3.5-0.8B (--model_id)`、`checkpoint_dir ← /opt/ml/checkpoints`、`packing 見積もり: samples=245, avg_tokens=127, pack_size=2048 → packs≈17`、`lr_warmup_steps ← N`、`=== effective config ===` |
| モデル DL | 1 分未満 | `Fetching 13 files` の進捗。`Warning: You are sending unauthenticated requests` は public モデルなので無視して構いません |
| サンプル確認 | 直後 | `=== Sample Prompt 0 (rendered with chat template) ===` に `<\|im_start\|>user ... <\|im_start\|>assistant` と空の `<think>` ブロックが出る |
| 学習 | 3〜5 分 | 最初のステップは Triton コンパイルで約 100 秒、最初の検証も約 40 秒かかります。以降は `step N \| epoch E \| loss ... \| grad_norm ... \| lr ... \| mem ... \| tps ...` が 2 秒間隔で進み、エポック末に `[val] ... loss ...` と `Saving checkpoint` が出ます |
| 終了 | 1 分 | `最終チェックポイント: ... (LOWEST_VAL)` → `成果物を /opt/ml/model へコピーしました` → `Reporting training SUCCESS` → `Uploading - Uploading generated training model` → `Completed` |

`[ERROR] ... is part of ...'s signature, but not documented` という行が複数出ますが、transformers の docstring チェックの表示で無害です（`06_troubleshooting.md` 問題13）。  
CloudWatch には `httpx` の HF Hub アクセスログも大量に出るため、学習ログを読むときは `step ` や `[val]` でフィルタすると見やすくなります。

## ステップ10: 成果物を確認する

セル「5. 成果物の確認」を実行します。

- `status : Completed` と `model : s3://.../output/model.tar.gz` が表示される
- 展開後の一覧に `model/adapter_model.safetensors`、`model/adapter_config.json`、`tokenizer/`、`config.yaml`、`effective_config.yaml`、`training.jsonl`、`validation.jsonl`、`training_info.json` が含まれる
- `training_info.json` の `checkpoint` が `epoch_*_step_*` で、`selected_by` が `LOWEST_VAL`
- 続くセルで学習曲線（train と val の loss）が表示され、val loss が下がっている

成果物の各ファイルの意味は `05_configuration.md` を参照してください。

## ステップ11: S3 側のチェックポイントを確認する（任意）

```bash
aws s3 ls s3://<bucket>/automodel/cooking_basics/checkpoints/qwen35-0.8b-lora-mtp-cheap-v1/ --region us-west-2
```

`epoch_*_step_*/` と `LATEST`、`LOWEST_VAL` が同期されていれば、次回同じ `RUN_TAG` で実行したときにここから自動で再開されます。  
モデルや PEFT の設定を変えて再実行するときは `RUN_TAG` を変えるか、`hyperparameters` の `fresh_start` を `1` にしてください。

## ステップ12: 後片付け

- Training Job はインスタンスの終了で課金が止まります。`Completed`、`Failed`、`Stopped` のいずれかになっていれば追加費用はありません
- JupyterLab space は起動中に課金されるため、作業を終えたら **Applications > JupyterLab** で **Stop space** を選びます（ホームディレクトリは保持されます）
- ジョブを途中で止めるときは、コンソール > Training > Training jobs > 該当ジョブ > **Stop**、または Notebook で `estimator.latest_training_job.stop()` を実行します

## 完了の確認

次がすべて確認できれば、SageMaker 上での疎通は完了です。

- ログに `torchrun --nnodes 1 --nproc_per_node 1 train.py ...` と `Reporting training SUCCESS` が出ている
- `model.tar.gz` に `model/adapter_model.safetensors` と `training_info.json` が含まれている
- val loss がエポックごとに下がっている

## 次のステップ

`04_scale_and_operations.md` で 8 GPU、チェックポイントからの再開、Spot を確認し、本番モデルに差し替えてください。  
問題が起きた場合は `06_troubleshooting.md` を参照してください。
