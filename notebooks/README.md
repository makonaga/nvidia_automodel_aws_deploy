# SageMaker Studio からの動作確認手順（初回 Training Job）

SageMaker Studio の JupyterLab から `01_launch_training_job.ipynb` を実行し、ECR に push 済みのイメージで最初の SageMaker Training Job を回す手順です。
初回は `ml.g5.2xlarge`（1×A10G 24 GB）で `Qwen/Qwen3.5-0.8B` + 料理データ（245 件）を 3 エポック学習し、
toolkit の torchrun 起動・チャネル写像・チェックポイント同期・`model.tar.gz` の出力を確認します。

所要時間の目安: 25〜35 分（インスタンス起動とイメージ pull 8〜12 分、モデル DL 1〜2 分、学習 5〜10 分、成果物アップロード 1 分）。
課金はインスタンス起動〜終了の間のみ（`ml.g5.2xlarge` は us-west-2 で 1 時間 1.5 USD 前後）。

## 前提

| 項目 | 内容 |
|---|---|
| イメージ | `290918126236.dkr.ecr.us-west-2.amazonaws.com/nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130`（`container/README.md` Step 6 で push 済み） |
| リージョン | **us-west-2**。Studio ドメインと ECR、Training Job は同一リージョンである必要がある |
| クォータ | Service Quotas > Amazon SageMaker > `ml.g5.2xlarge for training job usage` が 1 以上。0 なら申請（承認に数時間〜1 日） |
| 実行ロール | Studio のユーザープロファイルに紐づく実行ロール（Notebook 内で `sagemaker.get_execution_role()` が返すもの）。`AmazonSageMakerFullAccess` 相当があれば自アカウント ECR からの pull、既定バケットの読み書き、CloudWatch Logs が通る |
| ネットワーク | Training Job が HF Hub に到達できること（Estimator に VPC を指定しない、network isolation なし）。Studio ドメインが VPC-only モードでも Training Job には影響しない |
| Notebook 実行環境 | SageMaker Studio の JupyterLab（SageMaker Distribution イメージ、`ml.t3.medium` で十分）。`sagemaker` SDK は同梱で追加インストール不要 |

## Step 1: Studio で JupyterLab を起動する

1. コンソール > Amazon SageMaker AI > Studio（us-west-2）を開き、ユーザープロファイルで **Open Studio**
2. 左メニュー **Applications > JupyterLab** > **Create JupyterLab space**（既存の space があればそれを使う）
   - Instance: `ml.t3.medium`（Notebook 自体は軽い。GPU 不要）
   - Image: `SageMaker Distribution`（最新版）
   - Storage: 既定の 5 GB で足りる（成果物 `model.tar.gz` は数十 MB）
3. **Run space** → **Open JupyterLab**

## Step 2: リポジトリを取得する

JupyterLab で **File > New > Terminal** を開き:

```bash
cd ~
git clone https://github.com/makonaga/nvidia_automodel_aws_deploy.git
cd nvidia_automodel_aws_deploy
git checkout claude/automodel-sagemaker-integration-w5kdwn
```

- `sagemaker` SDK は SageMaker Distribution に同梱されているので追加インストールは不要（バージョンとリージョンは Step 4 のセル 1 で表示される）
- リポジトリが private の場合、`git clone` で GitHub のユーザー名と Personal Access Token（`repo` スコープ）を聞かれる。
  代わりにローカル PC で `git archive -o repo.zip HEAD` を作り JupyterLab の Upload ボタンで持ち込んでも構わない
- Studio のホームディレクトリ（`/home/sagemaker-user`）は space を停止しても保持される

## Step 3: Notebook を開く

左のファイルブラウザで `nvidia_automodel_aws_deploy/notebooks/01_launch_training_job.ipynb` を開き、
カーネルに **Python 3 (ipykernel)** を選びます。
Estimator の `source_dir='../src'` / `dependencies=['../configs']` は Notebook の置き場所からの相対パスなので、
**必ず `notebooks/` 配下の Notebook をそのまま開いてください**（別の場所へコピーしない）。

## Step 4: セル 1（セッション設定）を実行

確認すること:

- `region us-west-2` と表示される（違う場合は Studio ドメインのリージョンが違うので、us-west-2 の Studio で開き直す）
- `image_uri` が `290918126236.dkr.ecr.us-west-2.amazonaws.com/nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130` になっている
- `bucket` は既定の `sagemaker-us-west-2-290918126236`。既存のデータバケットを使う場合はここで書き換える（実行ロールがそのバケットを読み書きできること）
- Studio では `role` は自動取得される。`SAGEMAKER_ROLE` の設定は不要

## Step 5: セル「0. プリフライト確認」を実行

- `ECR image : 0.6.0-pt2.10-py313-cu130 9.6 GB (compressed)` と出れば ECR 側は OK
- `quota : ml.g5.2xlarge for training job usage = 1.0` 以上であること。`0.0` なら
  コンソールの Service Quotas から引き上げ申請し、承認まで待つ（このステップ以降は進めない）
- `ecr:DescribeImages` や `servicequotas:ListServiceQuotas` で `AccessDenied` が出る場合は
  Notebook 側の認証情報の権限不足。ジョブ自体には影響しないので、コンソールで目視確認して次へ進んでよい

## Step 6: セル「1. 学習データを S3 へ」を実行

`train.jsonl` / `val.jsonl` が `s3://<bucket>/automodel/cooking_basics/{train,validation}/` にアップロードされ、
S3 URI が 2 行表示されます。

## Step 7: セル「2. ベースモデルの渡し方」を実行

初回は既定のまま（`model_id = 'Qwen/Qwen3.5-0.8B'`、`model_s3 = None`）で HF Hub から直接取得します。

## Step 8: セル「3. Estimator」を実行

`target = 'cheap'` のまま実行します。エラーが出なければ Estimator が作られるだけで、まだジョブは始まりません。
`hyperparameters` は初回は変更不要です（3 エポック、global batch 4 pack、local batch 1、lr 1e-4、`LOWEST_VAL` を出力）。

## Step 9: セル「4. 実行」を実行し、ログを追う

`estimator.fit(...)` でジョブが作成され、CloudWatch のログが Notebook に流れます。ログの見どころを順に:

| 段階 | 目安 | 期待するログ |
|---|---|---|
| 起動 | 0〜2 分 | `Starting - Starting the training job...` → `Downloading - Downloading input data` |
| イメージ pull | 5〜10 分 | `Training - Training image download completed. Training in progress.`（ここまでログが止まって見えるのは正常） |
| toolkit | 直後 | `Invoking script with the following command:` に続いて `torchrun --nnodes 1 --nproc_per_node 1 ... train.py --config qwen3_5_cooking_lora.yaml --final_checkpoint LOWEST_VAL ...` |
| train.py | 直後 | `rank 0/1 \| config=... \| sagemaker=True`、`dataset ← /opt/ml/input/data/train/train.jsonl`、`model ← Qwen/Qwen3.5-0.8B (--model_id)`、`checkpoint_dir ← /opt/ml/checkpoints`、`packing 見積もり: samples=245, avg_tokens=127, pack_size=2048 → packs≈17`、`lr_warmup_steps ← 4`、`=== effective config ===` |
| モデル DL | 1〜2 分 | `Fetching 13 files` の進捗。`Warning: You are sending unauthenticated requests` は public モデルなので無視 |
| サンプル確認 | 直後 | `=== Sample Prompt 0 (rendered with chat template) ===` に `<\|im_start\|>user ... <\|im_start\|>assistant\n<think>\n\n</think>\n\n` の形が出る |
| 学習 | 5〜10 分 | 最初のステップは Triton コンパイルで 1〜2 分かかる。以降 `step N \| epoch E \| loss ... \| grad_norm ... \| lr ... \| tps ...` が 1 エポック 4〜5 ステップで進み、エポック末に `[val] ... loss ...` と `Saving checkpoint` |
| 終了 | 1 分 | `最終チェックポイント: ... (LOWEST_VAL)` → `成果物を /opt/ml/model へコピーしました` → `Reporting training SUCCESS` → `Uploading - Uploading generated training model` → `Completed` |

`[ERROR] ... is part of ...'s signature, but not documented` は transformers の docstring チェックで無害です（`docs/03` 3 章）。

## Step 10: セル「5. 成果物の確認」を実行

- `status : Completed` と `model : s3://.../output/model.tar.gz`
- 展開後の一覧に `model/adapter_model.safetensors`、`model/adapter_config.json`、`tokenizer/`、`config.yaml`、`effective_config.yaml`、`training.jsonl`、`validation.jsonl`、`training_info.json` が含まれる
- `training_info.json` の `final_checkpoint` が `epoch_*_step_*` で、`selection` が `LOWEST_VAL`
- 続くセルで学習曲線（train / val の loss）を表示。val loss が下がっていれば疎通確認は完了

## Step 11: S3 側の確認（任意）

```bash
aws s3 ls s3://sagemaker-us-west-2-290918126236/automodel/cooking_basics/checkpoints/qwen35-0.8b-lora-mtp-v1/ --region us-west-2
```

`epoch_*_step_*/` と `LATEST`/`LOWEST_VAL` が同期されていれば、次回同じ `RUN_TAG` で実行したときにここから自動再開されます。
モデルや PEFT 設定を変えて再実行するときは `RUN_TAG` を変えるか `fresh_start: 1` にしてください。

## Step 12: 後片付け

- Training Job はインスタンス終了で課金が止まる（`Completed` / `Failed` / `Stopped` になっていれば追加費用なし）
- JupyterLab space は起動中は課金されるため、作業を終えたら **Applications > JupyterLab** で **Stop space**（ホームディレクトリは保持される）
- 途中で止めたいときはコンソール > Training > Training jobs > 該当ジョブ > **Stop**、または Notebook で `estimator.latest_training_job.stop()`

## 失敗したときに共有するもの

1. Notebook に流れたログの、`Invoking script with the following command:` から最後のエラーまで
2. ログが長い場合は JupyterLab のターミナルで CloudWatch から取得し、生成された `job.log` をダウンロードして共有:

```bash
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <job_name> --region us-west-2 --since 2h > job.log
```

   コンソール > Training > Training jobs > 該当ジョブ > Monitor > **View logs** からも同じログを見られます。

3. `describe_training_job` の `FailureReason`（コンソールの Training jobs > 該当ジョブ > Status にも表示）

## よくある失敗

| 症状 | 原因と対処 |
|---|---|
| `ResourceLimitExceeded ... ml.g5.2xlarge for training job usage` | クォータ 0。Service Quotas で申請 |
| `CannotPullContainerError` / `no basic auth credentials` | 実行ロールに ECR の pull 権限が無い、またはリージョン違い。`AmazonSageMakerFullAccess` を付与し、ECR と同一リージョンで実行 |
| `Downloading` の後に `AccessDenied` (S3) | 実行ロールがバケットを読めない。既定バケット以外を使うときはロールのポリシーに追加 |
| `Fetching 13 files` で `Connection error` / `Max retries` | ジョブから HF Hub に出られない。VPC 設定を外すか、`model_s3` で S3 から渡す |
| `torch.OutOfMemoryError` | `local_batch_size: 1` でも A10G 24 GB に収まらない場合は `packed_sequence.packed_sequence_size: 1024` と `dataset.seq_length: 512` に下げる（`hyperparameters` に追加） |
| `Checkpoint key mismatch` / `TypeError: cannot pickle code objects` | 同じ `RUN_TAG` の S3 prefix に設定の違うチェックポイントが残っている。`RUN_TAG` を変えるか `fresh_start: 1` |
| ログが `Training in progress` のまま 15 分以上進まない | イメージ pull（9.6 GB）中。20 分を超える場合はコンソールでジョブを Stop し、ECR のリージョンとタグを再確認 |
