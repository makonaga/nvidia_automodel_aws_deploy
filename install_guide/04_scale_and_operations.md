# スケールと運用性の確認ガイド

このドキュメントでは、`03_training_job.md` で疎通した構成を、8 GPU での実行、チェックポイントからの再開、Spot インスタンス、本番モデルへの差し替えへ広げる手順を説明します。  
いずれも `notebooks/01_launch_training_job.ipynb` の「実行構成」セルと「3. Estimator」セルの値を変えて再実行するだけです。

> **対象リージョン**: `us-west-2`（オレゴン）

## 検証状況

| 項目                             | 検証状況                            |
| ------------------------------ | ------------------------------- |
| 8 GPU（`ml.p4d.24xlarge`、FSDP2） | 実施済み（2026-09-24）。課金 289 秒で完走    |
| チェックポイントからの再開                  | 未実施（手順は AutoModel の自動再開の仕様に基づく） |
| Spot インスタンス                    | 未実施                             |
| 本番モデルへの差し替え                    | 未実施（設定の換算は `train.py` に実装済み）    |

ステップ2 以降の「確認すること」に書いたログの文言は、AutoModel のソースと SageMaker の一般的な挙動から期待されるもので、本プロジェクトではまだ観測していません。

## 前提条件

- `03_training_job.md` が完了していること
- 使うインスタンスの Service Quotas が 1 以上であること（`ml.p4d.24xlarge for training job usage`、Spot の場合は `ml.g5.2xlarge for spot training job usage` など）

---

## ステップ1: 8 GPU（`ml.p4d.24xlarge`）で実行する

「実行構成」セルの `target` を `'p4d'` に変えて、セッション設定セルから「4. 実行」まで順に再実行します。  
GPU 数（8）、クォータ名（`ml.p4d.24xlarge for training job usage`）、`global_batch_size`（local 1 × 8 GPU = 8 pack）、`RUN_TAG`（`qwen35-0.8b-lora-mtp-p4d-v1`）がすべて連動します。  
`RUN_TAG` に `target` が含まれるため、`cheap` で作ったチェックポイントから自動再開されることはありません。

| target  | インスタンス             | GPU          | local_batch | global_batch |
| ------- | ------------------ | ------------ | ----------- | ------------ |
| `cheap` | `ml.g5.2xlarge`    | 1×A10G 24 GB | 1           | 4（勾配蓄積 4）    |
| `multi` | `ml.g5.12xlarge`   | 4×A10G 24 GB | 1           | 4            |
| `p4d`   | `ml.p4d.24xlarge`  | 8×A100 40 GB | 1           | 8            |
| `prod`  | `ml.p4de.24xlarge` | 8×A100 80 GB | 2           | 16           |

ログで確認することは次のとおりです。

- `torchrun --nnodes 1 --nproc_per_node 8` で起動している
- `rank 0/8` から `rank 7/8` の `sm_train` 行（各 rank が出力）と `World size: 8`
- モジュール名が `FSDPQwen3_5ForConditionalGeneration` / `FSDPQwen3_5DenseBlock` になっている（1 GPU では `World size ... is 1, skipping parallelization` と出て素の名前のまま）
- 17 pack / 8 なので 1 エポック 2 ステップ、3 エポックで 6 ステップの短いジョブになる（余りの 1 pack は捨てられる）

`ml.p4d.24xlarge` は在庫確保に時間がかかることがあり、実測では `waiting for capacity` に 6 分、インスタンス準備に 3.5 分かかりました。  
`waiting for capacity` が長引く場合はリージョン内の在庫不足が考えられます。検証では 6 分で確保できました。

実測では定常スループットが 21,000〜30,000 tokens/s（2,600〜3,800 /GPU）、VRAM が 14.1 GiB/GPU でした。  
/GPU の値が 1 GPU の A10G とほぼ同じで VRAM も減らないのは、データが小さすぎてオーバーヘッド支配であることと、0.8B モデルではパラメータより活性化メモリが支配的であるためです。  
8 GPU の効果は本番規模のデータと 7B 級モデルで評価してください。詳細は `reference/04_verification_log.md` 4.3 を参照してください。

`FSDP2` 特有の警告 `Model parameters are DTensors (FSDP2) — skipping fp32 parameter restoration ...` と `barrier(): using the device under current context` は無害です（`06_troubleshooting.md` 問題14）。

## ステップ2: チェックポイントから再開する

`checkpoint_s3_uri`（`RUN_TAG` の prefix）に残ったチェックポイントから AutoModel が自動再開することを確認します。  
費用を抑えるため `target = 'cheap'` で行います。

1. 1 回目: 「実行構成」セルを `target = 'cheap'` にして、3 エポック実行します。`RUN_TAG = qwen35-0.8b-lora-mtp-cheap-v1` の prefix に `epoch_2_step_14` まで溜まります
2. 2 回目: 「3. Estimator」セルの `'step_scheduler.num_epochs'` を `5` に変えて、同じ `RUN_TAG` のまま再実行します

2 回目のログで確認することは次のとおりです（期待値。`Loading checkpoint from ...` の文言はローカル検証で自動再開が起きたときに観測したもの）。

- 学習開始前に `Loading checkpoint from /opt/ml/checkpoints/epoch_2_step_14` が出る
- 最初の `step` 行が `step 15 | epoch 3` から始まる

同じ prefix に設定の違うチェックポイントがあると `Checkpoint key mismatch` や `TypeError: cannot pickle code objects` になります（`06_troubleshooting.md` 問題10）。  
再開の仕組みは Spot 中断からの復帰にも使われるため、この動作を先に確認しておきます。

## ステップ3: Spot インスタンスで実行する

「3. Estimator」セルの `PyTorch(...)` に次を追加します。

```python
    use_spot_instances=True,
    max_wait=2*3600,        # max_run 以上。Spot の確保待ち + 実行の上限
```

Spot 用のクォータは On-Demand とは別で、`ml.g5.2xlarge for spot training job usage` などを確認してください。

確認することは次のとおりです（期待値）。

- ジョブ詳細に Spot による節約額（Managed spot training savings）が表示される
- 中断が起きた場合は、再起動後に `Loading checkpoint from ...` が出て途中から続く

AutoModel の `StepScheduler` は既定で SIGTERM を受けるとチェックポイントを保存します（`step_scheduler.preemption_signal`）。  
SageMaker の Spot 中断時にコンテナへ SIGTERM が届き、保存が間に合うかは本プロジェクトでは未確認です。届かない場合でも、直近の `ckpt_every_steps` / エポック末のチェックポイントから再開されます。  
中断が起きなければ再開の経路はステップ2 で確認済みなので、Spot で完走すること自体を確認できれば十分です。

## ステップ4: 本番モデルに差し替える

疎通に使った `Qwen/Qwen3.5-0.8B` と料理データを、本番のモデルとデータに置き換えます。

1. データを `05_configuration.md` の形式（`prompt` と `output` を持つ JSONL）で用意し、学習用と検証用に分けて S3 に置きます。Notebook のセル「1. 学習データを S3 へ」のパスを書き換えます
2. ベースモデルを指定します。HF Hub なら `model_id`、S3 なら `model_s3`（非圧縮 prefix を推奨。`model.tar.gz` も可）
3. LoRA の設定を既存の学習設定に合わせます。rsLoRA を使っていた場合は `hyperparameters` に `'rslora_alpha': <alpha>` を追加すると、`train.py` が `peft.alpha = round(alpha × sqrt(peft.dim))` に換算します（`reference/02_composer_migration.md` 5.1）
4. `target = 'prod'`（`ml.p4de.24xlarge`、local batch 2）にします。7B 級での pack 長と `local_batch` は未検証で、0.8B での実測（pack 2048 で 14.5 GiB）から見て `local_batch: 1` から始めて OOM を見ながら上げてください
5. `RUN_TAG` はモデルごとに変えます（実行構成セルの文字列を編集）

既存の Composer 版と精度を比較する場合は、損失マスク（`answer_only_loss_mask`）、チャットテンプレートのレンダリング、LoRA のスケールの 3 点を先に揃えてください。  
切り分けの順序は `reference/02_composer_migration.md` 8 章にまとめています。

## 完了の確認

- 8 GPU で `World size: 8` と FSDP2 のシャーディングを確認できた
- 同じ `RUN_TAG` での再実行で `Loading checkpoint from ...` から続きが学習された
- Spot で完走した
- 本番モデルとデータで val loss が下がる

## 次のステップ

配信のために LoRA をベースモデルにマージした HF 形式のチェックポイントが必要です。  
Qwen3.5 で MTP ヘッドを含めてマージするには専用のツールが必要で、このリポジトリではまだ提供していません（`reference/04_verification_log.md` 2.2 と 5 章）。
