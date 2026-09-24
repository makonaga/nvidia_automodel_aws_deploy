# NeMo AutoModel on SageMaker 概要ガイド

このドキュメントでは、NVIDIA NeMo AutoModel を Amazon SageMaker Training Job 上で動かすプロジェクト全体の概要を説明します。  
他のガイドを読む前に、まずこのドキュメントで全体像を把握してください。

> **対象リージョン**: `us-west-2`（オレゴン）  
> 別リージョンで実施する場合は、コマンド中の `--region` や `REGION` の値を読み替えてください。ECR イメージ、SageMaker Studio、Training Job は同一リージョンに置く必要があります。

## このリポジトリのガイド構成

| ファイル                                       | 内容                                         |
| ------------------------------------------ | ------------------------------------------ |
| `00_overview.md`                           | 全体像とガイドの選び方（この文書）                          |
| `01_container_build.md`                    | コンテナイメージの作成と ECR への push                   |
| `02_local_verification.md`                 | ローカル GPU でのイメージと学習スクリプトの検証                 |
| `03_training_job.md`                       | SageMaker Studio からの Training Job 実行       |
| `04_scale_and_operations.md`               | 8 GPU、チェックポイント再開、Spot、本番モデルへの差し替え          |
| `05_configuration.md`                      | `train.py` のハイパーパラメータ、設定 YAML、データ形式、成果物の仕様 |
| `06_troubleshooting.md`                    | トラブルシューティング                                |
| `reference/01_design_rationale.md`         | 設計判断の根拠（AutoModel の起動方式、コンテナ方針の比較、落とし穴）    |
| `reference/02_composer_migration.md`       | MosaicML Composer 版 SFT からの移行設計と設定マッピング    |
| `reference/03_dlc_selection.md`            | ベース DLC イメージの選定                            |
| `reference/04_verification_log.md`         | 構築中に判明した問題・原因・判断と、実測値の記録                   |
| `reference/05_automodel_yaml_reference.md` | AutoModel 0.6.0 の設定 YAML パラメータ一覧           |

## NeMo AutoModel とは

NeMo AutoModel は、NVIDIA が公開している LLM / VLM の学習ライブラリです（https://github.com/NVIDIA-NeMo/Automodel）。  
Hugging Face Hub のモデルをそのまま読み込み、PyTorch の DTensor と FSDP2 を使って分散学習を行います。  
学習の内容は YAML ファイルで記述し、`automodel <config.yaml>` コマンド、または `torchrun` から起動します。

主な特徴は次のとおりです。

- 学習ループを書かずに、レシピ（`TrainFinetuneRecipeForNextTokenPrediction` など）と YAML で SFT / PEFT / VLM 学習を行える
- YAML の任意の項目を `--a.b.c=value` 形式のコマンドライン引数で上書きできる
- 外部の `torchrun` から起動されたことを検出すると、その場（in-process）でレシピを実行する
- チェックポイントは HF 互換の safetensors 形式で保存され、PEFT の場合は HF PEFT 形式のアダプタが出力される

## なぜ SageMaker Training Job で動かすのか

SageMaker Training Job は、指定したコンテナイメージと学習スクリプトを GPU インスタンス上で実行し、終了後にインスタンスを自動で解放するマネージドサービスです。  
このプロジェクトでは AutoModel を Training Job で動かすことで、次の状態を実現します。

インスタンスの確保と解放を SageMaker に任せられます。  
ジョブごとに `ml.g5.2xlarge` から `ml.p4de.24xlarge` までを選べ、課金は学習中のみです。

S3 上のモデルとデータをそのまま使えます。  
学習データとベースモデルは S3 の入力チャネルとして渡され、コンテナ起動前に `/opt/ml/input/data/` に配置されます。  
チェックポイントは `/opt/ml/checkpoints` と S3 の間で双方向に同期され、Spot 中断からの再開に使えます。

既存の MosaicML Composer 版 SFT ジョブと同じ運用に載ります。  
Notebook から Estimator を定義して `fit()` を呼ぶ流れ、CloudWatch でのログ確認、`model.tar.gz` の取得は同じです。  
Composer 版からの設定の対応関係は `reference/02_composer_migration.md` にまとめています。

## アーキテクチャと役割分担

Training Job の中では、3 つの層が次のように役割を分担します。

```
[ Notebook (SageMaker Studio) ]
        │  sagemaker.pytorch.PyTorch(image_uri=<自アカウントの ECR>, entry_point="train.py",
        │                            distribution={"torch_distributed": {"enabled": True}})
        │  .fit({"train": s3://..., "validation": s3://...})
        ▼
[ SageMaker Training Job ]
        │  ECR からカスタムコンテナを pull
        │  S3 の入力データを /opt/ml/input/data/<channel> に配置
        │  source_dir (src/ と configs/) を /opt/ml/code に展開
        ▼
[ sagemaker-training toolkit（DLC に同梱） ]
        │  torchrun --nnodes=1 --nproc_per_node=<GPU 数> train.py --<hyperparameter> <value> ...
        ▼
[ train.py（各 rank で 1 プロセス。このリポジトリの src/train.py） ]
        │  SageMaker の環境変数とパスを AutoModel の YAML 上書きに変換
        │  実効設定を書き出し、nemo_automodel.cli.app.main() を呼ぶ
        ▼
[ AutoModel InteractiveLauncher ]
        │  LOCAL_RANK と TORCHELASTIC_RUN_ID を検出 → in-process 実行（torchrun の二重起動なし）
        ▼
[ TrainFinetuneRecipeForNextTokenPrediction ]
           FSDP2 で学習
           checkpoint_dir → /opt/ml/checkpoints（checkpoint_s3_uri と自動同期）
           最終成果物     → /opt/ml/model（rank 0 が train.py でコピー → model.tar.gz として S3 へ）
```

| 層                          | 誰が提供するか                         | 役割                                                                                                             |
| -------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| sagemaker-training toolkit | AWS Deep Learning Container に同梱 | `hyperparameters` をコマンドライン引数に変換し、GPU 数に応じた `torchrun` でユーザースクリプトを起動する                                          |
| `train.py`                 | このリポジトリ                         | SageMaker の規約（チャネルのパス、`/opt/ml/checkpoints`、`/opt/ml/model`、`WORLD_SIZE`）を AutoModel の設定に写像する薄いアダプタ。学習ループは持たない |
| AutoModel                  | `pip install nemo-automodel`    | YAML とレシピに従って学習を実行する                                                                                           |

`train.py` と `configs/` は `source_dir` としてジョブごとにアップロードされるため、設定やスクリプトを変えてもコンテナの再ビルドは不要です。  
再ビルドが必要になるのは、AutoModel のバージョンや依存パッケージを変えるときだけです。

## 対象スコープ

このプロジェクトは次のスコープで設計し、検証しています。

| 項目         | 内容                                                              |
| ---------- | --------------------------------------------------------------- |
| モデル規模      | 7B〜13B 級の dense モデル（検証は `Qwen/Qwen3.5-0.8B` で実施）                |
| 学習の種類      | LoRA / PEFT による SFT                                             |
| 分散         | 単一ノード、多 GPU（FSDP2）。マルチノードと EFA は対象外                             |
| データ        | S3 の入力チャネル経由の JSONL                                             |
| ベース DLC    | `pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` |
| AutoModel  | 0.6.0                                                           |
| 検証済みインスタンス | `ml.g5.2xlarge`（1 GPU）、`ml.p4d.24xlarge`（8 GPU）                 |

## 主要な設計判断

構築中に決めた事項のうち、利用者が把握しておくべきものです。根拠と経緯は `reference/` の各ドキュメントを参照してください。

コンテナは AWS Deep Learning Container（DLC）を拡張します。  
DLC には SageMaker の toolkit、EFA、flash-attn、TransformerEngine が同梱されており、`pip install nemo-automodel` を追加するだけで済みます。  
NGC の AutoModel 公式イメージは数十 GB あり、非 root ユーザーや venv の摺り合わせも必要なため採用しませんでした（`reference/01_design_rationale.md`）。

Qwen3.5 の MTP（multi-token prediction）ヘッドは有効のまま学習します。  
vLLM / SGLang の投機的デコーディングで MTP ヘッドを使うため、本体と一緒に学習して整合を保つ方針です。  
AutoModel 0.6.0 の MTP はパディング付きバッチでは形状エラーになるため、packed sequence（`packing_strategy: neat`）と `causal-conv1d` を組み合わせます（`reference/04_verification_log.md` 2.2）。

packed sequence ではバッチサイズの単位が「サンプル」ではなく「pack（既定 2048 トークン）」になります。  
`global_batch_size` は `local_batch_size × GPU 数 × 勾配蓄積回数` で、Notebook の「実行構成」セルがインスタンスの選択から自動で導出します。

チェックポイントは `/opt/ml/checkpoints` に置き、`checkpoint_s3_uri` で S3 と同期します。  
AutoModel は同じディレクトリに前回のチェックポイントがあると自動で再開するため、モデルや PEFT の設定を変えるときは S3 の prefix（Notebook の `RUN_TAG`）を変えます。

## 全体の流れ

構築は次の順に進めます。所要時間はローカル PC の回線と GPU の有無で変わります。

| 順   | ガイド                          | 内容                                                                     | 目安                                                               |
| --- | ---------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1   | `01_container_build.md`      | ローカル PC で DLC を pull し、AutoModel を載せたイメージをビルドして ECR へ push する          | DLC の pull と push は回線次第。ビルドは causal-conv1d のコンパイルを含めて 15 分前後（実測） |
| 2   | `02_local_verification.md`   | ローカル GPU でスモークテスト、短い LoRA 学習、SageMaker 規約の模擬実行を行う。GPU が無ければスモークテストのみ   | 学習テストは 1 回 5 分前後（実測。初回はモデルの DL が加わる）                             |
| 3   | `03_training_job.md`         | SageMaker Studio の Notebook から `ml.g5.2xlarge` で最初の Training Job を実行する | ジョブは 8 分（実測）                                                     |
| 4   | `04_scale_and_operations.md` | 8 GPU、再開、Spot を確認し、本番モデルに差し替える                                         | 8 GPU のジョブは 14 分（実測。確保待ちを含む）。再開・Spot・本番差し替えは未実施                  |

設定の意味やデータ形式を確認したいときは `05_configuration.md`、問題が起きたときは `06_troubleshooting.md` を参照してください。

## 前提条件

AWS 環境として次が必要です。

- `us-west-2` で SageMaker Studio、ECR、S3、CloudWatch Logs を使える IAM 権限
- SageMaker 実行ロール（`AmazonSageMakerFullAccess` 相当）
- Service Quotas で対象インスタンスの `for training job usage` が 1 以上（検証したアカウントでは `ml.g5.2xlarge` が 4、`ml.p4d.24xlarge` も付与済みだった。無ければ申請が必要で、承認までの時間はアカウントによる）
- Training Job から Hugging Face Hub に到達できるネットワーク（Estimator に VPC を指定しない構成。検証はこの構成で行った）。到達できない場合はベースモデルを S3 に置き、`model` チャネルで渡す（`train.py` の `model` チャネル対応はローカルの模擬実行で確認済み、SageMaker 上では未実施）

ローカル PC として次が必要です。

- Docker Desktop または Docker Engine。空きディスク 60 GB 以上
- AWS CLI v2（`aws sts get-caller-identity` が通ること）
- 任意: NVIDIA GPU（`02_local_verification.md` の学習テスト用。24 GB 以上を推奨）

必要な知識として、SageMaker Training Job の基本（Estimator、入力チャネル、`model.tar.gz`）、Docker の基本操作、Hugging Face のモデルとチャットテンプレートの概念があると読み進めやすくなります。
