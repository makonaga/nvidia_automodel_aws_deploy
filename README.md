# NeMo AutoModel on Amazon SageMaker

このリポジトリは、NVIDIA の LLM 学習ライブラリ NeMo AutoModel を Amazon SageMaker Training Job 上で実行するためのガイドとコード一式を提供します。  
AWS Deep Learning Container に AutoModel を載せたカスタムコンテナ、SageMaker に AutoModel の設定を橋渡しする学習スクリプト、ジョブを起動する Notebook、動作確認用のサンプルデータで構成されます。

## プロジェクト概要

NeMo AutoModel は、Hugging Face Hub のモデルを PyTorch の FSDP2 で分散学習するライブラリです（https://github.com/NVIDIA-NeMo/Automodel）。  
学習の内容は YAML で記述し、学習ループを書く必要がありません。

このプロジェクトでは AutoModel を SageMaker Training Job で動かすことで、次の状態を実現します。

GPU インスタンスの確保と解放を SageMaker に任せられます。  
ジョブごとに `ml.g5.2xlarge` から `ml.p4de.24xlarge` までを選べ、課金は学習中のみです。

S3 上のモデルとデータをそのまま使えます。  
学習データとベースモデルは入力チャネルとして渡され、チェックポイントは S3 と自動で同期されます。

既存の MosaicML Composer 版 SFT ジョブと同じ運用に載ります。  
Notebook から Estimator を定義して `fit()` を呼ぶ流れ、CloudWatch でのログ確認、`model.tar.gz` の取得は同じです。

## アーキテクチャ概要

Training Job の中では、AWS の Deep Learning Container に同梱された `sagemaker-training` toolkit が、GPU 数に応じた `torchrun` でこのリポジトリの `train.py` を起動します。  
`train.py` は SageMaker の環境変数とパス（入力チャネル、`/opt/ml/checkpoints`、`/opt/ml/model`）を AutoModel の YAML 設定に写像し、AutoModel の CLI に処理を委譲します。  
AutoModel は外部の `torchrun` から起動されたことを検出してその場でレシピを実行するため、`torchrun` の二重起動はありません。

学習は FSDP2 による単一ノード多 GPU で行い、LoRA アダプタを HF PEFT 形式で出力します。  
チェックポイントは `/opt/ml/checkpoints` と S3 の間で双方向に同期され、Spot 中断からの再開や、同じ prefix での続きの学習に使えます。

コンテナは AWS Deep Learning Container `pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` をベースに、AutoModel 0.6.0 と Qwen3.5 に必要なカーネル（`flash-linear-attention`、`causal-conv1d`）を追加したものです。
※他のモデルへの対応は順次対応予定です。  

`train.py` と設定 YAML はジョブごとに `source_dir` としてアップロードされるため、設定の変更でコンテナを再ビルドする必要はありません。

## 前提条件

このプロジェクトを実施する前に、以下の前提条件を満たしていることを確認してください。

必要なツールとして  

- Docker Desktop または Docker Engine（空きディスク 60 GB 以上）
- AWS CLI v2 が設定済みであること
- 任意: NVIDIA GPU（ローカル検証用。24 GB 以上を推奨）  

が必要です。

AWS 環境の要件として  

- `us-west-2` で SageMaker Studio、ECR、S3、CloudWatch Logs を使える権限があること
- SageMaker 実行ロール（`AmazonSageMakerFullAccess` 相当）があること
- 対象インスタンスの Service Quotas（`ml.g5.2xlarge for training job usage` など）が 1 以上であること
- Training Job から Hugging Face Hub に到達できること（到達できない場合はベースモデルを S3 から渡す）  

が必要です。

必要な知識として  

- SageMaker Training Job の基本（Estimator、入力チャネル、`model.tar.gz`）
- Docker の基本操作
- Hugging Face のモデルとチャットテンプレートの概念  

が推奨されます。

## クイックスタート

**適切な開始点を選択してください。**

一から構築する場合は、まず **install_guide/00_overview.md** で全体像と役割分担を把握してください。  
次に **install_guide/01_container_build.md** に従ってコンテナイメージをビルドし、ECR へ push してください。  
GPU のあるローカル PC では **install_guide/02_local_verification.md** で学習まで確認してから、GPU が無ければスモークテストだけ確認してから進みます。  
その後 **install_guide/03_training_job.md** に従って SageMaker Studio から最初の Training Job を実行してください。  
問題が発生した場合は **install_guide/06_troubleshooting.md** を参照してください。

ECR に既にイメージがある場合は、**install_guide/03_training_job.md** から始められます。

自分のモデルとデータで学習する場合は、**install_guide/05_configuration.md** でデータ形式とハイパーパラメータを確認し、**install_guide/04_scale_and_operations.md** のステップ4 に従って差し替えてください。  
既存の Composer 版の学習設定を移植する場合は **install_guide/reference/02_composer_migration.md** を先に読んでください。

## ディレクトリ構成

このリポジトリは以下の構成になっています。

**install_guide** ディレクトリには、詳細な手順書が含まれています。

| ファイル                         | 内容                                                                |
| ---------------------------- | ----------------------------------------------------------------- |
| `00_overview.md`             | 全体像、アーキテクチャと役割分担、スコープ、主要な設計判断                                     |
| `01_container_build.md`      | コンテナイメージの作成と ECR への push                                          |
| `02_local_verification.md`   | ローカル GPU でのスモークテスト、短い LoRA 学習、SageMaker 規約の模擬実行                   |
| `03_training_job.md`         | SageMaker Studio からの Training Job 実行                              |
| `04_scale_and_operations.md` | 8 GPU、チェックポイント再開、Spot、本番モデルへの差し替え                                 |
| `05_configuration.md`        | `train.py` のハイパーパラメータ、設定 YAML、データ形式、成果物                           |
| `06_troubleshooting.md`      | トラブルシューティング                                                       |
| `07_inference_and_merge.md`  | 学習したアダプタの推論、マージ、vLLM 配信（検証中）                            |
| `reference/`                 | 設計判断の根拠、Composer 版からの移行設計、DLC の選定、検証記録、AutoModel 設定 YAML のパラメータ一覧 |

**container** ディレクトリには、コンテナイメージの定義とスクリプトが含まれています。`Dockerfile`、依存の pin（`constraints.txt`）、ビルドと push を行う `build_and_push.sh`、ローカル検証用の `smoke_test.sh`、`local_train_test.sh`、`local_sm_sim.sh` です。

**src** ディレクトリには、SageMaker のエントリポイント `train.py` が含まれています。Training Job の `source_dir` としてアップロードされます。`src/inference/` はアダプタ推論の検証ジョブ用で、`peft` を追加インストールする `requirements.txt` を持つため学習用とは分けています。

**configs** ディレクトリには、AutoModel の設定 YAML が含まれています。`sagemaker/` が Training Job 用、`local/` がローカル検証用です。Training Job では `dependencies` として `src` と一緒にアップロードされます。

**notebooks** ディレクトリには、Training Job を起動して成果物を確認する `01_launch_training_job.ipynb`、学習したアダプタを HF でロードして生成を検証する `02_verify_adapter_inference.ipynb`、アダプタをマージして AWS の vLLM DLC でエンドポイントに配信する `03_merge_and_deploy_vllm.ipynb` が含まれています。SageMaker Studio の JupyterLab で実行します。

**data** ディレクトリには、動作確認用のサンプルデータ（料理の基礎知識に関する日本語の instruction データ 306 件）と、その生成スクリプトが含まれています。形式は `data/README.md` を参照してください。

README.md ファイルは、このファイルです。プロジェクトの概要と開始方法を説明しています。

## 重要な注意事項

このプロジェクトを実施する前に、以下の重要事項を必ず理解してください。

チェックポイントの自動再開について、AutoModel は `checkpoint_s3_uri` の prefix に前回のチェックポイントがあると、設定が非互換でも自動で再開しようとします。  
モデルや LoRA の設定を変えて再実行するときは、Notebook の `RUN_TAG` を変えて prefix を分けるか、`fresh_start` を `1` にしてください。  
Notebook の `RUN_TAG` は選択したインスタンスの `target` を含むため、インスタンスを変えただけで prefix が分かれます。

Qwen3.5 の MTP ヘッドについて、このリポジトリは MTP ヘッドを有効にしたまま学習する構成です（vLLM / SGLang の投機的デコーディングで使うため）。  
AutoModel 0.6.0 の MTP はパディング付きバッチでは動かないため、設定 YAML は packed sequence を前提にしています。バッチサイズの単位が「サンプル」ではなく「pack」になる点に注意してください。

コストについて、`ml.p4d.24xlarge` は 1 時間 30 USD 超、`ml.g5.2xlarge` は 1 時間 1.5 USD 前後です（`us-west-2` の概算。最新の料金表で確認してください）。動作確認は `ml.g5.2xlarge` で行い、JupyterLab space は作業後に停止してください。

機密情報について、gated モデルを使う場合の `HF_TOKEN` は Notebook にベタ書きせず、環境変数から渡してください。Git リポジトリにコミットしないよう注意してください。

本番環境への適用について、このガイドは動作確認と開発時の利用を前提としています。  
**本番のモデル学習に使う場合は、既存モデルとの精度比較（損失マスク、チャットテンプレート、LoRA のスケールの整合）、コスト見積もり、チェックポイントの保持方針を必ず確認してください。**

配信用のチェックポイントについて、出力される成果物は LoRA アダプタです。  
vLLM / SGLang で MTP を使って配信するには、LoRA を本体と MTP ヘッドにマージした HF 形式のチェックポイントが必要で、このリポジトリではまだツールを提供していません（`install_guide/reference/04_verification_log.md` 2.2）。

## ガイド一覧

各ガイドの詳細は以下の通りです。

概要ガイド（install_guide/00_overview.md）では、NeMo AutoModel とは何か、なぜ SageMaker Training Job で動かすのか、toolkit と `train.py` と AutoModel の役割分担、対象スコープ、主要な設計判断、全体の流れを説明しています。  
すべての作業者がまず最初に読むべきドキュメントです。

コンテナイメージ作成ガイド（install_guide/01_container_build.md）では、AWS Deep Learning Container を pull し、AutoModel 0.6.0 と Qwen3.5 用のカーネルを載せたイメージをビルドして ECR へ push する手順を説明しています。  
ビルド時の自動検証の見方、ビルドオプション、再ビルドが必要になる条件も含みます。

ローカル検証ガイド（install_guide/02_local_verification.md）では、ビルドしたイメージをローカル PC で検証する 3 つのステップ（スモークテスト、短い LoRA 学習、SageMaker 規約の模擬実行）を説明しています。  
SageMaker にジョブを投げる前に、インスタンス課金なしで問題を切り分けられます。

Training Job 実行ガイド（install_guide/03_training_job.md）では、SageMaker Studio の JupyterLab から Notebook を実行し、`ml.g5.2xlarge` で最初のジョブを完走させるまでを説明しています。  
Notebook の各セルの意味、ログの見どころ、成果物の確認方法、後片付けを含みます。

スケールと運用性の確認ガイド（install_guide/04_scale_and_operations.md）では、8 GPU（`ml.p4d.24xlarge`）での実行、チェックポイントからの再開、Spot インスタンス、本番モデルへの差し替えの手順を説明しています。  
8 GPU は実施済みで、再開と Spot と本番差し替えは手順のみで未実施です。

設定リファレンス（install_guide/05_configuration.md）では、`train.py` のハイパーパラメータ、SageMaker の規約と AutoModel 設定の対応、設定 YAML の各セクションの意図、学習データの形式、`model.tar.gz` の内容、CloudWatch メトリクスを説明しています。

推論とマージのガイド（install_guide/07_inference_and_merge.md）では、学習した LoRA アダプタを HF transformers + PEFT でロードして生成する検証（実施済み。本体のアダプタはすべて一致し、MTP ヘッド分だけが未使用になることを確認）、マージ済みモデルの作成と AWS の vLLM DLC でのエンドポイント配信（実施済み）、MTP ヘッドを含めたマージ（未作成）の手順をまとめます。

トラブルシューティングガイド（install_guide/06_troubleshooting.md）では、コンテナのビルド、ローカル検証、Training Job で発生する問題と解決方法、ログに出る無害なメッセージの一覧をまとめています。  
問題が発生した際に参照してください。

設計判断の根拠（install_guide/reference/01_design_rationale.md）では、実装前に AutoModel のソースを読んで確認した起動方式と設定の仕組み、コンテナ方針（DLC 拡張と NGC 公式イメージ）の比較、想定した落とし穴と対策を記録しています。

Composer 版からの移行設計（install_guide/reference/02_composer_migration.md）では、既存の MosaicML Composer 版 SFT スクリプトとの対応関係、起動方式の変更、設定マッピング表、rsLoRA や `enable_thinking` などの機能ギャップと対策、精度が揃わない場合の切り分け順を説明しています。

ベース DLC の選定（install_guide/reference/03_dlc_selection.md）では、AutoModel 0.6.0 の要求と各 DLC の比較、Python 3.13 での wheel 互換性の検証、フォールバック案を説明しています。

AutoModel 設定 YAML リファレンス（install_guide/reference/05_automodel_yaml_reference.md）では、AutoModel 0.6.0 の LLM 微調整レシピが読み取る YAML の全セクションとパラメータ、既定値、このリポジトリでの設定値を、上流のソースを調査して一覧にしています。  
上流のドキュメントにはパラメータ一覧が無いため、設定を変えるときの一次資料として使えます。

検証記録（install_guide/reference/04_verification_log.md）では、構築中に発生した問題とその原因、判断の経緯（依存衝突、MTP の形状エラー、チェックポイントの自動再開など）と、ローカル GPU、`ml.g5.2xlarge`、`ml.p4d.24xlarge` での実測値（スループット、VRAM、所要時間、loss）を記録しています。

## ライセンスと参照

このプロジェクトのコードは、NVIDIA NeMo AutoModel（Apache License 2.0）と AWS Deep Learning Containers を利用しています。それぞれのライセンスに従ってください。

参考にした主なドキュメントは以下の通りです。NeMo AutoModel（https://github.com/NVIDIA-NeMo/Automodel）、AWS Deep Learning Containers（https://github.com/aws/deep-learning-containers）、SageMaker Python SDK の PyTorch Estimator（https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html）、SageMaker Training Toolkit（https://github.com/aws/sagemaker-training-toolkit）です。

## バージョン履歴

このリポジトリのバージョン履歴を記録します。

### バージョン1.0.0（2026-09-25）

**主な内容:**

- AWS Deep Learning Container をベースにした NeMo AutoModel 0.6.0 のコンテナイメージ（Qwen3.5 用の `flash-linear-attention` と `causal-conv1d` を同梱）
- SageMaker の規約と AutoModel の設定を橋渡しする `train.py`（入力チャネル、チェックポイント同期、バッチの整合、rsLoRA の alpha 換算、ウォームアップの算出、成果物の出力）
- Qwen3.5 の MTP ヘッドを有効にしたまま LoRA SFT を行う設定 YAML（packed sequence）
- SageMaker Studio から Training Job を起動し成果物を確認する Notebook（インスタンスの選択からバッチサイズとチェックポイント prefix を導出）
- 動作確認用の日本語 instruction データ 306 件
- `ml.g5.2xlarge`（1 GPU）と `ml.p4d.24xlarge`（8 GPU、FSDP2）での完走を確認
- 構築ガイド、設定リファレンス、AutoModel 設定 YAML のパラメータ一覧、トラブルシューティング、設計判断と検証の記録

---
