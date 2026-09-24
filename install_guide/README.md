# install_guide

NeMo AutoModel を Amazon SageMaker Training Job で動かすためのガイド集です。

まず `00_overview.md` で全体像と役割分担を確認してください。

## ファイル一覧

| ファイル | 内容 | 検証状況 |
| --- | --- | --- |
| `00_overview.md` | 全体像、アーキテクチャと役割分担、スコープ、主要な設計判断 | — |
| `01_container_build.md` | DLC ベースのコンテナイメージの作成と ECR への push | 実施済み（ローカル Linux PC、us-west-2） |
| `02_local_verification.md` | ローカル GPU でのスモークテスト、短い LoRA 学習、SageMaker 規約の模擬実行 | 実施済み（RTX 3090 24 GB） |
| `03_training_job.md` | SageMaker Studio からの Training Job 実行 | 実施済み（`ml.g5.2xlarge`） |
| `04_scale_and_operations.md` | 8 GPU、チェックポイント再開、Spot、本番モデルへの差し替え | 8 GPU は実施済み（`ml.p4d.24xlarge`）。再開・Spot・本番差し替えは**未実施** |
| `05_configuration.md` | `train.py` のハイパーパラメータ、設定 YAML、データ形式、成果物 | — |
| `06_troubleshooting.md` | トラブルシューティング | — |
| `reference/01_design_rationale.md` | 設計判断の根拠（AutoModel の起動方式、コンテナ方針の比較、落とし穴） | — |
| `reference/02_composer_migration.md` | MosaicML Composer 版 SFT からの移行設計と設定マッピング | 設定 YAML と `train.py` に反映済み。本番モデルでの精度比較は未実施 |
| `reference/03_dlc_selection.md` | ベース DLC イメージの選定 | 選定結果でビルド・実行済み |
| `reference/04_verification_log.md` | 構築中に判明した問題・原因・判断と、ローカル / SageMaker での実測値 | — |

## 目的別の入口

| やりたいこと | 読むガイド |
| --- | --- |
| 一から構築する | `01_container_build.md` → `02_local_verification.md` → `03_training_job.md` |
| ECR にイメージがある状態で Training Job を実行する | `03_training_job.md` |
| 8 GPU で実行する、Spot を使う、本番モデルに差し替える | `04_scale_and_operations.md` |
| 自分のデータとモデルに合わせて設定を変える | `05_configuration.md` |
| Composer 版の学習設定を移植する | `reference/02_composer_migration.md` → `05_configuration.md` |
| AutoModel のバージョンやベース DLC を変える | `01_container_build.md`（ビルドオプション）→ `reference/03_dlc_selection.md` |
| 問題が起きた | `06_troubleshooting.md` → `reference/04_verification_log.md` |
| なぜこの構成なのかを知る | `00_overview.md` → `reference/01_design_rationale.md` |

## 対象リージョン

各ガイドの冒頭に「対象リージョン」を記載しています。すべて `us-west-2` を前提としたコマンドですが、`REGION` や `--region` の値を読み替えれば別リージョンでそのまま使用できます。  
ECR イメージ、SageMaker Studio、Training Job は同一リージョンに置く必要があります。
