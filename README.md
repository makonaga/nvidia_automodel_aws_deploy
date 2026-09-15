# NeMo AutoModel on Amazon SageMaker

NVIDIA の LLM/VLM 学習フレームワーク [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
を Amazon SageMaker Training Job 上で実行するための構成一式。

## 方針

1. AutoModel 導入済みのカスタムコンテナ（NGC 公式イメージ + SageMaker 対応）を ECR に用意する
2. `train.py`（SageMaker ↔ AutoModel のアダプタ）と Jupyter Notebook からジョブを起動する

## ドキュメント

- [docs/00_approach.md](docs/00_approach.md) — 構築アプローチの検討結果（調査結果・設計・手順・リスク）
- [docs/01_composer_to_automodel.md](docs/01_composer_to_automodel.md) — 既存 MosaicML Composer 版 SFT からの移行設計（設定マッピング・機能ギャップと対策）
- [docs/02_dlc_selection.md](docs/02_dlc_selection.md) — ベース DLC イメージの選定（候補比較・wheel 互換性・Dockerfile 案）
- [docs/03_phase2_findings.md](docs/03_phase2_findings.md) — Phase 2（コンテナ作成）で判明した問題・原因・判断の記録

## コンテナ

- [container/README.md](container/README.md) — Docker イメージの作成と ECR への push 手順（ステップバイステップ）

## 学習スクリプトと Notebook

- [src/README.md](src/README.md) — `train.py`（SageMaker ↔ AutoModel アダプタ）の仕様
- `configs/sagemaker/qwen3_5_cooking_lora.yaml` — SageMaker 用ベース設定
- `notebooks/01_launch_training_job.ipynb` — Training Job の起動・成果物確認

## 学習データ

- [data/README.md](data/README.md) — 料理の基礎知識 SFT サンプル（306 件、train/val 分割済み）

## ステータス

構築アプローチの検討フェーズ。スコープ確定済み（7B〜13B 級 / LoRA・PEFT / 単一ノード多 GPU / S3 チャネル）。コンテナは AWS DLC 拡張方式を採用。Phase 2（コンテナ作成）はローカル GPU での学習テストまで完了。Phase 3〜5（train.py / YAML / Notebook）を実装済み。MTP 有効 + packed + causal-conv1d の構成でローカル学習テスト済み。train.py の模擬検証まで完了。ECR push と初回ジョブが次のステップ。
