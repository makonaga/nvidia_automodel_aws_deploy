# NeMo AutoModel on Amazon SageMaker

NVIDIA の LLM/VLM 学習フレームワーク [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
を Amazon SageMaker Training Job 上で実行するための構成一式。

## 方針

1. AutoModel 導入済みのカスタムコンテナ（NGC 公式イメージ + SageMaker 対応）を ECR に用意する
2. `train.py`（SageMaker ↔ AutoModel のアダプタ）と Jupyter Notebook からジョブを起動する

## ドキュメント

- [docs/00_approach.md](docs/00_approach.md) — 構築アプローチの検討結果（調査結果・設計・手順・リスク）

## ステータス

構築アプローチの検討フェーズ。実装は未着手。
