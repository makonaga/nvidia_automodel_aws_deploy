# `src/train.py` — SageMaker ↔ AutoModel アダプタ

SageMaker の toolkit が `torchrun` 経由で各 rank で起動するエントリポイントです。学習ループは持ちません。

## 処理の流れ

1. `--config` で指定したベース YAML（`configs/sagemaker/` 配下）を読み込む
2. SageMaker の規約を YAML に写像する（`SM_*` 環境変数が無いローカル実行ではスキップ）
3. ハイパーパラメータを上書きとして適用する
4. 実効 YAML を `/opt/ml/output/data/effective_config.yaml` に書き出し、`nemo_automodel.cli.app.main()` に委譲する
5. 学習後、rank 0 が `LOWEST_VAL`（または `LATEST`）のチェックポイントを `/opt/ml/model` にコピーする

## SageMaker の規約との対応

| SageMaker | YAML |
|---|---|
| `SM_CHANNEL_TRAIN` 内の `*.jsonl` / `*.json` | `dataset.path_or_dataset_id`（複数ファイルならリスト） |
| `SM_CHANNEL_VALIDATION` | `validation_dataset.path_or_dataset_id`。チャネルが無ければ `validation_dataset` を削除 |
| `SM_CHANNEL_MODEL`（非圧縮ディレクトリ or `model.tar.gz`） | `model.pretrained_model_name_or_path`。tar.gz はノードごとに 1 回だけ展開 |
| `/opt/ml/checkpoints` | `checkpoint.checkpoint_dir`（`checkpoint_s3_uri` と双方向同期される） |
| `SM_MODEL_DIR` (`/opt/ml/model`) | 学習後の成果物コピー先 → `model.tar.gz` |
| `SM_OUTPUT_DATA_DIR` | `effective_config.yaml` の置き場 → `output.tar.gz` |
| `WORLD_SIZE`（torchrun） | `global_batch_size % (local_batch_size × world_size) == 0` になるよう自動調整 |

## ハイパーパラメータ

| キー | 意味 |
|---|---|
| `config` | ベース YAML 名（必須） |
| `set` | `a.b=1,c.d=x` 形式の上書き。Notebook から dict で渡すと toolkit がこの形式に変換する |
| `model_id` | HF Hub の id またはパス。`model` チャネルより優先 |
| `warmup_epochs` | 学習データ件数と `global_batch_size` から `lr_scheduler.lr_warmup_steps` を算出 |
| `rslora_alpha` | rsLoRA 相当の alpha。`peft.alpha = round(alpha × sqrt(peft.dim))` に換算 |
| `final_checkpoint` | `LOWEST_VAL`（既定）または `LATEST` |
| `print_sample` | 学習前にレンダリング済みプロンプトを N 件表示（既定 1、0 で無効） |
| **上記以外の `a.b.c` 形式** | そのまま AutoModel の設定上書き（例: `step_scheduler.num_epochs: 3`） |

値は YAML と同じ規則で型変換されます（`1e-4` → float、`true` → bool、`null` → None）。
リストは `--set` では渡せないので YAML 側に書いてください。

## `/opt/ml/model` の中身

```
model/adapter_model.safetensors   # LoRA アダプタ (PEFT のとき)
model/...                         # フル微調整のときは model/consolidated/ に HF 形式
config.yaml                       # AutoModel が保存したチェックポイント時点の設定
effective_config.yaml             # train.py が組み立てた実効設定
tokenizer/                        # ベースモデルの tokenizer
training.jsonl / validation.jsonl # ステップごとのメトリクス
training_info.json                # 選択したチェックポイントとベースモデル
```

## ローカルでの検証

`container/local_sm_sim.sh` が `/opt/ml` のディレクトリ構造と `SM_*` 環境変数を再現し、
イメージ内で toolkit と同じ `torchrun ... train.py` を実行します。SageMaker のジョブを投げる前にこれで通してください。
