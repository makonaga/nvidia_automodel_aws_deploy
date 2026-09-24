# 設定リファレンス

このドキュメントでは、`train.py` のハイパーパラメータ、SageMaker の規約と AutoModel 設定の対応、設定 YAML、学習データの形式、成果物の内容を説明します。  
`03_training_job.md` を一度通した後、自分のモデルとデータに合わせて設定を変えるときに参照してください。

## `train.py` の役割

`src/train.py` は、SageMaker の toolkit が `torchrun` 経由で各 rank で起動するエントリポイントです。学習ループは持ちません。  
処理の流れは次のとおりです。

1. `--config` で指定したベース YAML（`configs/sagemaker/` 配下）を読み込む
2. SageMaker の規約を YAML に写像する（`SM_*` 環境変数が無いローカル実行ではスキップ）
3. ハイパーパラメータを上書きとして適用し、派生値（バッチの整合、rsLoRA の alpha、ウォームアップのステップ数）を計算する
4. 実効 YAML を `/opt/ml/output/data/effective_config.yaml` に書き出し、`nemo_automodel.cli.app.main()` に委譲する
5. 学習後、rank 0 が `LOWEST_VAL`（または `LATEST`）のチェックポイントを `/opt/ml/model` にコピーする

AutoModel の CLI に委譲することで、AutoModel 側のレシピ追加やランチャーの変更に自動で追従します。

## SageMaker の規約と AutoModel 設定の対応

| SageMaker | AutoModel の設定 |
| --- | --- |
| `SM_CHANNEL_TRAIN` 内の `*.jsonl` / `*.json` | `dataset.path_or_dataset_id`（複数ファイルならリスト） |
| `SM_CHANNEL_VALIDATION` | `validation_dataset.path_or_dataset_id`。チャネルが無ければ `validation_dataset` を削除する |
| `SM_CHANNEL_MODEL`（非圧縮ディレクトリまたは `model.tar.gz`） | `model.pretrained_model_name_or_path`。tar.gz はノードごとに 1 回だけ展開する |
| `/opt/ml/checkpoints` | `checkpoint.checkpoint_dir`（Estimator の `checkpoint_s3_uri` と双方向同期される） |
| `SM_MODEL_DIR`（`/opt/ml/model`） | 学習後の成果物のコピー先。`model.tar.gz` として S3 に置かれる |
| `SM_OUTPUT_DATA_DIR`（`/opt/ml/output/data`） | `effective_config.yaml` の置き場。`output.tar.gz` として S3 に置かれる |
| `WORLD_SIZE`（torchrun が設定） | `global_batch_size % (local_batch_size × world_size) == 0` になるよう切り上げる |
| `dist_env.timeout_minutes` | 10 分未満なら 10 分に引き上げる（イメージ pull やモデル DL の待ち時間対策） |

Training Job の入力チャネルは `.tar.gz` を自動展開しません。  
`model` チャネルに `model.tar.gz` を渡した場合は `train.py` が展開しますが、S3 側で一度展開して非圧縮 prefix として置き直す方が単純です。

## ハイパーパラメータ

Notebook の `hyperparameters` は、toolkit が `--key value` のコマンドライン引数に変換して `train.py` に渡します。

| キー | 意味 |
| --- | --- |
| `config` | ベース YAML のファイル名（`configs/sagemaker/` 配下）またはパス。必須 |
| `set` | `a.b=1,c.d=x` 形式の上書き。Notebook から dict で渡すと toolkit がこの形式に変換する |
| `model_id` | HF Hub の model id またはローカルパス。`model` チャネルより優先 |
| `warmup_epochs` | 学習データ件数（packed の場合は pack 数の見積もり）と `global_batch_size` から `lr_scheduler.lr_warmup_steps` を算出する |
| `rslora_alpha` | rsLoRA 相当の alpha。`peft.alpha = round(alpha × sqrt(peft.dim))` に換算する |
| `final_checkpoint` | `/opt/ml/model` にコピーするチェックポイント。`LOWEST_VAL`（既定）または `LATEST` |
| `checkpoint_dir` | `checkpoint.checkpoint_dir` の明示指定。既定は `/opt/ml/checkpoints` |
| `print_sample` | 学習前にレンダリング済みプロンプトを N 件表示する。既定 1、`0` で無効 |
| `fresh_start` | `1` で `checkpoint_dir` の既存チェックポイントを削除してから始める。既定 `0` |
| 上記以外の `a.b.c` 形式 | そのまま AutoModel の設定上書き。例: `step_scheduler.num_epochs: 3`、`optimizer.lr: 1e-4`、`model.attn_implementation: flash_attention_2` |

値は YAML と同じ規則で型変換されます（`1e-4` は float、`true` は bool、`null` は None）。  
リストは `--set` では渡せないため、YAML 側に書いてください。

`warmup_epochs` の算出では、packed sequence の場合に学習データの先頭 200 件をトークナイズして平均トークン数から pack 数を見積もります。  
AutoModel は `global_batch_size` に満たない最後のバッチを捨てるため、1 エポックのステップ数は `pack 数 // global_batch_size` です。

## 設定 YAML

`configs/sagemaker/qwen3_5_cooking_lora.yaml` が SageMaker 用のベース設定です。  
各セクションの全パラメータと既定値は `reference/05_automodel_yaml_reference.md` を参照してください。  
パス類（`dataset`、`validation_dataset`、`model`、`checkpoint_dir`）は `train.py` が上書きするため、YAML の値はローカル実行時の既定値でしかありません。

主要セクションと、このリポジトリでの設定値の意図を示します。

| セクション | 設定値 | 意図 |
| --- | --- | --- |
| `seed` | `17` | 乱数シード。トップレベルに書く（`rng:` セクションは v0.6.0 のレシピでは読まれない） |
| `step_scheduler` | `global_batch_size: 8`、`local_batch_size: 1`、`num_epochs: 3`、`val_every_steps: 20`、`ckpt_every_steps: 50`、`save_checkpoint_every_epoch: true` | packed では単位が pack。Notebook が GPU 数から `global_batch_size` を上書きする |
| `model.backend` | `attn: sdpa`、`linear: torch`、`rms_norm: torch_fp32` | AutoModel の `BackendConfig` は TransformerEngine が使える環境では既定が `te` になる。packed の mask 経路を確実にし、後段の LoRA マージを単純にするため明示する |
| `model`（MTP） | `num_nextn_predict_layers` を上書きしない | HF config の値のまま MTP ヘッドを有効にする。無効にするなら `model.num_nextn_predict_layers: 0` |
| `peft` | `target_modules: '*_proj'`、`dim: 16`、`alpha: 32`、`dropout: 0.05`、`use_triton: true` | LoRA。rsLoRA 相当にするなら `rslora_alpha` を渡す |
| `distributed` | `strategy: fsdp2`、`tp_size: 1`、`cp_size: 1` | 単一ノード多 GPU は FSDP2 のみ |
| `dataset` / `validation_dataset` | `ColumnMappedTextInstructionDataset`、`column_mapping: {question: prompt, answer: output}`、`use_hf_chat_template: true`、`answer_only_loss_mask: true`、`seq_length: 1024`、`truncation: true` | HF のチャットテンプレートでレンダリングし、回答部分のみで損失を計算する |
| `packed_sequence` | `packed_sequence_size: 2048`、`packing_strategy: neat`、`drop_long_samples: true` | 複数サンプルを 1 本の系列に詰める。MTP を有効にするために必須 |
| `optimizer` | `AdamW`、`lr: 1e-4`、`weight_decay: 0.01` | |
| `lr_scheduler` | `cosine`、`min_lr: 6e-6`、`lr_warmup_steps: 0` | `warmup_epochs` を渡すと `train.py` が上書きする |
| `checkpoint` | `safetensors`、`save_consolidated: true` | PEFT では HF PEFT 形式のアダプタが `model/` に保存される |

`configs/local/qwen3_5_cooking_lora_local.yaml` はローカル検証用で、pack 長 1024、`max_steps: 20`、出力先が `out/local_test/` になっている以外は同じ構成です。

### MTP と packed sequence の関係

Qwen3.5 のチェックポイントは MTP ヘッドの重みを同梱しており、AutoModel は学習時に MTP 損失を自動で有効にします。  
AutoModel 0.6.0 の MTP はパディング付きバッチでは形状エラーになり、packed sequence（`packing_strategy: neat`）と `model.backend.attn: sdpa` の組み合わせでのみ正しく動きます。  
packed では Qwen3.5 の線形 attention 層が文書境界を守るために `causal-conv1d` を使うため、コンテナに同梱しています。  
検証（eval）は MTP が実行されないためパディングのままで問題ありません。詳細は `reference/04_verification_log.md` 2.2 を参照してください。

### バッチサイズの単位

packed sequence では `global_batch_size` と `local_batch_size` の単位が「サンプル」ではなく「pack（`packed_sequence_size` トークン）」になります。  
料理データ（245 件、平均 127 トークン）は pack 2048 で 17 pack になり、`global_batch_size: 8` では 1 エポック 2 ステップです。  
1 pack あたりの VRAM は約 14.5 GiB（pack 2048、Qwen3.5-0.8B、LoRA、MTP 有効）で、A10G 24 GB と A100 40 GB では `local_batch_size: 1`、A100 80 GB では 2〜4 が目安です。

## 学習データの形式

`train` と `validation` チャネルには JSONL（1 行 1 レコード）または JSON（配列）を置きます。  
`ColumnMappedTextInstructionDataset` は `context` を system ロール、`question` を user ロールに入れるため、instruction と input を結合した `prompt` を `question` に写像しています。

```json
{"prompt": "次の食材に適した下ごしらえを答えてください。\n\nこんにゃく", "output": "こんにゃくは下ゆでをして臭みを抜くのが基本の下ごしらえです。…"}
```

- `prompt`: user メッセージ。既存の Composer 版と同じく `instruction + "\n\n" + input`（`input` が空なら `instruction` のみ）
- `output`: assistant メッセージ。必ず文字列にする（list は不可）
- 他のフィールド（`instruction`、`input` など）があっても無視される

レンダリング結果は学習開始時に `=== Sample Prompt 0 ===` として出力されます。  
Qwen3.5 のテンプレートでは assistant 応答の前に空の `<think>` ブロックが入り、`answer_only_loss_mask: true` ではこのブロックも損失の対象になります（非思考モードでの配信と整合します）。

サンプルデータ `data/cooking_basics/` の内容と再生成方法は `data/README.md` を参照してください。

## 成果物（`model.tar.gz`）の内容

```
model/adapter_model.safetensors   # LoRA アダプタ（HF PEFT 形式）
model/adapter_config.json         # HF PEFT の設定
model/automodel_peft_config.json  # AutoModel の PEFT 設定
model/tokenizer.json など         # AutoModel がチェックポイントと一緒に保存した tokenizer
config.yaml                       # AutoModel が保存したチェックポイント時点の設定
effective_config.yaml             # train.py が組み立てた実効設定
tokenizer/                        # ベースモデルの tokenizer（chat_template.jinja を含む）
training.jsonl / validation.jsonl # ステップごとのメトリクス（loss、lr、tps、mem など）
training_info.json                # 選択したチェックポイント、選択基準、ベースモデル、world_size
```

フル微調整（`peft` セクションなし）の場合は `model/` に HF 形式の consolidated 重みが入ります。

チェックポイントそのもの（`epoch_*_step_*/`、`LATEST`、`LOWEST_VAL`）は `checkpoint_s3_uri` の prefix に残ります。  
`/opt/ml/checkpoints` の同期エージェントが置く `*.sagemaker-uploaded` マーカーは `model.tar.gz` には含まれません。

## CloudWatch メトリクス

Notebook の `metric_definitions` は AutoModel のログ形式に合わせた正規表現です。

```
step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}
[val] name "{}" | step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}
```

`train:loss`、`train:grad_norm`、`train:lr`、`train:tps`、`eval:loss` が Training Job のメトリクスとして記録され、コンソールのグラフや `TrainingJobAnalytics` で参照できます。
