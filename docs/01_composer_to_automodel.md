# Composer 版 SFT から NeMo AutoModel への移行設計

参照した既存実装:

| ファイル | 役割 |
|---|---|
| `launcher_sft.py` | SageMaker の entry_point。`composer -n 8 <train script>` を subprocess 起動 |
| `judgemodel_sft_qwen31.7b_20251029_fsdp_rsLoRA.py` | Composer Trainer による Qwen3-1.7B judge モデルの rsLoRA SFT |
| `judgemodel_sft_qwen31.7b_251026_rsLoRA_fsdp.ipynb` | `HuggingFace` Estimator によるジョブ投入 |

ベースモデル・学習データはいずれも S3 上にある前提です。

---

## 1. 既存構成の要約

```
HuggingFace Estimator (image_uri=llm-development:hf-pt260-py312-cu126-mosaic-tf4511)
  instance_type = ml.p4de.24xlarge, instance_count = 1, volume_size = 200
  distribution   = （コメントアウト。Composer が自前で分散を張るため）
  entry_point    = launcher_sft.py
        │
        ├─ SageMaker は各ノードで 1 プロセスだけ起動する
        ▼
  launcher_sft.py → subprocess: composer -n 8 <train script> <hyperparameters>
        ▼
  8 ranks で FSDP FULL_SHARD + rsLoRA 学習
        ├─ モデル: rank0 が boto3 で model.tar.gz を DL → tarfile 展開 → dist.barrier()
        ├─ データ: train チャネル経由の JSON を datasets で読み込み
        └─ 保存: /opt/ml/model/checkpoint に 1ep ごと
```

主要ハイパーパラメータ:

| 項目 | 値 |
|---|---|
| ベースモデル | Qwen3-1.7B（S3 の `pretrained_model/Qwen3-1.7B/model.tar.gz`） |
| LoRA | `r=128, alpha=256, dropout=0.08, use_rslora=True`, target = q/k/v/o/gate/up/down_proj |
| 精度 / 並列 | `amp_bf16`, FSDP `FULL_SHARD`, `mixed_precision=PURE`, activation checkpointing 有効 |
| バッチ | DataLoader `batch_size=16`（per-device）× 8 GPU = グローバル 128、microbatch は `auto` |
| optimizer | `DecoupledAdamW(lr=1e-4, betas=[0.9,0.999], eps=1e-6, weight_decay=0.01)` |
| scheduler | `CosineAnnealingWithWarmupScheduler(t_warmup='1ep', t_max='10ep', alpha_f=0.06)` |
| 学習量 | `max_duration='10ep'`, `eval_interval='25ba'`, EarlyStopper(patience=2ep) |
| その他 | grad clipping norm=1.0, seed=17, `max_length=1024` |

---

## 2. 起動方式をどう変えるか

### 2.1 変更点

Composer 版は「SageMaker が 1 プロセス起動 → その中から `composer -n 8` で 8 プロセスを起動」という
**二段構え**でした。`launcher_sft.py` はそのための薄いラッパーです。

AutoModel では**この中間層が不要になります**。`nemo_automodel/components/launcher/interactive.py` の
`InteractiveLauncher._is_torchrun_worker()` が `LOCAL_RANK` と `TORCHELASTIC_RUN_ID` を検出すると
torchrun を再起動せず in-process でレシピを実行するため、SageMaker のネイティブな torchrun 機構に
そのまま載せられます。

```python
# 既存 notebook でコメントアウトされているこの行を有効化する
distribution={"torch_distributed": {"enabled": True}}
```

```
SageMaker (torch_distributed)
  └─ torchrun --nproc_per_node=8 train.py --<hyperparameters>
       └─ train.py（各 rank） → nemo_automodel.cli.app.main()
            └─ InteractiveLauncher が torchrun 環境を検出 → in-process 実行
```

`launcher_sft.py` 相当は**廃止**し、`entry_point='train.py'` 一本になります。

### 2.2 なぜこちらが良いか

`MASTER_ADDR` / `MASTER_PORT` / `node_rank` / `nnodes` を SageMaker 側が管理してくれるため、
将来マルチノードへ拡張するときに `instance_count` を増やすだけで済みます。
Composer 方式（自前 launcher）はノードをまたぐ調整を自分で書く必要がありました。

**代替案**: 既存構成に寄せて `train.py` から `automodel <config.yaml> --nproc-per-node 8` を
subprocess 起動することも可能です（`launcher_sft.py` とほぼ同じ形）。
単一ノードなら動作しますが、上記の理由から `torch_distributed` を推奨します。

---

## 3. S3 上のモデル / データの受け渡し

### 3.1 既存方式の課題

```python
if local_rank == 0:
    s3.download_file(s3_bucket, s3_model_path, pretrain_model_path)
    tarfile.open(pretrain_model_path).extractall(model_dir)
if world_size > 1:
    dist.barrier()      # 全 rank が rank0 の完了を待つ
```

全 rank を起動してから rank0 だけがダウンロードするため、`dist.initialize_dist(timeout=1800)` で
タイムアウトを延ばす対処が必要になっていました。ダウンロード失敗時のリトライも自前です。

### 3.2 推奨方式: SageMaker の入力チャネルに任せる

```python
estimator.fit({
    "train":      "s3://<bucket>/data/judge_sft/train.json",
    "validation": "s3://<bucket>/data/judge_sft/val.json",
    "model":      "s3://<bucket>/pretrained_model/Qwen3-1.7B/",   # 非圧縮 prefix
})
```

コンテナ起動**前**に SageMaker が `/opt/ml/input/data/{train,validation,model}` へ配置するため、
rank 間の同期も barrier も不要になり、転送失敗時は SageMaker 側でリトライされます。
`dist` 初期化前にモデルが揃っている状態になるので、タイムアウト延長も不要です。

**注意**: training の入力チャネルは `.tar.gz` を自動展開しません（自動展開は推論のモデルアーティファクトのみ）。
既存の `model.tar.gz` を使うには次のどちらかが必要です。

- **(a) 推奨** — S3 側で一度展開し、非圧縮 prefix として置き直す。以降のジョブすべてで展開処理が不要になる
- (b) 既存踏襲 — `train.py` の rank0 で展開して barrier。1.7B 程度なら実用上問題ないが、(a) の方が単純

いずれの場合も AutoModel 側の指定は同じです。

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /opt/ml/input/data/model
```

### 3.3 チェックポイントの置き場所

既存は `/opt/ml/model/checkpoint` に保存していました（`model.tar.gz` に同梱される）。
AutoModel では **`/opt/ml/checkpoints` + `checkpoint_s3_uri`** を推奨します。

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: /opt/ml/checkpoints
  model_save_format: safetensors
  save_consolidated: true
```

`checkpoint_s3_uri` を指定すると学習中に S3 へ継続同期され、次ジョブ開始時に自動で戻されます。
さらに `StepScheduler` は `preemption_signal=signal.SIGTERM` を既定で持つため、
**Spot 中断時に SIGTERM を受けてチェックポイントを保存**します。Spot 運用との相性が良い部分です。

最終成果物（LoRA アダプタ）だけを `train.py` の rank0 で `/opt/ml/model` にコピーします。

---

## 4. 設定マッピング表

| Composer | NeMo AutoModel YAML | 備考 |
|---|---|---|
| `HuggingFaceModel(model, tokenizer, metrics)` | `recipe: TrainFinetuneRecipeForNextTokenPrediction` | レシピが内包 |
| `AutoModelForCausalLM.from_pretrained(model_dir)` | `model.pretrained_model_name_or_path: /opt/ml/input/data/model` | |
| `fsdp_config.sharding_strategy: FULL_SHARD` | `distributed.strategy: fsdp2` | FSDP2 は DTensor ベース |
| `mixed_precision: PURE` / `precision: amp_bf16` | レシピ既定の bf16 | |
| `activation_checkpointing: True` | `distributed.activation_checkpointing: true` | キー名は要確認 |
| `state_dict_type: 'full'` | `checkpoint.save_consolidated: true` | |
| DataLoader `batch_size=16` × 8 GPU | `step_scheduler.global_batch_size: 128` | |
| `device_train_microbatch_size='auto'` | `step_scheduler.local_batch_size: 4` | **auto 相当なし。手動で決める** |
| `DecoupledAdamW(lr=1e-4, wd=0.01)` | `optimizer: {_target_: torch.optim.AdamW, lr: 1.0e-4, weight_decay: 0.01}` | Decoupled ≒ AdamW |
| `CosineAnnealingWithWarmup(t_warmup='1ep', alpha_f=0.06)` | `lr_scheduler: {lr_decay_style: cosine, min_lr: 6.0e-6, lr_warmup_steps: N}` | `min_lr = lr × alpha_f` |
| `max_duration='10ep'` | `step_scheduler.num_epochs: 10` | |
| `eval_interval='25ba'` | `step_scheduler.val_every_steps: 25` | |
| `save_interval='1ep'` | `step_scheduler.save_checkpoint_every_epoch: true` | 既定で true |
| `GradientClipping(norm, 1.0)` | `clip_grad_norm.max_norm: 1.0` | |
| `seed=17` | `rng.seed: 17` | |
| `max_length=1024` | `dataset.seq_length: 1024` | |
| `LoraConfig(r=128, alpha=256, ...)` | `peft:` セクション | **5.1 参照（そのままでは等価にならない）** |
| `EarlyStopper(patience='2ep')` | — | **相当機能なし。5.4 参照** |

---

## 5. 機能ギャップと対策

移行にあたって**そのままでは再現できない点**が 4 つあります。いずれも回避策があります。

### 5.1 rsLoRA が非対応 → alpha 換算で等価にできる

`nemo_automodel/components/_peft/lora.py` の `PeftConfig` には `use_dora` はありますが
**`use_rslora` はありません**。スケーリングは固定です（`lora.py:189`）。

```python
obj.scale = alpha / dim        # 通常の LoRA
```

rsLoRA のスケーリングは `alpha / sqrt(r)` なので、既存設定の実効スケールは

```
256 / sqrt(128) = 22.627
```

これを AutoModel の `alpha / dim` で再現するには `alpha = 22.627 × 128 = 2896.3` とします。

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
  dim: 128
  alpha: 2896        # rsLoRA(r=128, alpha=256) と等価: 2896/128 = 22.625
  dropout: 0.08
```

`alpha` は int 型のため 2896 とすると実効スケールは 22.625 となり、誤差は 0.01% です。実質同一とみなせます。

> `dim` を変える場合は `alpha` の再計算が必要です（`alpha = alpha_rs × sqrt(dim)`）。
> この換算式を `train.py` に持たせて `--use_rslora` フラグで自動計算させるのが安全です。

### 5.2 `enable_thinking=False` が渡らない → 自前 dataset で回避

既存スクリプトは Qwen3 の思考モードを明示的に無効化しています。

```python
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False,
    enable_thinking=False,      # ← 思考モードを無効化
)
```

一方 AutoModel の `format_chat_template`（`formatting_utils.py:720`）は
`apply_chat_template` を **`enable_thinking` なしで**呼びます。
Qwen3 のテンプレートは `enable_thinking` の既定値が `True` のため、
**学習データのレンダリング結果が既存と変わります**。

対策は 2 段構えを推奨します。

**Step 1（まずはこれで通す）** — 標準の `ColumnMappedTextInstructionDataset` を使う。
instruction / input / output 形式にそのまま対応できます。

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /opt/ml/input/data/train/train.json
  column_mapping: {context: input, question: instruction, answer: output}
  use_hf_chat_template: true
  answer_only_loss_mask: true
  seq_length: 1024
  truncation: longest_first
```

**Step 2（レンダリング差分が問題になったら）** — `source_dir` に自前の dataset builder を置き、
`_target_` でそれを指す。`source_dir` は `/opt/ml/code` に展開されて `sys.path` 上にあるため、
モジュール名だけで解決できます。

```yaml
dataset:
  _target_: sm_dataset.build_judge_dataset
  path: /opt/ml/input/data/train/train.json
  enable_thinking: false
  seq_length: 1024
```

自前 dataset にすると、後述の `output` の型正規化や損失マスクの方針も自分で握れるため、
**既存の挙動を厳密に再現したい場合はこちらが確実**です。

### 5.3 損失マスクの方針が変わる（意図せぬ精度変化に注意）

これはギャップというより**改善**ですが、既存モデルと比較する際は把握しておく必要があります。

既存 Composer 版は `DataCollatorForLanguageModeling(mlm=False)` を使っており、
**プロンプト部分も含めた全トークンで損失を計算**しています。
AutoModel の `answer_only_loss_mask: true` は**回答部分のみ**で損失を計算します。

SFT では一般に後者が望ましい挙動ですが、**既存モデルと厳密に条件を揃えて比較したい場合は
`answer_only_loss_mask: false` にしてください**。ここを揃えずに「AutoModel にしたら精度が変わった」と
判断すると原因を取り違えます。

### 5.4 EarlyStopper 相当がない

`train_ft.py` に early stopping / patience の実装はありません。代替案:

- **推奨** — `num_epochs: 10` で完走させ、`save_checkpoint_every_epoch: true`（既定）で
  全エポックのチェックポイントを残し、`[val]` ログの loss を見て事後に最良を選ぶ。
  既存も `save_num_checkpoints_to_keep=-1` で全保存していたので運用は近い
- `BaseRecipe` のフックで自作する（実装コストは中程度）

### 5.5 その他の小さな差分

| 項目 | 対策 |
|---|---|
| `output` フィールドが list のことがある | 既存は学習時に `json.dumps` で文字列化していた。**S3 へのアップロード時に正規化しておく**のが最も安全（notebook のデータ準備セルで実施） |
| `train_test_split(test_size=0.2)` を学習時に実行 | AutoModel は train / validation を別データセットとして受ける。**事前に分割して 2 つの S3 オブジェクトにする** |
| `<think>` / `</think>` の特殊トークン追加 | Qwen3 のトークナイザには既に含まれるため、既存コードの `if` は実質 no-op のはず。自前 dataset を使う場合のみ要確認 |
| `device_train_microbatch_size='auto'` | 相当機能なし。`local_batch_size` を 4 あたりから始めて OOM を見ながら調整 |

---

## 6. 学習設定 YAML（案）

```yaml
# configs/judge_qwen3_1_7b_rslora_sm.yaml
recipe: TrainFinetuneRecipeForNextTokenPrediction

step_scheduler:
  global_batch_size: 128        # 既存: per-device 16 × 8 GPU
  local_batch_size: 4           # microbatch。OOM を見ながら調整
  num_epochs: 10
  val_every_steps: 25
  save_checkpoint_every_epoch: true

dist_env:
  backend: nccl
  timeout_minutes: 10

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 17
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /opt/ml/input/data/model   # train.py が上書き
  trust_remote_code: true

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
  dim: 128
  alpha: 2896                   # rsLoRA(r=128, alpha=256) 等価。5.1 参照
  dropout: 0.08

distributed:
  strategy: fsdp2
  dp_size: none
  tp_size: 1
  cp_size: 1

clip_grad_norm:
  max_norm: 1.0

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /opt/ml/input/data/train/train.json    # train.py が上書き
  column_mapping: {context: input, question: instruction, answer: output}
  use_hf_chat_template: true
  answer_only_loss_mask: true   # 既存と厳密比較するなら false（5.3 参照）
  seq_length: 1024
  truncation: longest_first

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /opt/ml/input/data/validation/val.json # train.py が上書き
  column_mapping: {context: input, question: instruction, answer: output}
  use_hf_chat_template: true
  answer_only_loss_mask: true
  seq_length: 1024
  truncation: longest_first

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: true
  num_workers: 2

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  betas: [0.9, 0.999]
  eps: 1.0e-6
  weight_decay: 0.01

lr_scheduler:
  lr_decay_style: cosine
  min_lr: 6.0e-6                # 既存 alpha_f=0.06 × lr=1e-4
  lr_warmup_steps: 0            # train.py が「1 エポック相当」を計算して上書き

checkpoint:
  enabled: true
  checkpoint_dir: /opt/ml/checkpoints
  model_save_format: safetensors
  save_consolidated: true
```

> `distributed.activation_checkpointing` と `truncation` の指定値は実装で最終確認が必要です（Phase 1 で検証）。
> `lr_warmup_steps` は既存の `t_warmup='1ep'` に合わせ、
> `len(dataset) / global_batch_size` を `train.py` で計算して渡すのが確実です。

---

## 7. Notebook の変更点

既存 notebook からの差分は以下だけです。

```python
estimator = PyTorch(                      # HuggingFace → PyTorch（または汎用 Estimator）
    image_uri=my_image_uri,               # AutoModel カスタムイメージ
    entry_point='train.py',               # launcher_sft.py は廃止
    source_dir='./src',
    instance_type='ml.p4de.24xlarge',     # 既存と同じ
    instance_count=1,
    volume_size=200,
    distribution={"torch_distributed": {"enabled": True}},   # ← コメントアウトを解除
    hyperparameters={
        'config': 'judge_qwen3_1_7b_rslora_sm.yaml',
        'set': 'step_scheduler.num_epochs=10,optimizer.lr=1e-4',
    },
    checkpoint_s3_uri=f's3://{bucket}/automodel/judge-sft/ckpt/',
    metric_definitions=[...],             # ← 下記の通り全面差し替え
)
estimator.fit({
    'train':      's3://.../data/judge_sft/train.json',
    'validation': 's3://.../data/judge_sft/val.json',
    'model':      's3://.../pretrained_model/Qwen3-1.7B/',
})
```

### metric_definitions の書き換え

既存の Regex は Composer のログ形式（`Train metrics/train/LanguageCrossEntropy: ...`）に依存しており、
**AutoModel では 1 件もマッチしません**。AutoModel のログ形式は `train_ft.py` で確認済みです。

```
step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}
[val] name "{}" | step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}
```

対応する定義:

```python
metric_definitions = [
    {'Name': 'train:loss',      'Regex': r'step \d+ \| epoch \d+ \| loss ([0-9\.]+)'},
    {'Name': 'train:grad_norm', 'Regex': r'\| grad_norm ([0-9\.]+)'},
    {'Name': 'train:lr',        'Regex': r'\| lr ([0-9\.eE\+\-]+)'},
    {'Name': 'train:tps',       'Regex': r'\| tps ([0-9\.]+)'},
    {'Name': 'eval:loss',       'Regex': r'\[val\][^|]*\| step \d+ \| epoch \d+ \| loss ([0-9\.]+)'},
]
```

Composer の `MaskedAccuracy` に直接対応するメトリクスは AutoModel の標準ログにはありません。
loss と perplexity（loss から算出可能）での監視に切り替えます。

### データ準備セルの追加

アップロード前に次の 2 つを済ませておきます（5.5 参照）。

1. `output` フィールドが list の場合に `json.dumps(..., ensure_ascii=False)` で文字列化
2. train / validation に 8:2 で分割し、それぞれ別オブジェクトとして S3 へ配置

---

## 8. 移行手順

既存の Composer 版が動いている前提で、差分の大きい順に潰します。

1. **データ準備** — `output` 正規化と train/val 分割を行い S3 へ配置。ベースモデルを非圧縮 prefix に展開
2. **コンテナ** — `nvcr.io/nvidia/nemo-automodel:25.11` + `sagemaker-training` を ECR へ push
3. **`train.py`** — SageMaker のチャネルパスを YAML 上書きに変換。rsLoRA の alpha 換算と
   `lr_warmup_steps` の算出もここに置く
4. **短時間ジョブで疎通** — `num_epochs=1` かつデータを数百件に絞って、
   ログ・チェックポイント・`/opt/ml/model` への出力までを確認
5. **レンダリング検証** — 学習開始直後にサンプルプロンプトをログ出力し、
   既存 Composer 版の `=== Sample Prompt ===` 出力と突き合わせる。
   差分があれば 5.2 の Step 2（自前 dataset）に切り替える
6. **本番設定で実行** — 10 エポック。既存モデルと eval loss を比較
7. **精度が揃わない場合の切り分け順** — (a) 損失マスク（5.3）、(b) チャットテンプレート（5.2）、
   (c) LoRA スケール（5.1）。この 3 つで説明がつくはずです

**ステップ 5 と 7 が移行で最も重要**です。学習は「動くけれど結果が微妙に違う」形で失敗しやすく、
その原因はほぼ 5.1〜5.3 の 3 点に集約されます。先にプロンプトのレンダリング結果を
文字列レベルで突き合わせておけば、後段の切り分けが大幅に楽になります。
