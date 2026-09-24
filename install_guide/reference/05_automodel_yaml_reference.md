# AutoModel 設定 YAML リファレンス

このドキュメントでは、NeMo AutoModel 0.6.0 の LLM 微調整レシピ `TrainFinetuneRecipeForNextTokenPrediction` が読み取る設定 YAML のパラメータを一覧にまとめています。  
このリポジトリの `configs/sagemaker/qwen3_5_cooking_lora.yaml` を自分のモデルとデータに合わせて変えるときに参照してください。

> **調査対象**: AutoModel `v0.6.0` タグ（コミット `89c248a`）のソース。レシピが設定を型付きオブジェクトへ変換する `nemo_automodel/recipes/_typed_config.py` と、レシピ本体 `nemo_automodel/recipes/llm/train_ft.py`、各コンポーネントの設定 dataclass を読んで作成しました。  
> **範囲**: 単一ノードの LLM SFT / PEFT に関係するセクションを網羅し、VLM、diffusion、MoE、パイプライン並列に固有の項目は名前と概要のみ記載しています。上流のドキュメント（`docs/guides/configuration.mdx`）にはパラメータ一覧が無いため、ソースを正としています。

## 設定の仕組み

YAML は `ConfigNode` として読み込まれ、次のように扱われます。

- 文字列のスカラーは型変換されます（`"10"` → `10`、`"true"` → `True`、`"none"` / `"None"` → `None`）。YAML ネイティブの型はそのままです
- `_target_` キーを持つマッピングは、そのクラスや関数を `instantiate()` で呼び出して生成されます。ドット区切りの import パス（`pkg.module.symbol`）と、ファイルパス（`/path/file.py:symbol`）が使えます。`_target_` の解決は任意のコード実行になるため、信頼できる YAML だけを使ってください
- 文字列の中で環境変数を展開できます（`${VAR}`、`${VAR,default}`、`$VAR`）
- コマンドラインから `--a.b.c=value` 形式で任意の項目を上書きできます。`train.py` はこの仕組みでハイパーパラメータを渡します

レシピは `RecipeConfig` という型付きビューを通して設定を読みます。  
`step_scheduler`、`lr_scheduler`、`optimizer`、`loss_fn`、`checkpoint`、`wandb`、`mlflow`、`comet`、`prewarm`、`embedding_row_repair`、`dataset` / `dataloader` は dataclass に変換され、未知のキーは `TypeError` になります。  
`model`、`peft`、`distributed` などはそのまま `ConfigNode` として扱われます。

## トップレベルセクション一覧

| セクション | 必須 | このリポジトリの設定 | 内容 |
| --- | --- | --- | --- |
| `recipe` | 必須 | `TrainFinetuneRecipeForNextTokenPrediction` | 実行するレシピクラス |
| `seed` | 任意 | `17` | 乱数シード（既定 42）。後述の注意を参照 |
| `dist_env` | 任意 | `backend: nccl`、`timeout_minutes: 10` | 分散初期化 |
| `step_scheduler` | 任意 | あり | バッチサイズ、エポック、検証とチェックポイントの間隔 |
| `model` | 必須 | `NeMoAutoModelForCausalLM.from_pretrained` | モデルのロード方法とバックエンド |
| `peft` | 任意 | LoRA | あると PEFT 学習。無ければフル微調整 |
| `distributed` | 任意 | `fsdp2` | 並列化の戦略とサイズ |
| `clip_grad_norm` | 任意 | `max_norm: 1.0` | 勾配クリッピング |
| `loss_fn` | 必須 | `MaskedCrossEntropy` | 損失関数 |
| `dataset` / `validation_dataset` | `dataset` は必須 | `ColumnMappedTextInstructionDataset` | データセット |
| `packed_sequence` | 任意 | `neat`、2048 | sequence packing |
| `dataloader` / `validation_dataloader` | 任意 | `StatefulDataLoader` | DataLoader の引数 |
| `optimizer` | 必須 | `torch.optim.AdamW` | オプティマイザ |
| `lr_scheduler` | 任意 | `cosine` | 学習率スケジューラ。無ければ一定 |
| `checkpoint` | 任意 | `/opt/ml/checkpoints`（`train.py` が上書き） | チェックポイントの保存と再開 |
| `wandb` / `mlflow` / `comet` | 任意 | なし | 実験管理への記録 |
| `fp8` / `compile` / `qat` / `quantization` | 任意 | なし | 低精度、torch.compile、QAT、BitsAndBytes |
| `prewarm` / `embedding_row_repair` / `neftune` / `moe_metrics` / `tool_call_eval` / `nvtx` | 任意 | なし | 補助機能 |

セクションの見出しごとに、パラメータ、既定値、このリポジトリでの値、説明を示します。既定値の欄が空の項目は必須です。

---

## `recipe`

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `recipe` | | `TrainFinetuneRecipeForNextTokenPrediction` | レシピクラス名または import パス。LLM の SFT / PEFT はこのクラス。他に `TrainSeqClsRecipe`（分類）、`KnowledgeDistillationRecipeForNextTokenPrediction`（蒸留）などがある |

## `seed`、`dist_env`、`nvtx`

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `seed` | `42` | `17` | 乱数シード。レシピは `StatefulRNG(seed=..., ranked=True)` を生成し、rank ごとに異なるシードを派生させる。DataLoader のシャッフルにも使われる |
| `dist_env.backend` | `nccl` | `nccl` | `torch.distributed` のバックエンド |
| `dist_env.timeout_minutes` | `1` | `10` | 集団通信のタイムアウト。`train.py` は 10 分未満なら 10 分に引き上げる |
| `nvtx` | `false` | | NVTX の範囲マーカーを有効にする（Nsight でのプロファイル用） |

**注意**: 上流の example YAML の多くにある `rng:` セクション（`_target_: StatefulRNG`、`seed`、`ranked`）は、v0.6.0 の `train_ft.py` では参照されません。  
シードはトップレベルの `seed` だけで決まり、`rng.seed` を書いても無視されます。このリポジトリの YAML は当初 `rng.seed: 17` と書いていたため実際には 42 で動いており、`seed: 17` に修正しました。

## `step_scheduler`

`StepSchedulerConfig` に対応します。`local_batch_size` はここに書きますが dataclass の外で読まれる実行時の値です。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `global_batch_size` | `32` | `8`（Notebook が GPU 数から上書き） | 1 オプティマイザステップあたりの全 GPU 合計サンプル数。packed sequence では単位が pack。`local_batch_size × dp_size` で割り切れる必要があり、商が勾配蓄積回数になる |
| `local_batch_size` | `1` | `1` | 1 GPU あたりのマイクロバッチ。DataLoader の `batch_size` の既定値にもなる |
| `num_epochs` | `10` | `3` | エポック数。`max_steps` が無ければ `num_epochs × エポック長` が総ステップ数 |
| `max_steps` | `None` | | 総ステップ数の上限。`num_epochs` より先に到達したら終了 |
| `ckpt_every_steps` | `100` | `50` | N ステップごとにチェックポイントを保存。`None` でエポックごとのみ |
| `save_checkpoint_every_epoch` | `true` | `true` | エポック末にもチェックポイントを保存する |
| `val_every_steps` | `None` | `20` | N ステップごとに検証。`None` で定期検証なし。エポック末の検証は `validation_dataset` があれば行われる |
| `log_remote_every_steps` | `1` | | wandb / mlflow へ N ステップごとに記録 |
| `loss_average_window_steps` | `50` | | 平均 loss の移動窓 |
| `gc_every_steps` | `None` | | N ステップごとに `gc.collect()` |
| `start_step` / `start_epoch` | `0` | | 再開時の初期値（通常はチェックポイントから復元される） |
| `preemption_signal` | `"SIGTERM"` | | 受信したらチェックポイントを保存して終了するシグナル。SageMaker の Spot 中断は SIGTERM なので既定のままでよい。`None` で無効 |

1 エポックのステップ数は、データセットの長さ（packed では pack 数）を `global_batch_size` で割った商です。余りのサンプルは捨てられます。

## `model`

`_target_` に `nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained` を指定するのが標準です。  
以下は `from_pretrained` の引数で、それ以外のキーは Hugging Face の `from_pretrained` / モデル config にそのまま渡されます。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `pretrained_model_name_or_path` | | `Qwen/Qwen3.5-0.8B`（`train.py` が上書き） | HF Hub の model id またはローカルパス |
| `torch_dtype` | `auto` | `bf16` | パラメータの dtype。`bf16`、`fp16`、`fp32`、`auto`。transformers 5 は `dtype` を推奨するが、AutoModel の引数名は `torch_dtype` |
| `attn_implementation` | `flash_attention_2`（利用可能な場合）、それ以外は `sdpa` | | HF 実装を使うときの attention。ネイティブ実装（Qwen3.5 など）では `backend.attn` が優先される |
| `backend` | `BackendConfig()` | `attn: sdpa`、`linear: torch`、`rms_norm: torch_fp32` | ネイティブ実装のカーネル選択。下表を参照 |
| `trust_remote_code` | モデルに応じて自動判定 | | HF Hub のカスタムコードを許可する |
| `cache_dir` | `HF_HUB_CACHE` | | 重みのキャッシュ先 |
| `force_hf` | `false` | | AutoModel のネイティブ実装があっても HF の実装を使う |
| `use_liger_kernel` | `true` | | Liger Kernel（RMSNorm、SwiGLU などの融合カーネル）を適用する。入っていなければスキップ |
| `use_sdpa_patching` | `true` | | SDPA のパッチを適用する |
| `sdpa_method` | `None` | | SDPA のバックエンドを限定する（`flash`、`efficient`、`math`、`cudnn` のリスト）。トップレベルの `sdpa_method` でも指定できる |
| `quantization_config` | `None` | | BitsAndBytes の設定。トップレベルの `quantization` からも生成される |
| `load_base_model` | `false` | | 学習済みチェックポイントではなくベースモデルを明示的に読む（再開時の挙動制御） |
| `freeze_config` | `None` | | 凍結するモジュールの指定 |
| HF config の項目 | | | `num_nextn_predict_layers: 0` のように書くと HF config を上書きできる。Qwen3.5 では MTP ヘッドの層数で、`0` にすると MTP を無効化する |

### `model.backend`（`BackendConfig`）

`_target_: nemo_automodel.components.models.common.BackendConfig` を付けて指定します。既定値は TransformerEngine（TE）が import でき CUDA が使える環境で変わる点に注意してください。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `attn` | TE があれば `te`、無ければ `sdpa` | `sdpa` | attention のカーネル。`te`、`sdpa`、`flex`、`eager`、`tilelang` |
| `linear` | TE があれば `te`、無ければ `torch` | `torch` | Linear 層。`torch`、`te`、`quack` |
| `rms_norm` | `torch_fp32` | `torch_fp32` | RMSNorm。`torch`、`torch_fp32`（fp32 で計算）、`te`、`quack` |
| `rope` | `torch` | | RoPE。`torch`、`quack` |
| `rope_fusion` | TE があれば `true` | | 融合 RoPE（TE が必要）。v0.6.0 では上流の問題で強制的に無効化される（起動時に警告が出る） |
| `enable_hf_state_dict_adapter` | `true` | | HF 形式の state dict との相互変換を有効にする。チェックポイントを HF 形式で保存するために必要 |
| `enable_fsdp_optimizations` | `false` | | FSDP2 向けの最適化 |
| `compile_attn` | `false` | | attention を `torch.compile` する。`attn: sdpa`、`linear: torch`、`rms_norm: torch`、`rope_fusion: false` が条件 |
| `experts` / `dispatcher` / `dispatcher_num_sms` / `dispatcher_share_token_dispatcher` / `dispatcher_async_dispatch` / `fake_balanced_gate` / `fake_gate_noise` / `gate_precision` | | | MoE 向け。dense モデルでは無関係 |
| `cuda_graph` | 無効 | | CUDA Graph の設定 |

DLC には TE 2.11 が入っているため、`backend` を書かないと `attn` と `linear` の既定が `te` になります。  
このリポジトリは packed sequence の mask 経路を確実にし、後段の LoRA マージを単純にするため `sdpa` / `torch` を明示しています。

## `peft`

`_target_: nemo_automodel.components._peft.lora.PeftConfig` を指定します。このセクションがあると PEFT 学習になり、無ければフル微調整です。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `target_modules` | `[]` | `'*_proj'` | LoRA を付けるモジュール名のパターン（ワイルドカード可、リスト可）。`*_proj` は `q_proj` などサフィックスが `_proj` のもの全部 |
| `exclude_modules` | `[]` | | 除外するモジュール名のパターン |
| `match_all_linear` | `false` | | すべての `nn.Linear` を対象にする |
| `dim` | `8` | `16` | LoRA のランク r |
| `alpha` | `32` | `32` | スケーリング係数。実効スケールは `alpha / dim`。rsLoRA（`alpha / sqrt(r)`）は未対応で、`train.py` の `rslora_alpha` で換算する |
| `dropout` | `0.0` | `0.05` | LoRA のドロップアウト率 |
| `dropout_position` | `post` | | ドロップアウトを LoRA の前（`pre`）か後（`post`）に入れるか |
| `lora_A_init` | `xavier` | | A 行列の初期化 |
| `lora_dtype` | `None` | | LoRA 重みの dtype。`None` でベースモデルと同じ |
| `use_memory_efficient_lora` | `true` | | メモリ効率の良い実装を使う |
| `use_triton` | `false` | `true` | Triton カーネルで LoRA を計算する |
| `use_dora` | `false` | | DoRA を使う（`nn.Linear` のみ対応） |
| `moe_rank_scaling` | `false` | | MoE 向けのランクスケーリング |

## `distributed`

`_target_` は使わず、固定のスキーマで解釈されます。共通項目と、戦略ごとの項目があります。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `strategy` | `fsdp2` | `fsdp2` | `fsdp2`、`ddp`、`megatron_fsdp` |
| `dp_size` | 自動（`world_size / (tp × pp × cp)`） | `none` | データ並列サイズ。`none` で自動 |
| `dp_replicate_size` | `None` | | HSDP のレプリカ数（FSDP2） |
| `tp_size` | `1` | `1` | テンソル並列サイズ |
| `pp_size` | `1` | | パイプライン並列サイズ。2 以上なら `pipeline:` サブセクションが必要 |
| `cp_size` | `1` | `1` | コンテキスト並列サイズ |
| `ep_size` | `1` | | エキスパート並列サイズ（MoE） |

### `strategy: fsdp2` のときの項目（`FSDP2Config`）

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `sequence_parallel` | `false` | `false` | TP と組み合わせるシーケンス並列 |
| `tp_plan` | `None` | | TP のシャーディング計画の上書き |
| `activation_checkpointing` | `false` | | `true` / `full` で全ブロックを再計算、`selective` で選択的に再計算。VRAM を抑える |
| `activation_checkpointing_scope` | `all` | | 再計算の対象。`all`、`language`、`vision`、`audio`、`multimodal`（VLM 向け） |
| `mp_policy` | bf16 の混合精度 | | `MixedPrecisionPolicy` の上書き |
| `offload_policy` | `None` | | `CPUOffloadPolicy` でパラメータを CPU にオフロード |
| `autocast_dtype` | `None` | | autocast の dtype |
| `defer_fsdp_grad_sync` | `true` | | 勾配蓄積中は勾配同期を遅らせる |
| `reshard_after_forward` | 自動 | | forward 後にパラメータを再シャードする（メモリと通信のトレードオフ） |
| `enable_async_tensor_parallel` | `false` | | 非同期 TP |
| `enable_compile` | `false` | | FSDP2 と `torch.compile` の併用 |
| `patch_is_packed_sequence` | `false` | | packed sequence 向けのパッチ |
| `multimodal.frozen_sharding` | `root` | | 凍結した vision / audio タワーのシャーディング方針（`root`、`per_layer`、`replicate`） |
| `pipeline.*` | | | `pp_size > 1` のときの `pp_schedule`、`pp_microbatch_size`、`layers_per_stage` など |
| `moe.*` | | | MoE の並列化設定（`wrap_outer_model`、`reshard_after_forward` など） |

`strategy: ddp` では `activation_checkpointing`、`broadcast_buffers`、`find_unused_parameters`、`static_graph`、`bucket_cap_mb`、`gradient_as_bucket_view`、`autocast_dtype` が、`strategy: megatron_fsdp` では `zero_dp_strategy`、`megatron_fsdp_unit_modules`、`overlap_grad_reduce`、`overlap_param_gather` などが使えます。

## `clip_grad_norm`

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `max_norm` | `1.0`（未指定時にログを出して 1.0 を使う） | `1.0` | 勾配ノルムの上限 |

## `loss_fn`

`_target_` にクラスを指定するか、レジストリ名（`MaskedCrossEntropy`、`FusedLinearCrossEntropy`、`TEParallelCrossEntropy`、`KDLoss`）を指定します。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `_target_` | | `nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy` | 損失関数。`FusedLinearCrossEntropy` は lm_head と CE を融合してメモリを節約する |
| `fp32_upcast` | `true` | | logits を fp32 に上げてから CE を計算する |
| `ignore_index` | `-100` | | 損失から除外するラベル |
| `reduction` | `sum` | | `sum` または `mean`。`sum` のとき recipe がトークン数で正規化する |

Qwen3.5 の MTP 損失は `mtp` セクションを書かなくても `MTPLossConfig()` の既定（`scaling_factor: None`、`ignore_index: -100`）で自動的に有効になります。

## `dataset` / `validation_dataset`

`_target_` にデータセットクラスを指定し、残りのキーがそのコンストラクタに渡されます（`tokenizer` はレシピが注入します）。  
`validation_dataset` の他に `validation_dataset_<name>` の形で複数の検証セットを書けます。

このリポジトリで使う `ColumnMappedTextInstructionDataset` の引数は次のとおりです。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `path_or_dataset_id` | | `train.py` が上書き | ローカルの JSON / JSONL パス（リスト可）または HF Hub のデータセット id |
| `column_mapping` | | `{question: prompt, answer: output}` | 論理列（`context`、`question`、`answer`）から実際の列名への対応。`context` は system ロール、`question` は user ロール、`answer` は assistant ロールになる |
| `split` | `train` | | HF Hub データセットの split。ローカルファイルでは無視される |
| `name` | `None` | | HF Hub データセットの config 名 |
| `use_hf_chat_template` | `false` | `true` | tokenizer のチャットテンプレートでレンダリングする。`false` なら `context` / `question` / `answer` を改行で連結する |
| `answer_only_loss_mask` | `true` | `true` | assistant の回答部分だけを損失の対象にする |
| `seq_length` | `None` | `1024` | 系列長。`padding` / `truncation` の基準 |
| `padding` | `do_not_pad` | | `max_length` でパディング、`do_not_pad` でしない。packed sequence では不要 |
| `truncation` | `do_not_truncate` | `true` | `true` / `longest_first` / `only_first` などの HF の truncation 指定 |
| `limit_dataset_samples` | `None` | | 先頭 N 件だけ使う（動作確認用） |

他のデータセットクラス（`HellaSwag`、`SQuAD`、`ChatDataset`、`MockIterableDataset`、Megatron 形式など）は `nemo_automodel/components/datasets/llm/` にあり、引数はクラスごとに異なります。

## `packed_sequence`

複数サンプルを 1 本の系列に詰めて padding を無くします。`packed_sequence_size > 0` で有効になります。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `packed_sequence_size` | `0`（無効） | `2048` | 1 pack のトークン数。有効時はバッチサイズの単位が pack になる |
| `packing_strategy` | `thd` | `neat` | `thd`（`seq_lens` を渡す方式。TE attention 向け）または `neat`（bin packing + 文書 index 付き mask。sdpa 向け）。独自の `PackingConfig` サブクラスの import パスも可 |
| `drop_long_samples` | `true` | `true` | `packed_sequence_size` より長いサンプルを捨てる（`neat` のみ） |
| `max_packs` | `None` | | pack 数の上限 |
| `prepacked` | `false` | | データセットが既に packed 済みで再パックしない |
| `num_proc` | `1` | | packing 前のトークナイズの並列数 |
| `split_across_pack` | | | 旧項目。指定しても無視され警告が出る |

検証セットは、学習側が `thd` で TE attention を使う場合などを除き、packing せずに padding で評価されます。  
Qwen3.5 の MTP は `neat` + `sdpa` の組み合わせでのみ正しく動きます（`04_verification_log.md` 2.2）。

## `dataloader` / `validation_dataloader`

`_target_` は `torch.utils.data.DataLoader` または `torchdata.stateful_dataloader.StatefulDataLoader` のみ対応です。以下の項目以外を書くと `TypeError` になります。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `collate_fn` | `default_collater` | `nemo_automodel.components.datasets.utils.default_collater` | collate 関数。packing 有効時は packing 側の collater に置き換わる |
| `shuffle` | `None`（学習は有効） | `true` | シャッフル |
| `batch_size` | `step_scheduler.local_batch_size` | | マイクロバッチ。通常は書かない |
| `num_workers` | `0` | `2` | DataLoader のワーカー数。`/dev/shm` が小さい環境では控えめにする |
| `pin_memory` | `false` | | |
| `persistent_workers` | `false` | | |
| `prefetch_factor` | `None` | | |
| `drop_last` | `false` | | |
| `group_by_length` | `false` | | 長さの近いサンプルをまとめる |
| `shuffle_buffer_size` | `10000` | | iterable データセットのシャッフルバッファ |
| `dataloader_type` | `None` | | Megatron データセット専用（`single` / `cyclic`） |

## `optimizer`

`_target_` にレジストリ名（`adam`、`adamw`、`fused_adam`、`flash_adamw`、`muon`、`normuon`、`dion`、`dion2`）か、`torch.optim.AdamW` のような import パスを指定します。  
import パスで torch のオプティマイザを指定した場合はそのコンストラクタ引数がそのまま渡され、レジストリ名の場合は型付き config の項目が使えます。

| パラメータ | 既定値（`adamw`） | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `lr` | `1e-4` | `1.0e-4` | 学習率。`lr_scheduler` の `max_lr` の既定になる |
| `weight_decay` | `0.01` | `0.01` | |
| `betas` | `(0.9, 0.999)` | `[0.9, 0.999]` | |
| `eps` | `1e-8` | `1.0e-6` | |
| `amsgrad` | `false` | | |
| `fused` | `false` | | 融合カーネル版 |
| `param_group_overrides` | `[]` | | パラメータ名のパターンごとに `lr` / `weight_decay` を変える |

## `lr_scheduler`

`OptimizerParamScheduler` の設定です。`None` の項目は総ステップ数とオプティマイザの `lr` / `weight_decay` から計算されます。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `lr_decay_style` | `cosine` | `cosine` | `constant`、`linear`、`cosine`、`inverse-square-root`、`WSD` |
| `lr_warmup_steps` | `min(1000, 総ステップ数 // 10)` | `train.py` が `warmup_epochs` から算出 | ウォームアップのステップ数。1 未満の小数は拒否される |
| `lr_decay_steps` | 総ステップ数 | | 減衰の終点 |
| `init_lr` | `lr × 0.1` | | ウォームアップ開始時の学習率 |
| `max_lr` | `lr` | | ピークの学習率 |
| `min_lr` | `lr × 0.01` | `6.0e-6` | 減衰後の下限 |
| `start_wd` / `end_wd` | `weight_decay` | | weight decay の開始値と終了値 |
| `wd_incr_steps` | 総ステップ数 | | weight decay を変化させるステップ数 |
| `wd_incr_style` | `constant` | | `constant`、`linear`、`cosine` |
| `use_checkpoint_opt_param_scheduler` | `true` | | 再開時にチェックポイントのスケジューラ状態を使う |
| `override_opt_param_scheduler` | `false` | | 再開時に YAML の値でスケジューラを上書きする |
| `wsd_decay_steps` / `lr_wsd_decay_style` | `None` | | `WSD`（warmup-stable-decay）の減衰区間と形状（`linear`、`cosine`、`exponential`、`minus_sqrt`） |

`lr_scheduler` セクションを書かない場合、学習率は一定です。

## `checkpoint`

`CheckpointingConfig` に対応します。`model_repo_id`、`model_cache_dir`、`is_peft` は `model` と `peft` から自動で埋まります。

| パラメータ | 既定値 | このリポジトリ | 説明 |
| --- | --- | --- | --- |
| `enabled` | `true` | `true` | チェックポイントの保存と読み込みを行う |
| `checkpoint_dir` | `checkpoints/` | `train.py` が `/opt/ml/checkpoints` に上書き | 保存先。同じディレクトリに前回のチェックポイントがあると自動で再開する |
| `restore_from` | `None`（自動検出） | | 再開元。`LATEST`、`epoch_0_step_100` のようなサブディレクトリ名、またはパス。`None` でも最新を自動検出して再開する |
| `model_save_format` | `safetensors` | `safetensors` | `safetensors` または `torch_save`。PEFT では `safetensors` に強制される |
| `save_consolidated` | `final` | `true`（`final` と同義） | HF 形式の consolidated 重みをいつ書き出すか。`false`、`final`（最後のみ）、`every`（毎回）。`true` は `final` |
| `best_metric_key` | `default` | | `LOWEST_VAL` の判定に使う検証セット名（`validation_dataset_<name>` の `<name>`） |
| `max_recent_checkpoints` | `None`（全部残す） | | 直近 N 個だけ残す。`LATEST` / `LOWEST_VAL` が指すものは保護される |
| `is_async` | `false` | | 非同期保存 |
| `wait_for_staging` | `false` | | 非同期保存でステージング完了を待つ |
| `cpu_offload` | `false` | | 保存前に state dict を CPU に移す |
| `allow_legacy_pickle_restore` | `false` | | pickle 形式の旧チェックポイントの読み込みを許可する |
| `dequantize_base_checkpoint` | `None` | | 量子化されたベースモデルを保存時に逆量子化する |
| `original_model_root_dir` | `None` | | 元のモデルディレクトリ（config などのコピー元） |
| `skip_task_head_prefixes_for_base_model` | `None` | | ベースモデル読み込み時に無視するパラメータのプレフィックス |
| `single_rank_consolidation` | `false` | | rank 0 だけで consolidation を行う |
| `staging_dir` | `None` | | consolidation 時の一時ディレクトリ |
| `v4_compatible` | `false` | | transformers v4 形式の `config.json` を保存する（vLLM など v4 形式を期待する消費者向け） |
| `diffusers_compatible` | `false` | | diffusers 形式のインデックス名で保存する |
| `consolidation_timeout_minutes` | `30` | | consolidation の同期タイムアウト |

再開時の互換性チェックは警告のみで、モデルや PEFT の構成が違っても読み込みを続行します（`04_verification_log.md` 2.2c）。

## `wandb` / `mlflow` / `comet`

セクションがあると、rank 0 が学習メトリクスを記録します。認証情報は環境変数（`WANDB_API_KEY`、`MLFLOW_TRACKING_URI` など）で渡します。

| セクション | パラメータ |
| --- | --- |
| `wandb` | `project`（`automodel`）、`entity`、`name`、`group`、`tags`、`notes`、`extra`（`wandb.init` への追加引数） |
| `mlflow` | `experiment_name`（`automodel-experiment`）、`run_name`、`tracking_uri`、`artifact_location`、`tags`、`resume`（`true`）、`description`、`flatten_depth`（`1`） |
| `comet` | `project_name`（`automodel`）、`workspace`、`api_key`、`experiment_name`、`tags`、`auto_metric_logging`（`false`） |

このリポジトリは `WANDB_MODE=disabled` を環境変数で渡し、CloudWatch のメトリクス（`metric_definitions`）で代替しています。

## その他のセクション

| セクション | パラメータ | 説明 |
| --- | --- | --- |
| `fp8` | `enabled`（`false`）、`recipe_name`（`tensorwise` / `rowwise` / `rowwise_with_gw_hp`）、`filter_fqns`、`emulate`、`enable_fsdp_float8_all_gather` | torchao の FP8 学習。H100 以降 |
| `compile` | `enabled`（`false`）、`mode`（`default`）、`fullgraph`、`dynamic`、`backend`、`options`、`dynamo_cache_size_limit`（`256`） | モデル全体の `torch.compile` |
| `qat` | `enabled`（`false`）、`quantizer_type`（`int8_dynact_int4weight` / `int4_weight_only`）ほか | torchao の量子化認識学習。`fake_quant_after_n_steps` で開始ステップを指定 |
| `quantization` | BitsAndBytes の設定（`load_in_4bit` など） | QLoRA 向け。`create_bnb_config` で `quantization_config` に変換される |
| `prewarm` | `cublas_backward`、`fla_gdn_autotune`、`mamba_ssd_autotune`、`comm_groups`（すべて `false`） | 最初のステップで失敗する場合に、setup 時に初期化を先に行う。`fla_gdn_autotune` は Qwen3.5 の GDN 層の Triton autotune を setup で済ませる |
| `embedding_row_repair` | `min_norm`（`1.0e-4`）、`max_rows`（`256`） | 壊れた入力埋め込み行の修復 |
| `neftune` | `noise_alpha`（`5.0`）または数値 | NEFTune（埋め込みへのノイズ付加） |
| `moe_metrics` | `enabled`、`mode`（`brief`）、`top_k_experts`、`detailed_every_steps` | MoE のルーティング統計 |
| `tool_call_eval` | | ツール呼び出しの評価 |
| `benchmark` | `warmup_steps`、`peak_tflops`、`nsys_*`、`num_nodes` | ベンチマーク用（`benchmark.py` レシピ） |

## このリポジトリの YAML との対応で注意する点

- `seed` はトップレベルに書く。`rng:` セクションは無視される
- `distributed.dp_size: none` は文字列 `"none"` が `None` に変換され、自動計算になる
- `model.backend` を書かないと DLC 上では TE バックエンドになる。TE の attention は packed（`neat`）の 4D mask を padding mask として解釈するため、`neat` を使うときは `attn: sdpa` を明示する
- `packed_sequence` を有効にすると `global_batch_size` / `local_batch_size` の単位が pack になる。`train.py` の `warmup_epochs` は pack 数を見積もってウォームアップを計算する
- `checkpoint.save_consolidated: true` は `final` と同じ意味で、学習終了時に HF 形式の重みを書き出す。PEFT ではアダプタが HF PEFT 形式で `model/` に保存される
- `step_scheduler.ckpt_every_steps` を `None` にするとエポック末のみの保存になる。`50` などの値は SageMaker の `checkpoint_s3_uri` 同期と組み合わせて Spot 中断の損失を抑える
