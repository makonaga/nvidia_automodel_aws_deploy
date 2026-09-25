# 推論とマージのガイド

このドキュメントでは、学習で得た LoRA アダプタを配信に使うまでの手順を説明します。  
アダプタのまま Hugging Face でロードして生成する検証、アダプタをベースモデルにマージした HF 形式モデルの作成と vLLM での配信、Qwen3.5 の MTP ヘッドを含めたマージ、の順に進めます。

> **対象リージョン**: `us-west-2`（オレゴン）  
> 検証はすべて SageMaker Studio の Notebook から SageMaker Training Job を起動する形で行います（ローカル GPU は使いません）。

## 検証状況

| ステップ | 内容 | 検証状況 |
| --- | --- | --- |
| 1 | アダプタのまま HF transformers + PEFT でロードして生成する（`mtp.*` の LoRA 重みの扱いを確認） | 実施済み（2026-09-25、`ml.g5.2xlarge`、課金 325 秒） |
| 2 | アダプタをマージした HF 形式モデルを作り、AWS の vLLM DLC で SageMaker エンドポイントとして配信する | 実施済み（2026-09-25。マージジョブ 315 秒、エンドポイントは `ml.g5.xlarge` で InService まで 573 秒） |
| 3 | 本体と MTP ヘッドの両方にマージするツールの実装と、MTP 有効での配信 | ツール（`src/inference/merge_adapter_mtp.py`）と Notebook（`04_merge_mtp_and_deploy_vllm.ipynb`）は作成済み。**AWS 上では未実施** |

## 前提として分かっていること（ソースで確認済み）

学習の成果物 `model.tar.gz` の `model/` に入っているのは LoRA アダプタ（`adapter_model.safetensors`、`adapter_config.json`）だけで、ベースモデルの重みは含まれません。  
配信にはベースモデルと組み合わせる必要があり、AutoModel の上流ドキュメントは「ベース + アダプタ」でのロード（HF の `PeftModel`、vLLM の `LoRARequest`）と、`tools/merge_lora.py` によるマージの 2 通りを示しています。

アダプタのキーは AutoModel のネイティブ実装のモジュールパスで保存されます。Qwen3.5 では次のとおりです。

| 対象 | AutoModel ネイティブ実装 | HF `Qwen3_5ForConditionalGeneration` | HF `Qwen3_5ForCausalLM`（`AutoModelForCausalLM` が返す text-only クラス） |
| --- | --- | --- | --- |
| 層のパス | `model.language_model.layers.N` | `model.language_model.layers.N`（一致） | `model.layers.N`（不一致） |
| attention | `self_attn.{q,k,v,o}_proj` | 同名 | 同名 |
| MLP | `mlp.{gate,up,down}_proj` | 同名 | 同名 |
| Gated DeltaNet | `linear_attn.out_proj`（`in_proj_*` は `*_proj` にマッチしないため LoRA なし） | 同名 | 同名 |
| MTP ヘッド | `mtp.layers.0.*` | 無し（HF は `mtp.*` を読み飛ばす） | 無し |

ステップ1 の検証で、HF の `Qwen3_5ForConditionalGeneration` に載せると本体のアダプタ 228 キーがすべて一致し、`mtp.*` の 16 キーだけが未使用になることを確認しました。  
`AutoModelForCausalLM`（text-only クラス）では、`adapter_config.json` の `target_modules` がフルパス（`model.language_model.layers.N...`）で書かれているため 1 つも一致せず、PEFT が `ValueError: Target modules ... not found` で明示的に失敗します。黙って効かない状態にはなりません。

---

## ステップ1: アダプタのまま HF でロードして生成する

`notebooks/02_verify_adapter_inference.ipynb` を SageMaker Studio で実行します。  
Notebook は学習と同じイメージで Training Job（`ml.g5.2xlarge`、1 GPU、`distribution` なし）を起動し、`src/inference/infer_adapter.py` が次を行います。

1. `adapter` チャネルの `model.tar.gz` を展開し、`adapter_config.json` と `adapter_model.safetensors` の内訳（キー数、層ごとの LoRA 数、`mtp.*` のキー数）を出力する
2. `validation` チャネルの `val.jsonl` から先頭 N 件のプロンプトを取り、Qwen のチャットテンプレート（`enable_thinking=False`）でレンダリングする
3. HF の `Qwen3_5ForConditionalGeneration` でベースモデルをロードし、ベース単体で生成する
4. `PeftModel.from_pretrained` でアダプタを載せ、safetensors の各キーがモデルのパラメータに入ったか（一致 / 未使用 / 値の不一致）を照合し、アダプタ付きで生成する
5. `AutoModelForCausalLM` でも同じ照合を行い、一致数を記録する
6. 結果を `output.tar.gz` の `adapter_inference.json` に書き出す

`src/inference/requirements.txt` の `peft==0.18.1` は toolkit がジョブ起動前にインストールします。学習用の `src/` とは分けてあるため、学習ジョブの環境は変わりません。

### 実行手順

1. Studio のターミナルで `cd ~/nvidia_automodel_aws_deploy && git pull` を実行し、`notebooks/02_verify_adapter_inference.ipynb` を開く
2. セッション設定セルを実行し、`image_uri` が学習に使ったものと同じであることを確認する
3. セル「1. 検証対象のアダプタ」で `training_job_name` に `03_training_job.md` で完走した学習ジョブ名を入れる（`model.tar.gz` の S3 URI を `adapter_s3` に直接書いてもよい）
4. セル「2.」「3.」を実行する。ジョブは学習ジョブと同じくイメージの pull に数分かかる
5. セル「4.」で結果を取得する

### ログと結果の見方

| 確認項目 | 期待する結果 | 意味 |
| --- | --- | --- |
| `adapter keys: N (mtp.* : M)` | `M > 0` | MTP ヘッドの LoRA 重みがアダプタに含まれている |
| `key check` (ConditionalGeneration) の `unused` | `mtp.*` のキーだけ | 本体のアダプタはすべて一致し、MTP 分だけが HF 側に受け皿が無い |
| `key check` の `mismatched` | `0` | 値が正しく入っている |
| `key check` (AutoModelForCausalLM) の `matched` | `0` になる想定 | text-only クラスにはそのまま載せられないことの確認 |
| `generations: k/N prompts で出力がベースと異なる` | `k > 0` | アダプタが生成に効いている |
| `adapter` の出力 | `expected`（学習データの回答）に近い文体・内容 | 学習内容が反映されている |

`mtp.*` 以外のキーが `unused` に出た場合、または `matched` が 0 の場合は、モジュールパスの対応（上表）が想定と違うので、`unused_keys` の名前を確認してください。

### 検証結果（2026-09-25）

| 項目 | 結果 |
| --- | --- |
| アダプタ | 244 キー（本体 228、`mtp.*` 16）。`adapter_config.json` は `r=16`、`lora_alpha=32`、`task_type=CAUSAL_LM`、`target_modules` はフルパスで 124 モジュール（`mtp.layers.0.*` の 8 個を含む） |
| `Qwen3_5ForConditionalGeneration` | matched 228 / unused 16（すべて `mtp.*`）/ mismatched 0。モデル側の LoRA パラメータ数 228 と一致 |
| `AutoModelForCausalLM` | `ValueError: Target modules {...} not found in the base model`。ロード不可 |
| 生成 | 5/5 でベースと出力が変わった。ベースは Markdown 見出し付きの長文、アダプタ付きは学習データと同じ文体（見出しなし、直接回答、5 文前後）になった。1 件は greedy デコーディングで同じ句の繰り返しに退化した |
| 所要時間 | ジョブ全体 6 分（イメージ pull 3 分、peft のインストール数秒、モデルのロード 21 秒）。課金 325 秒 |

生成の内容面（料理知識としての正しさ）は 0.8B に 245 件・3 エポックという条件では期待できず、ここでは「アダプタが載っている」ことの確認に留めます。  
繰り返しの退化は `--repetition_penalty 1.1` などで抑えられますが、検証の既定値は変えていません。

### 完了の確認

- `key check` (ConditionalGeneration) で `unused` が `mtp.*` のみ、`mismatched` が 0
- アダプタ付きの生成が学習データの文体に変わっている

### ステップ2 以降への含意

- HF 経由でアダプタを扱うときは `Qwen3_5ForConditionalGeneration`（`AutoModelForImageTextToText`）でロードする。上流の `tools/merge_lora.py` は `task_type=CAUSAL_LM` から `AutoModelForCausalLM` を選ぶため、そのままでは同じ `ValueError` になる。`--model-class AutoModelForImageTextToText` の指定が必要
- `adapter_config.json` の `target_modules` に `mtp.layers.0.*` が含まれる。HF PEFT は他のモジュールが見つかれば無視するが、vLLM の LoRA ロードで問題になるかは未確認

## ステップ2: マージ済みモデルを作り、vLLM DLC エンドポイントで配信する

### 前提として調べたこと（2026-09-25 時点、ソースと公開リポジトリで確認）

vLLM 側（`vllm-project/vllm`、安定版 `v0.30.0`）:

- `Qwen3_5ForConditionalGeneration` をサポートし、LoRA 対応（`SupportsLoRA`）。MTP 用の `Qwen3_5MTP` クラスも含まれる（投機的デコーディングの `mtp` 方式が使える）
- 要求環境は torch 2.13.0、CUDA 13.0.3、Python 3.12。このリポジトリの学習イメージ（torch 2.10）とは両立しないため、vLLM は別イメージで動かす
- AutoModel のアダプタをそのまま vLLM の LoRA として読ませることはできない。vLLM は `mtp.` で始まる重みを読み捨てる設定（`WeightsMapper` で `None`）にしているが、LoRA の重み名の変換ではこれが `ValueError: Mapped LoRA weight name cannot be None.` になる（`vllm/lora/utils.py` の `parse_fine_tuned_lora_name`）。`mtp.*` を除いたアダプタを別に作れば読める見込みだが未検証
- 上記の理由と運用の単純さから、ステップ2 は「マージ済みのフルモデルを配信する」方式にする

AWS 側（`aws/deep-learning-containers`）:

- AWS 公式の vLLM Deep Learning Container が SageMaker 向けに提供されている。AL2023 版 `vllm:server-sagemaker-cuda-v2.5`（2026-09-23 リリース、vLLM 0.30.0、CUDA 13.0.2、Python 3.12）を使う。Ubuntu 版（`vllm:0.30.0-gpu-py312`）は AWS がセキュリティパッチを保証しないと明記しているため AL2023 版を選ぶ
- 同リポジトリのモデル一覧に `Qwen/Qwen3.5-0.8B` が「Smoke + Benchmark」として載っている
- SageMaker 用イメージはポート 8080 で vLLM の OpenAI 互換サーバーを起動し、`SM_VLLM_*` 環境変数を CLI フラグに変換する（`SM_VLLM_MAX_MODEL_LEN=4096` → `--max-model-len 4096`）。`model_data` の `model.tar.gz` は `/opt/ml/model` に展開され、`SM_VLLM_MODEL` を指定しなければ自動で `--model /opt/ml/model` になる
- `/invocations` に OpenAI Chat Completions 形式の JSON（`messages`、`max_tokens` など）をそのまま送れる
- デプロイ例では `inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1"` を指定している（CUDA 13 系のイメージに合わせた GPU 推論 AMI）
- LMI コンテナ（`djl-serving`）は最新の v0.36.0 で vLLM 0.17 系のため、Qwen3.5 の要件を満たさない。採用しない

### 手順

`notebooks/03_merge_and_deploy_vllm.ipynb` を SageMaker Studio で実行します。  
Notebook 側に vLLM を入れる必要はありません。Notebook は `sagemaker` SDK でジョブとエンドポイントを作り HTTP で呼び出すだけで、vLLM は AWS の vLLM DLC のコンテナとしてエンドポイントのインスタンス上で動きます。

1. セッション設定セルで、学習イメージと vLLM DLC のイメージ URI を確認する
2. セル「1.」で学習ジョブ名を指定する（ステップ1 と同じ）
3. セル「2.」でマージジョブを実行する。`src/inference/merge_adapter.py` が次を行う
   - `Qwen3_5ForConditionalGeneration` にアダプタを載せ、キーの一致を照合する（`mtp.*` 以外に未使用があれば失敗させる）
   - `merge_and_unload()` でマージし、`/opt/ml/model` に `save_pretrained`（safetensors）、tokenizer、processor 設定を保存する
   - 保存したモデルを読み直し、アダプタ付きモデルと同じプロンプトで生成して一致を確認する（`verify: 1`）
   - 結果を `output.tar.gz` の `merge_info.json` に書く
4. セル「3.」でエンドポイントをデプロイする（`ml.g5.2xlarge`、InService まで数分）
5. セル「4.」で 5 件のプロンプトを Chat Completions 形式で送り、出力を確認する。`chat_template_kwargs: {"enable_thinking": false}` で学習時と同じレンダリングにする
6. セル「5.」でステップ1 の HF + PEFT の出力と並べて比較する（任意）
7. **セル「6.」でエンドポイントとモデルを削除する**

### ログと結果の見方

| 確認項目 | 期待する結果 |
| --- | --- |
| マージジョブの `adapter key check` | `matched=228 unused=16 (mtp 以外の未使用 0)` |
| `mtp.* keys in output` | `0`（HF 経由のマージでは MTP ヘッドは含まれない） |
| `verify: k/3 prompts で ... 一致` | 3/3（同じ重み・同じ実装なので一致するはず。bf16 の丸めで 1 件程度ずれることはあり得る） |
| エンドポイントのデプロイ | `InService` になり、`predict` が `choices[0].message.content` を返す |
| vLLM の出力 | ステップ1 のアダプタ付き生成と同じ文体。カーネルの違いで文字列は完全一致しない |

### 検証結果（2026-09-25）

| 項目 | 結果 |
| --- | --- |
| マージジョブ | `adapter key check: matched=228 unused=16 (mtp 以外の未使用 0)`、`mtp.* keys in output: 0`、読み直し後の生成が 3/3 で一致。課金 315 秒（確保待ち 15 分は課金外） |
| 出力ファイル | `config.json`、`model.safetensors`（1.7 GB）、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`processor_config.json`。transformers 5 系は画像・動画の processor 設定も `processor_config.json` にまとめて保存する（旧形式の `preprocessor_config.json` は出ない）が、vLLM DLC（transformers 5.17）で問題なく読めた |
| エンドポイント | `ml.g5.2xlarge` と `ml.g6.2xlarge` は `InsufficientInstanceCapacity` で失敗（各 5〜10 分待ってから失敗が返る）。`ml.g5.xlarge` で InService まで 573 秒 |
| 呼び出し | Chat Completions 形式（`chat_template_kwargs: {enable_thinking: false}`、`temperature: 0`）で 5 件とも応答。出力は HF + PEFT（ステップ1）と冒頭 100 文字前後が完全一致し、後半で分岐する。bf16 の演算順とカーネルの違いによる差で、同じ重みが載っていることの確認としては十分 |

### 完了の確認

- マージ済み `model.tar.gz` が S3 にあり、`config.json` が最上位にある
- vLLM DLC エンドポイントで生成でき、学習データの文体になっている
- エンドポイントを削除した

## ステップ3: 本体と MTP ヘッドの両方にマージし、MTP 投機的デコーディングで配信する（未実施）

vLLM の MTP 投機的デコーディングで学習済みの MTP ヘッドを使うには、本体と MTP ヘッドの両方に LoRA をマージした HF 形式（`mtp.*` を含む）のチェックポイントが必要です。  
ステップ2 の HF クラス経由のマージでは `mtp.*` が落ちるため、`src/inference/merge_adapter_mtp.py` で MTP ヘッド分を別に計算して出力に加えます。  
ツールと Notebook は作成済みですが、AWS 上での実行はまだ行っていません。以下の「期待する結果」はソースから導いたもので、実測ではありません。

### 前提として調べたこと（2026-09-25 時点、ソースで確認）

AutoModel 側（v0.6.0、`nemo_automodel/components/models/qwen3_5/`）:

- MTP ヘッドは HF の `Qwen3_5DecoderLayer`（full attention）1 層に `enorm`、`hnorm`、`eh_proj`、`final_layernorm` を足した構造（`Qwen3_5DenseMTPSublayer`）。LoRA は `eh_proj`、`self_attn.{q,k,v,o}_proj`、`mlp.{gate,up,down}_proj` の 8 モジュールに付く（ステップ1 で確認した `mtp.*` 16 キー = 8 モジュール × A/B）
- `state_dict_adapter.py` が HF 形式との間で名前を変換するのは 4 キーだけ。`mtp.layers.0.eh_proj.weight` ↔ `mtp.fc.weight`、`enorm` ↔ `mtp.pre_fc_norm_embedding.weight`、`hnorm` ↔ `mtp.pre_fc_norm_hidden.weight`、`final_layernorm` ↔ `mtp.norm.weight`。decoder 層の `mtp.layers.0.self_attn.*` などは HF でも同名
- LoRA のスケールは `alpha / dim`（`_peft/lora.py`）で、`adapter_config.json` の `lora_alpha / r` と同じ。PEFT の `merge` と同じ `W + (lora_alpha / r) · B · A` で足し込める。`adapter_config.json` には `use_dora` も書かれる（このリポジトリの設定では false）

vLLM 側（`v0.30.0`）:

- `Qwen3_5MTP` が登録されており、`--speculative-config '{"method": "mtp", ...}'` で `model_type` が `qwen3_5` のモデルはドラフトとして `Qwen3_5MTP` が選ばれる（`vllm/config/speculative.py`）。ドラフトの層数は `config.json` の `mtp_num_hidden_layers`（最上位または `text_config`）から読み、`num_speculative_tokens` を省略した場合の既定値になる
- ドラフトモデルは本体と同じチェックポイントから重みを読む。`Qwen3_5MTP.load_weights` は `mtp.` で始まるキーを `model.` に読み替え、`embed_tokens` と `lm_head` は本体の重みを使う。モジュール名は `fc`、`pre_fc_norm_embedding`、`pre_fc_norm_hidden`、`norm`、`layers.N`（full-attention の decoder 層）で、上記の HF キー名と一致する
- 本体側（`Qwen3_5ForConditionalGeneration`）は `mtp.` で始まるキーを読み捨てる設定なので、`mtp.*` を含めても本体のロードには影響しない
- `Qwen3_5MTP` は Mamba キャッシュモード `all` では動かない（`align` を使う）。v0.30.0 の既定は `none`（prefix caching 無効時）か `align`（有効時）で、`all` は非推奨のため、既定のままで問題ない見込み

transformers 側（5.12.1、学習イメージ）:

- `Qwen3_5TextConfig` に MTP の項目は無いが、`PretrainedConfig` は未知のキーを属性として保持し `to_dict` で書き出すため、ステップ2 の `save_pretrained` でも `mtp_num_hidden_layers` は残る見込み。ツールはこれに頼らず、ベースの `config.json` から読んだ値を出力の `config.json` に明示的に書き、書く前の値を `merge_mtp_info.json` の `config_field_before` に記録する

AWS 側（vLLM DLC のエントリポイント `sagemaker_args.py`）:

- `SM_VLLM_*` の値が 1 つの JSON オブジェクトなら、そのまま 1 引数として渡される。`SM_VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'` → `--speculative-config '{...}'`

ベースのチェックポイント（`Qwen/Qwen3.5-0.8B`、Hub の `config.json` と `model.safetensors.index.json` で確認）:

- `config.json` の `text_config` に `mtp_num_hidden_layers: 1` がある（最上位には無い）。`mtp_use_dedicated_embeddings: false` なので MTP 用の埋め込みは別に持たず、vLLM のドラフトは本体の埋め込みと `lm_head` を使う
- 重みは `model.safetensors-00001-of-00001.safetensors` 1 ファイルと index で、`mtp.` で始まるキーは 15 個。LoRA が付く 8 モジュール（`mtp.fc.weight`、`mtp.layers.0.self_attn.{q,k,v,o}_proj.weight`、`mtp.layers.0.mlp.{gate,up,down}_proj.weight`）はすべて存在する。残る 7 個（`mtp.layers.0.{input_layernorm,post_attention_layernorm}.weight`、`mtp.layers.0.self_attn.{q,k}_norm.weight`、`mtp.norm.weight`、`mtp.pre_fc_norm_{embedding,hidden}.weight`）は LoRA 対象外で、ベースの値をそのまま書き出す

未確認のこと:

- vLLM が実際に MTP ドラフトを起動すること、受理率、速度はジョブとエンドポイントで確認する

### ツールの処理（`src/inference/merge_adapter_mtp.py`）

1. 本体: ステップ2 と同じく `Qwen3_5ForConditionalGeneration` + `PeftModel.merge_and_unload()` でマージして保存する。ステップ2 の `model.tar.gz` を `merged` チャネルに渡すとこの処理を省略し、その中身を使う
2. ベースの `mtp.*`: `huggingface_hub.snapshot_download` でベースの safetensors を取り、`mtp.` で始まるテンソルを直接読む（transformers を通さない）
3. MTP ヘッドのマージ: アダプタの `base_model.model.mtp.layers.0.<module>.lora_{A,B}.weight` ごとに、AutoModel のキー名を HF のキー名に変換してベースの重みを探し、`W + (lora_alpha / r) · B · A` を bf16 で書き戻す。形状不一致、対応するベースの重みが無い、A/B が揃わない、DoRA や `rank_pattern` 付き、のいずれかなら失敗する。モジュールごとに `|delta| / |W|` をログと `merge_mtp_info.json` に出す
4. 出力: `model.safetensors` が 1 ファイルなら `mtp.*` を加えて書き直す。分割（`model.safetensors.index.json` あり）なら `model-mtp.safetensors` を追加し index を更新する。`config.json` に MTP 層数を書く
5. 検証（`verify: 1`）: 出力の `mtp.*` の数と値を読み直して照合し、HF で本体を読み直してアダプタ付きの生成と一致することを確認する（HF は `mtp.*` を読み飛ばすので、本体の結果はステップ2 と同じになるはず）

MTP ヘッド自体の動作は学習イメージの中では確認しません（AutoModel の MTP 経路は packed 入力専用で、推論用の経路が無い）。動作確認は vLLM のエンドポイントで行います。

### 手順

`notebooks/04_merge_mtp_and_deploy_vllm.ipynb` を SageMaker Studio で実行します。

1. セッション設定セルで、学習イメージと vLLM DLC のイメージ URI を確認する
2. セル「1.」で学習ジョブ名を指定する。ステップ2 のマージ済み `model.tar.gz` があれば `merged_s3` に指定する（本体のマージを省略）
3. セル「2.」でマージジョブを実行する（`ml.g5.2xlarge`）。終了後に `merge_mtp_info.json` を取り出し、マージした MTP モジュールと相対変化量、検証結果を表示する
4. セル「3.」で `SM_VLLM_SPECULATIVE_CONFIG` を付けてエンドポイントをデプロイする（在庫不足時の候補切り替えはステップ2 と同じ）
5. セル「4.」で 5 件のプロンプトを greedy で生成する
6. セル「5.」でステップ2（MTP なし）の vLLM 出力と比較する
7. セル「6.」で CloudWatch のコンテナログから `SpecDecoding metrics`（受理率）と `Qwen3_5MTP` のロード行を拾う
8. **セル「7.」でエンドポイントとモデルを削除する**

### ログと結果の見方（期待する結果。実測ではない）

| 確認項目 | 期待する結果 | 意味 |
| --- | --- | --- |
| `base mtp.* keys: N \| mtp_num_hidden_layers=1 (text_config)` | `N = 15` | ベースの MTP ヘッド 15 テンソルと層数が読めた |
| `mtp.layers.0.xxx -> mtp.yyy ... \|delta\|/\|W\|` | 8 行。`eh_proj` は `mtp.fc.weight` に対応 | アダプタの MTP 分がすべてベースの重みに対応付いた |
| `\|delta\|/\|W\|` | 0 より大きい | LoRA が実際に MTP ヘッドを変えている（大きさは学習量次第） |
| `verify mtp.*: written 15 / expected 15 / equal 15` | 3 つとも 15 | 出力に `mtp.*` が正しく入った |
| `verify body: 3/3 ... 一致` | 3/3（`merged` チャネルを使った場合は比較なし） | 本体の重みはステップ2 と同じ |
| エンドポイントのコンテナログ | `Qwen3_5MTP` のロード行と `SpecDecoding metrics: Mean acceptance length ...` | vLLM が MTP ドラフトを使っている |
| セル「5.」の比較 | MTP なしと同じ出力（greedy の投機的デコーディングは出力を変えない方式） | 本体の出力が変わっていない |

`Mean acceptance length` は 1 回のステップで確定するトークン数の平均（1.0 なら投機が全く当たっていない）です。0.8B に 245 件・3 エポックの学習では MTP ヘッドの精度は限られるため、受理率の絶対値より「MTP ドラフトが動いていること」の確認を目的とします。

### 検証結果（2026-09-25、マージジョブまで）

| 項目 | 結果 |
| --- | --- |
| マージジョブ（`ml.g5.2xlarge`、`merged` チャネルにステップ2 の `model.tar.gz`） | `base mtp.* keys: 15 \| mtp_num_hidden_layers=1 (text_config)`。LoRA `r=16`、`lora_alpha=32`、scaling 2.0。8 モジュールすべてがベースの重みに対応付いた。`verify mtp.*: written 15 / expected 15 / equal 15`。課金 330 秒（ベースのダウンロード 17 秒） |
| `\|delta\|/\|W\|` | `mtp.fc.weight` 0.021、`mlp.{down,gate,up}_proj` 0.007 / 0.008 / 0.010、`self_attn.{q,k,v,o}_proj` 0.007 / 0.012 / 0.005 / 0.006 |
| 出力 | `model.safetensors` 1 ファイル（1747 MB。ステップ2 の 1706 MB に `mtp.*` 分が加わった）。`config.json` の `text_config.mtp_num_hidden_layers` はステップ2 の出力にも残っていた（`before: 1`）ため書き換えなし |
| HF での読み直し | 473 テンソルを読み（`mtp.*` は読み飛ばし）、3 件とも生成できた。文体はステップ1・2 と同じ |
| エンドポイント（MTP 有効） | 未実施 |

### 完了の確認

- `mtp.*` 入りの `model.tar.gz` が S3 にあり、`config.json` に `mtp_num_hidden_layers` がある
- vLLM DLC エンドポイントが `SM_VLLM_SPECULATIVE_CONFIG` 付きで InService になり、ログに `SpecDecoding metrics` が出る
- エンドポイントを削除した
