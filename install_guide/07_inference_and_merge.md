# 推論とマージのガイド

このドキュメントでは、学習で得た LoRA アダプタを配信に使うまでの手順を説明します。  
アダプタのまま Hugging Face でロードして生成する検証、アダプタをベースモデルにマージした HF 形式モデルの作成と vLLM での配信、Qwen3.5 の MTP ヘッドを含めたマージ、の順に進めます。

> **対象リージョン**: `us-west-2`（オレゴン）  
> 検証はすべて SageMaker Studio の Notebook から SageMaker Training Job を起動する形で行います（ローカル GPU は使いません）。

## 検証状況

| ステップ | 内容 | 検証状況 |
| --- | --- | --- |
| 1 | アダプタのまま HF transformers + PEFT でロードして生成する（`mtp.*` の LoRA 重みの扱いを確認） | **未実施**（手順と検証スクリプトを用意済み） |
| 2 | `tools/merge_lora.py` で MTP なしのマージ済みモデルを作り、vLLM で配信する | 未実施（手順未作成） |
| 3 | 本体と MTP ヘッドの両方にマージするツールの実装と、MTP 有効での配信 | 未実施（手順未作成） |

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

したがって、HF の `Qwen3_5ForConditionalGeneration` に載せれば本体のアダプタは一致し、`mtp.*` の LoRA 重みだけが未使用になる見込みです。  
`AutoModelForCausalLM` に載せるとパスが一致せず、PEFT は `strict=False` でロードするため、エラーにならずにアダプタが効かない状態になる可能性があります。ステップ1 の検証スクリプトは、この 2 経路でキーの一致数を自前で照合します。

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

### 完了の確認

- `key check` (ConditionalGeneration) で `unused` が `mtp.*` のみ、`mismatched` が 0
- アダプタ付きの生成が学習データの回答に近い

この結果と実測値を確認したら、本ガイドの検証状況と `reference/04_verification_log.md` に記録します。

## ステップ2: MTP なしのマージ済みモデルを作り vLLM で配信する（未作成）

AutoModel 同梱の `tools/merge_lora.py`（`--base-model`、`--adapter-path`、`--output-dir`、`--dtype`）は HF の `PeftModel.merge_and_unload()` でマージして保存します。  
ステップ1 の結果を踏まえて、どのクラスでロードするか（`mtp.*` の扱い）を決めてから手順を作成します。

## ステップ3: MTP ヘッドを含めてマージする（未作成）

vLLM / SGLang の MTP 投機的デコーディングで学習済みの MTP ヘッドを使うには、本体と MTP ヘッドの両方に LoRA をマージした HF 形式（`mtp.*` を含む）のチェックポイントが必要です。  
HF クラス経由のマージでは `mtp.*` が落ちるため、AutoModel のネイティブモデル上でマージして書き出すツールを実装します（`reference/04_verification_log.md` 2.2）。
