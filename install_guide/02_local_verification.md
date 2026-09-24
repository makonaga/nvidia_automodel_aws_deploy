# ローカル検証ガイド

このドキュメントでは、ビルドしたイメージをローカル PC で検証する手順を説明します。  
SageMaker にジョブを投げる前にここで通しておくと、コンテナ・モデル・データ・`train.py` の問題をインスタンス課金なしで切り分けられます。

ステップ1 は GPU が無くても実行できます。ステップ2 とステップ3 は NVIDIA GPU（24 GB 以上を推奨）と NVIDIA Container Toolkit が必要です。

> **対象リージョン**: なし（ローカル PC のみで完結します）

## 検証状況

| 項目 | 内容 |
| --- | --- |
| 検証環境 | NVIDIA GeForce RTX 3090 24 GB、Docker Engine、Linux |
| モデル | `Qwen/Qwen3.5-0.8B`（HF Hub から取得、約 1.6 GB） |
| データ | `data/cooking_basics/`（学習 245 件、検証 61 件） |
| 結果 | ステップ1〜3 すべて完走。実測値は `reference/04_verification_log.md` 4 章 |

## 前提条件

- `01_container_build.md` ステップ4 でイメージがビルド済みであること
- ステップ2 以降は `nvidia-smi` が成功し、`docker run --gpus all` が使えること
- Hugging Face Hub に到達できること（モデルのダウンロード）

---

## ステップ1: スモークテスト

```bash
cd container
./smoke_test.sh
```

次を確認します。

- 主要パッケージの import とバージョン（torch 2.10.0、nemo_automodel 0.6.0、transformers 5.12.1）
- `automodel --help`（CLI エントリポイント）
- SageMaker toolkit のエントリポイント `sagemaker_pytorch_container.training`
- `pip check`（既知の 2 件の衝突は許容）
- GPU がある場合のみ、flash-attn、TransformerEngine、flash-linear-attention、causal-conv1d の import

GPU の有無は `nvidia-smi` が成功するかで自動判定します。判定を上書きするときは `SMOKE_GPU=0`（GPU なし）または `SMOKE_GPU=1` を指定してください。  
GPU の無い PC では `cuda available: False` と表示されますが、それ自体は問題ありません。

## ステップ2: ローカル GPU で学習テスト

イメージ内から Qwen3.5-0.8B の LoRA 学習を 20 ステップだけ実行し、SageMaker と同じ経路（torchrun → `nemo_automodel.cli.app` → in-process 実行）でモデルのロード、データの読み込み、LoRA、packed sequence、MTP、チェックポイント保存までを検証します。

```bash
./local_train_test.sh
```

処理の流れは次のとおりです。

1. 前回の出力（`out/local_test/checkpoints`）をコンテナ経由で削除する。AutoModel は同じディレクトリにチェックポイントがあると自動で再開するため、設定を変えて試すときに前回の状態を引き継がないようにしています。`CLEAN=0` を指定すると削除せず、再開の動作を確認できます
2. HF Hub からモデルをダウンロードする（キャッシュは `<repo>/.hf_cache`。2 回目以降はスキップ）
3. `configs/local/qwen3_5_cooking_lora_local.yaml` で `torchrun --nproc_per_node=1` を実行する
4. 終了時にコンテナが root で作ったファイルの所有者をホストのユーザーに戻す

オプションは環境変数で指定します。

| 環境変数 | 既定値 | 内容 |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3.5-0.8B` | 別のモデルで試す |
| `NPROC` | `1` | GPU が複数あれば増やす |
| `CLEAN` | `1` | `0` で前回のチェックポイントを残す（再開テスト） |

最後に `step N | epoch E | loss ...` の行と `out/local_test/checkpoints/` の一覧が表示されれば成功です。  
最初のステップは flash-linear-attention の Triton カーネルのコンパイルで 1〜2 分かかり、以降は 1 秒未満です。

ローカル設定は `num_epochs: 10` で `max_steps: 20` により打ち切ります。  
packed sequence では 1 エポックのステップ数が少なく（cooking データは pack 1024 で 33 pack、global batch 8 で 4 ステップ）、`max_steps` より先にエポック末で正常終了することがあります。

## ステップ3: SageMaker の規約を模して `train.py` を検証

```bash
./local_sm_sim.sh
```

`out/sm_sim/opt_ml/` に SageMaker の `/opt/ml` 構造（`input/data/{train,validation}`、`code`、`checkpoints`、`model`、`output/data`）を作り、`SM_*` 環境変数を与えてイメージ内で `torchrun ... /opt/ml/code/train.py` を実行します。  
SageMaker の toolkit が行う起動形と同じで、`train.py` のパス写像から成果物のコピーまでを 10 ステップで通します。

確認するのは次の 4 点です。

1. `=== effective config ===` で `dataset`、`validation_dataset`、`checkpoint_dir` がチャネルのパスに置き換わっている
2. `packing 見積もり: samples=245, avg_tokens=127, pack_size=1024 → packs≈34` と `lr_warmup_steps ← 4` が表示される
3. `=== Sample Prompt 0 ===` に Qwen のチャットテンプレートでレンダリングされた 1 件目が出る（assistant の前に空の `<think>` ブロックが入る形）
4. 最後の `/opt/ml/model` の一覧に `model/adapter_model.safetensors`、`effective_config.yaml`、`tokenizer/`、`training_info.json` がある

オプションは環境変数で指定します。

| 環境変数 | 既定値 | 内容 |
| --- | --- | --- |
| `NPROC` | `1` | GPU が複数あれば増やす |
| `MODEL_TAR` | なし | `model.tar.gz` のパスを指定すると `model` チャネル（tar.gz の展開経路）も検証する |

引数はそのまま `train.py` に渡されます。たとえば `./local_sm_sim.sh --rslora_alpha 256` で rsLoRA の alpha 換算を試せます。

## 完了の確認

3 つのステップがすべて通れば、コンテナと `train.py` は SageMaker 上でもそのまま動く状態です。  
`01_container_build.md` ステップ6 に戻って ECR へ push し、`03_training_job.md` へ進んでください。

ローカル PC で観測した値（スループット、VRAM、loss の推移）は `reference/04_verification_log.md` 4 章に記録しています。  
SageMaker 上の値と比較するときの参考にしてください。
