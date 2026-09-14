# Phase 2（コンテナ作成）で判明したこと

作業日: 2026-09-14
検証環境: ローカル PC（NVIDIA GeForce RTX 3090 24 GB、Docker Engine、AWS us-west-2）
成果物: `nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130`（19.4 GB。DLC 単体とほぼ同サイズ）

本ドキュメントは「発生した問題 → 原因 → 判断」の記録です。スクリプトの実装詳細は
`container/README.md` と各ファイルのコメントに委ねます。

---

## 1. 依存関係

### 1.1 `pip check` が 2 件の衝突を報告した

| 衝突 | 原因 | 判断 |
|---|---|---|
| `s3fs 2026.7.0` が `fsspec>=2026.7.0` を要求するが `fsspec 2025.9.0` になる | AutoModel の lock にある `datasets 4.2.0` が `fsspec<=2025.9.0` を要求し、DLC 同梱の `s3fs` と整合が取れない。**AutoModel 側の pin が原因** | 許容する（下記） |
| `aiobotocore 3.9.1` が `botocore<1.43.76` を要求するが `botocore 1.43.89` | DLC が `boto3` を後から最新にしているため、**ベース DLC の時点で既に衝突**。aiobotocore の最新版でも 1.43.89 は受け付けない | 無視する（AWS 側の事情） |

`s3fs` を削除して解消を試みたところ、`sagemaker`（v3 メタパッケージ）→ `sagemaker-mlops` → `s3fs` の
依存連鎖で今度は `sagemaker-mlops` が衝突を報告しました。連鎖を断つには `sagemaker` SDK ごと削除する
ことになり DLC からの乖離が大きすぎるため、**DLC のパッケージはすべて残し、この衝突は「学習経路で
未使用」として許容**しました。根拠は、`s3fs` を import するのがクライアント側 SDK の `sagemaker-mlops`
だけで、学習ジョブの経路（`sagemaker-training` → `sagemaker-pytorch-training` → `train.py`）が
`sagemaker` SDK に依存していないことを PyPI の依存情報で確認したことです。

同時に、DLC に元からあった `skops requires prettytable` の衝突は AutoModel の依存が `prettytable` を
入れたことで解消しました。

### 1.2 Qwen3.5 には `flash-linear-attention`（fla）が必要だった

Qwen3.5 は Gated DeltaNet（線形 attention）とフル attention のハイブリッド構造で、線形 attention 部分は
fla の Triton カーネルを前提にしています。AutoModel のコア依存には含まれず（`[fla]` extra）、初回の
イメージには入っていませんでした。無くても transformers の torch フォールバックで動きますが大幅に
遅くなるため、lock と同じ `fla 0.4.2` をイメージに追加しました（pure Python なので追加コストは小さい）。

同じく lock にある `causal-conv1d 1.6.0` は sdist のみで nvcc によるコンパイルが必要なため、既定では
入れず opt-in にしました。無い場合は kernel_size 4 の短い畳み込みが `F.conv1d` にフォールバックするだけで、
影響は軽微です。

**注意**: transformers は `causal_conv1d` と `fla` の**両方**が揃ったときだけ「fast path」と呼ぶため、
fla を入れても `The fast path is not available` の警告は出続けます。fla のカーネル自体は使われています
（最初のステップで Triton のコンパイルに 77 秒かかることが証拠）。

### 1.3 Python 3.13 での問題は出なかった

AutoModel の開発環境は 3.12 ですが、DLC 2.10 の 3.13 で import・学習ともに問題は観測されませんでした。

---

## 2. Qwen3.5 固有の事項

### 2.1 AutoModel 0.6.0 のサポート状況

- `nemo_automodel/components/models/qwen3_5/` にネイティブ実装（`Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration`）があり、
  レジストリが HF の `Qwen3_5ForConditionalGeneration` アーキテクチャをここに振り分ける
- HF の `Qwen/Qwen3.5-0.8B` は VL（画像入力可）のチェックポイントで、AutoModel はビジョンタワーを
  ロードしたうえでテキスト経路だけを使う。LoRA の `target_modules: '*_proj'` はビジョン側の `attn.proj`
  にはマッチしない（`_proj` サフィックスが必要）ため、学習対象はテキスト側のみ
- `transformers 5.12.1` は `qwen3_5` アーキテクチャを含む
- v0.6.0 の `examples/llm_finetune/` に Qwen3.5 用の YAML は無く、`qwen/qwen3_0p6b_hellaswag_peft.yaml` を雛形にした

### 2.2 MTP ヘッドが自動で有効化され、形状エラーになった

Qwen3.5 のチェックポイントは MTP（multi-token prediction）ヘッドの重みを同梱しており、AutoModel は
HF config の `num_nextn_predict_layers` を見て学習時に MTP 損失を自動で有効化します。この経路で、
パディング付きバッチ（`packed_sequence_size: 0`）の 2D attention mask がそのまま SDPA に渡され、

```
RuntimeError: The expanded size of the tensor (129) must match the existing size (2) at non-singleton dimension 2.
Target sizes: [2, 8, 129, 129]. Tensor sizes: [2, 129]
```

で失敗しました（traceback に `self.mtp(` を含む）。MTP は投機的デコーディング用の補助ヘッドで SFT には
不要なため、`model.num_nextn_predict_layers: 0` で無効化しました。無効化により学習対象パラメータは
3.86M → 3.64M、総パラメータは 877M → 857M に減ります（MTP ヘッド分）。

AutoModel 側の不具合と考えられるため、NVIDIA-NeMo/Automodel への報告候補です。

### 2.3 チャットテンプレートの検証は次フェーズ

`use_hf_chat_template: true` で Qwen3.5 のテンプレートが適用されることは確認しましたが、
既存 Composer 版の `enable_thinking=False` と同じレンダリングになるかは未検証です。
`train.py` が学習前に出力する `=== Sample Prompt 0 ===` で突き合わせます（`docs/01` 5.2 参照）。

---

## 3. ログの読み方（無害なメッセージ）

| メッセージ | 正体 |
|---|---|
| `[ERROR] `loss` is part of ...'s signature, but not documented` | transformers の `@auto_docstring` による docstring チェック。`print` されるだけで例外ではない。Qwen2-VL / Qwen2.5-VL / Qwen3.5 / Qwen3.5-MoE の 4 種分が出る |
| `grouped_gemm is not available` | MoE 用カーネル。dense モデルには無関係 |
| `Skipping import of cpp extensions ... torchao` | torchao の C++ 拡張が torch 2.10 用に無い。FP8 を使わない限り無関係 |
| `torch_dtype is deprecated! Use dtype instead!` | transformers 5 系の警告 |
| `[Gloo] Rank 0 is connected to 0 peer ranks` | 単一プロセス時の分散初期化ログ |
| `Warning: You are sending unauthenticated requests to the HF Hub` | `HF_TOKEN` 未設定。public モデルなら不要 |
| `Fetching 13 files: 92%` で長時間停止 | 最後の `model.safetensors`（1.6 GB）をダウンロード中。進捗はファイル単位でしか進まない |

---

## 4. 観測値（Qwen3.5-0.8B、LoRA dim 8、seq 512、local batch 2、RTX 3090）

| 項目 | 値 |
|---|---|
| 20 ステップの loss | 2.82 → 2.23（val 2.40 → 2.31） |
| VRAM | 約 3.4 GiB |
| スループット | 約 1,400 tokens/s（1 GPU） |
| 最初のステップ | 77 秒（fla の Triton カーネルのコンパイル）。以降は 1 秒未満 |
| モデルのロード | 2.1 GB を 0.5 秒 |
| チェックポイント | `epoch_0_step_N/` に保存、`LATEST` / `LOWEST_VAL` のシンボリックリンクが更新される |

SageMaker への含意: Triton のコンパイルはジョブごと・GPU アーキテクチャごとに発生するため、
短いジョブでは相対的に無視できない。反復開発ではウォームプールの利用を検討する。

---

## 5. 未検証・残課題

- `causal-conv1d` を入れた場合の速度差（A100 で評価する）
- DLC 同梱の TE 2.11 と AutoModel の要求 2.14 の差（TE attention を使わないため未検証）
- `attn_implementation: flash_attention_2`（今回は既定の sdpa で検証。A10G / A100 では FA2 が速いはず）
- 複数 GPU での FSDP2（ローカルは 1 GPU のため未検証。SageMaker の `ml.g5.12xlarge` で確認予定）
- MTP の形状エラーを AutoModel に報告するか
