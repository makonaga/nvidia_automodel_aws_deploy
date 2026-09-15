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

### 2.2 MTP ヘッドが自動で有効化され、padded バッチで形状エラーになった

Qwen3.5 のチェックポイントは MTP（multi-token prediction）ヘッドの重みを同梱しており、AutoModel は
HF config の `num_nextn_predict_layers` を見て学習時に MTP 損失を自動で有効化します。この経路で、
パディング付きバッチ（`packed_sequence_size: 0`）の 2D attention mask がそのまま SDPA に渡され、

```
RuntimeError: The expanded size of the tensor (129) must match the existing size (2) at non-singleton dimension 2.
Target sizes: [2, 8, 129, 129]. Tensor sizes: [2, 129]
```

で失敗しました（traceback に `self.mtp(` を含む）。上流の `main` でも同じ実装で、修正は入っていません。

**判断の経緯**: 当初は `model.num_nextn_predict_layers: 0` で MTP を無効化しました（LoRA SFT の精度には
影響しない）。しかし推論で vLLM / SGLang の MTP 投機的デコーディングを使う方針が確定したため、
**MTP を有効にしたまま学習する方針に変更**しました。理由は、vLLM で LoRA アダプタを重ねて MTP を使うと
ドラフト（元の MTP ヘッド）とターゲット（アダプタ付き本体）の予測がずれて受理率が下がることが報告されており
（vLLM RFC #44826）、MTP ヘッドも本体と一緒に学習して整合を保つ必要があるためです。

**MTP を有効にするための条件**（ソースで確認）:

| 条件 | 根拠 |
|---|---|
| packed sequence（`packing_strategy: neat`）で学習する | MTP は文書 index 付き mask（`_packed_seq_ids`）から block-causal mask を作る経路（`_mtp_block_causal_mask`、NVBugs 6330129）だけが正しく動く。padded バッチの経路は上記の通り壊れている |
| `causal-conv1d` をイメージに入れる | packed 時、Qwen3.5 の GDN 層は `causal_conv1d_fn` に `seq_idx` を渡して文書境界を守る。無いと `F.conv1d` にフォールバックし、pack 内の別文書の末尾 3 トークンが混入する。torch 2.10 / CUDA 13 / py3.13 向けのビルド済み wheel は GitHub Releases に無く、nvcc でコンパイルする（10 分前後） |
| `model.backend.attn: sdpa` を明示する | neat の collater は sdpa 向けに 4D block-causal mask を作り、index mask を `_packed_seq_ids` として別途渡す。TE バックエンドは 4D mask を padding mask として解釈するため経路が異なる |
| 検証（validation）は padded のままでよい | MTP は `self.training` のときだけ実行されるため、eval では経路に入らない |

無効化していたときに比べ、学習対象パラメータは 3.64M → 3.86M に戻ります（MTP ヘッドにも LoRA が付く）。

**配信側の含意**: LoRA アダプタを重ねる配信では学習した MTP ヘッドが使われないため、
**LoRA を本体と MTP ヘッドの両方にマージした HF 形式のフルチェックポイント**を作って配信する必要があります。
AutoModel 同梱の `tools/merge_lora.py` は HF の `AutoModelForCausalLM` + `PeftModel` 経由でマージするため、
transformers 側で `mtp.*`（と `model.visual.*`）が読み飛ばされ、出力に MTP の重みが含まれません。
AutoModel のネイティブモデル上でマージし（`LinearLoRA.materialize_effective_weight`）、state dict adapter で
HF キー（`mtp.layers.0.*`）に戻して書き出すツールを別途用意します（Phase 7）。

### 2.2b AutoModel の attention / Linear バックエンドは既定で TE になる

`BackendConfig` は TE が import でき CUDA が使える環境では `attn` と `linear` の既定が `"te"` になります。
DLC には TE 2.11 が入っているため、ローカルテスト（MTP 無効・padded）は TE attention と TE Linear で動いていました。
問題は出ていませんが、packed の mask 経路と後段の LoRA マージ（TE Linear ではなく torch Linear の方が単純）を
考え、YAML で `backend: {attn: sdpa, linear: torch, rms_norm: torch_fp32}` を明示する方針にしました。

### 2.2c チェックポイントの自動再開は設定変更に弱い

AutoModel は `checkpoint.restore_from` が未指定だと **`checkpoint_dir` にある最新のチェックポイントから
自動で再開**し、モデル構成が非互換でも「警告を出して続行」します（`base_recipe.load_checkpoint`）。
MTP を有効にして再実行した際、前回（MTP 無効）のチェックポイントが残っていたため自動再開が走り、
LoRA のキー不足（`base_model.model.mtp.layers.0.*.lora_A.weight` など 16 件）の警告の後、
オプティマイザ状態の読み込みで `TypeError: cannot pickle code objects` になりました。

自動再開そのものは Spot 中断からの復帰に必要な挙動です。運用ルールとして、
**モデル・PEFT・packing の設定を変えるときは `checkpoint_s3_uri` の prefix（Notebook の `RUN_TAG`）を変える**か、
`train.py` の `fresh_start: 1` で既存チェックポイントを捨てます。ローカルテストは既定で消してから始めます。

### 2.3 チャットテンプレートのレンダリング

`train.py` が学習前に出力する `=== Sample Prompt 0 ===` は次の形でした（2026-09-15）。

```
<|im_start|>user
「いちょう切り」とはどのような切り方ですか？<|im_end|>
<|im_start|>assistant
<think>

</think>

いちょう切りは、…<|im_end|>
```

assistant 応答の前に空の `<think>\n\n</think>\n\n` ブロックが入ります。Qwen3 系テンプレートが非思考モードで
出す形式で、既存 Composer 版の `enable_thinking=False` も同じ空ブロックを出していたはずです。
既存出力と突き合わせて一致していれば、`docs/01` 5.2 の Step 2（自前 dataset）は不要です。
なお `answer_only_loss_mask: true` ではこの空ブロックも assistant 側として損失対象になるため、
モデルは「空の think ブロックを出してから答える」ことを学習します（非思考モードの配信と整合）。

### 2.4 `train.py` の模擬検証（`local_sm_sim.sh`）

SageMaker の `/opt/ml` 構造と `SM_*` 環境変数を再現し、toolkit と同じ `torchrun ... train.py` で 10 ステップ実行。
チャネルのパス写像、実効設定の書き出し（`/opt/ml/output/data/effective_config.yaml`）、cosine スケジューラ、
エポック末の検証とチェックポイント、`LOWEST_VAL` の選択、`/opt/ml/model` への成果物コピー
（`model/adapter_model.safetensors`、`adapter_config.json`、`tokenizer/`、`training_info.json`、`training.jsonl`）まで動作した。
判明した不具合: transformers 5 の `apply_chat_template(tokenize=True)` は dict を返すため
pack 数の見積もりが壊れていた（修正済み）。

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

- **MTP 有効 + packed (neat) + causal-conv1d の構成でのローカル学習テスト**（方針変更後の再検証。次の作業）
- LoRA を本体と MTP ヘッドにマージして HF 形式で書き出すツール（Phase 7。vLLM / SGLang 配信に必須）
- DLC 同梱の TE 2.11 で TE attention / TE Linear が動くことは確認できたが、AutoModel の要求（2.14）との差は未評価
- 複数 GPU での FSDP2（ローカルは 1 GPU のため未検証。SageMaker の `ml.g5.12xlarge` で確認予定）
- MTP の padded 経路の形状エラーを AutoModel に報告するか
