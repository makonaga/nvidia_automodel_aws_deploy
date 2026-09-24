# 検証記録（問題・原因・判断と実測値）

作業日: 2026-09-14〜15、2026-09-24（初回 SageMaker ジョブ）
検証環境: ローカル PC（NVIDIA GeForce RTX 3090 24 GB、Docker Engine、AWS us-west-2）
成果物: `nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130`（19.7 GB。DLC 単体とほぼ同サイズ。ECR 上は圧縮で約 9.6 GB）
ECR: `290918126236.dkr.ecr.us-west-2.amazonaws.com/nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130`（2026-09-15 push）

本ドキュメントは「発生した問題 → 原因 → 判断」と実測値の記録です。手順は `../01_container_build.md` 以降のガイド、
スクリプトの実装詳細は各ファイルのコメントに委ねます。アカウント固有の値（アカウント ID、バケット名）は実施時のものです。

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
既存出力と突き合わせて一致していれば、`02_composer_migration.md` 5.2 の Step 2（自前 dataset）は不要です。
なお `answer_only_loss_mask: true` ではこの空ブロックも assistant 側として損失対象になるため、
モデルは「空の think ブロックを出してから答える」ことを学習します（非思考モードの配信と整合）。

### 2.4 `train.py` の模擬検証（`local_sm_sim.sh`）

SageMaker の `/opt/ml` 構造と `SM_*` 環境変数を再現し、toolkit と同じ `torchrun ... train.py` で 10 ステップ実行。
チャネルのパス写像、実効設定の書き出し（`/opt/ml/output/data/effective_config.yaml`）、cosine スケジューラ、
エポック末の検証とチェックポイント、`LOWEST_VAL` の選択、`/opt/ml/model` への成果物コピー
（`model/adapter_model.safetensors`、`adapter_config.json`、`tokenizer/`、`training_info.json`、`training.jsonl`）まで動作した。
判明した不具合: transformers 5 の `apply_chat_template(tokenize=True)` は dict を返すため
pack 数の見積もりが壊れていた。修正後の再検証で `avg_tokens=127 → packs≈34`（実際は 33 pack）、
`warmup_epochs 0.5 → lr_warmup_steps 4` となり、LR が 4 ステップで立ち上がってから cosine 減衰することを確認した。

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

### 4.1 MTP 有効・packed (neat) 構成（2026-09-15、同じモデル・LoRA dim 8、seq 512、pack 1024、local batch 2）

| 項目 | 値 |
|---|---|
| スループット | 約 5,500 tokens/s（padded の約 4 倍。パディングが無くなり 1 pack に複数サンプルが詰まるため） |
| VRAM | 約 14.5 GiB（padded の 3.4 GiB から大幅増。pack 長 1024 の 4D block-causal mask と MTP ヘッドの追加計算による） |
| val loss | 約 2.38（padded・MTP 無効の 2.31 と同水準。MTP 損失が train loss に加算されるため、train loss は単純比較できない） |
| 1 エポックの pack 数 | 33 pack（245 サンプル、平均 127 トークン）。`global_batch_size: 8` では **1 エポック = 4 ステップ** |

**気づき: packed ではバッチサイズの単位が「サンプル」から「pack」に変わる。** `global_batch_size: 8` は
8 pack ≈ 60 サンプル前後に相当し、1 エポックのステップ数が padded の 1/8 程度になる。
初回の packed 実行が「step 4 で止まった」ように見えたのはエラーではなく、`num_epochs: 1` の
エポック末で正常終了しただけだった（`max_steps: 20` には届かない）。この性質から:

- ローカル設定は `num_epochs` を増やして `max_steps` で打ち切る形にした
- SageMaker 用の既定は `global_batch_size: 4`（pack 単位）に下げ、Notebook のコメントに pack 換算を書いた
- `warmup_epochs` からウォームアップ step 数を求めるには pack 数の見積もりが必要で、`train.py` が
  データを tokenize して算出する（§2.4）。padded 時はサンプル数から直接求める

VRAM の増加は `packed_sequence_size` にほぼ比例するため、7B〜13B 級で A100 80GB を使うときは
pack 長 2048〜4096 と `local_batch_size: 1` から始めて上げていく。

---

## 4.2 SageMaker 上での初回ジョブ（2026-09-24、ml.g5.2xlarge = 1×A10G 24 GB）

`notebooks/01_launch_training_job.ipynb` を Studio から実行。Qwen3.5-0.8B、LoRA dim 16、seq 1024、pack 2048、
global batch 4 pack、3 エポック。**一発で完走**し、ローカル模擬検証（§2.4）と同じ経路が SageMaker 上でも動いた。

| 項目 | 値 |
|---|---|
| ジョブ全体 | 8 分（起動〜Completed）。課金 443 秒 |
| イメージ pull | 約 3 分（ECR 上 9.6 GB。ローカルの見積もり 5〜10 分より速い） |
| モデル DL | 16 秒（HF Hub → 2.1 GB。SageMaker の回線は速い） |
| packing | 見積もり 17 pack、実際 17 pack（充填率 92.9%）。1 エポック 5 ステップ、計 15 ステップ、warmup 5 |
| 最初のステップ | 102 秒（Triton コンパイル）。最初の検証も 42 秒（eval 経路のコンパイル）。以降は 2 秒/ステップ、検証 5 秒 |
| スループット | 約 3,900 tokens/s（A10G。RTX 3090 の 5,500 より低い） |
| VRAM | 14.5 GiB（pack 2048 × local batch 1。ローカルの pack 1024 × batch 2 と同じトークン数で同じ値） |
| loss | train 3.28 → 2.41、val 2.56 → 2.45 → 2.42（エポック末ごと）。`LOWEST_VAL` = `epoch_2_step_14` |
| 成果物 | `model.tar.gz` に `model/adapter_model.safetensors`、`adapter_config.json`、`automodel_peft_config.json`、tokenizer、`config.yaml`、`effective_config.yaml`、`training.jsonl`、`validation.jsonl`、`training_info.json` |

**気づき**

- `/opt/ml/checkpoints` を `checkpoint_s3_uri` と同期するエージェントが、アップロード済みファイルごとに
  `<name>.sagemaker-uploaded` というマーカーを置く。チェックポイントの `model/` をそのまま `/opt/ml/model` へコピーすると
  このマーカーも `model.tar.gz` に入る（無害だが紛らわしい）。`train.py` のコピーで除外するようにした
- ジョブ全体 8 分のうち学習は 3 分。短い反復ではイメージ pull（3 分）と Triton コンパイル（約 2.5 分）が半分を占めるため、
  反復開発ではウォームプール（`keep_alive_period_in_seconds`）が効く
- `Fetching 13 files` の停止（ローカルで見えた症状）は SageMaker では出ない。回線が速く 16 秒で終わる
- CloudWatch のログには `httpx` の HF Hub アクセスログが大量に出る。学習ログを読むときは `step ` や `[val]` でフィルタする
- DLC 起動時の `CUDA compat package should be installed for NVIDIA driver smaller than 580.178.04 / Current installed ... 595.91.07 / Skipping CUDA compat setup` は
  DLC の起動スクリプトの情報表示。ホストのドライバが新しいので compat 層は不要という意味で正常

## 4.3 8 GPU（ml.p4d.24xlarge = 8×A100 40GB）での確認（2026-09-24）

「実行構成」セルの `target='p4d'` だけ変えて再実行（global batch 8 pack = local 1 × 8 GPU、3 エポック）。**完走**。

| 項目 | 値 |
|---|---|
| 起動 | `waiting for capacity` 6 分 + インスタンス準備 3.5 分（g5.2xlarge の 1 分に比べ p4d は確保に時間がかかる）。イメージ pull は 30 秒未満 |
| 分散 | toolkit が `torchrun --nnodes 1 --nproc_per_node 8`。`rank 0/8`〜`7/8`、`World size: 8`、`NCCL 2.28.9+cuda13.0`。モジュール名が `FSDPQwen3_5ForConditionalGeneration` / `FSDPQwen3_5DenseBlock` になり FSDP2 でシャーディングされている |
| ステップ | 17 pack / 8 = **1 エポック 2 ステップ**（余り 1 pack は捨てられる）、計 6 ステップ。`train.py` は ceil で 3 と見積もっていたので floor に修正 |
| 最初のステップ | 121 秒（8 rank が並列に Triton コンパイル）。最初の検証 41 秒 |
| スループット | 定常 21,000〜30,000 tokens/s（2,600〜3,800 /GPU）。1 GPU の A10G（3,900）とほぼ同じ /GPU 値で、データが小さすぎて（1 GPU あたり 1 pack、1 エポック 2 ステップ）オーバーヘッド支配。スケーリングの評価には本番規模のデータが必要 |
| VRAM | 14.1 GiB/GPU（1 GPU の 14.5 GiB とほぼ同じ）。0.8B のパラメータは 1.6 GB しかなく、メモリは pack 2048 の活性化と MTP が支配的なので FSDP2 で分散しても減らない。7B 級ではパラメータ・オプティマイザ状態の分散が効く |
| loss | train 3.24 → 2.70、val 2.65 → 2.57 → 2.51。1 GPU の 15 ステップより学習量が少ない（6 ステップ、バッチ 8）ため高めで、期待どおり |
| 課金 | 289 秒 |

**気づき**

- `Model parameters are DTensors (FSDP2) — skipping fp32 parameter restoration ... Only buffers will be restored to fp32` の警告:
  `rms_norm: torch_fp32` の RMSNorm 重みを fp32 に戻す処理が FSDP2 では（パラメータ群の dtype を揃える必要があるため）スキップされる。
  RMSNorm の計算自体は fp32 で行われるので実害なし。1 GPU（FSDP2 なし）では出ない
- `[Gloo] Rank N is connected to 7 peer ranks` が 8 rank 分×複数回、行が混ざって出るが無害
- `barrier(): using the device under current context` の UserWarning はチェックポイント保存時の torch の注意で無害
- `sm_train` の行（`dataset ←`、`packing 見積もり` など）は各 rank が出すので 8 回並ぶ。`effective config` と `Sample Prompt` は rank 0 のみ
- p4d は確保待ちが数分ある。`waiting for capacity` が 10 分を超えるようならリージョン内の在庫不足で、時間をずらすか On-Demand Capacity Reservation を検討する

## 5. 未検証・残課題

- ~~MTP 有効 + packed (neat) + causal-conv1d の構成でのローカル学習テスト~~ → 2026-09-15 完了（§4.1）
- ~~SageMaker 上での初回ジョブ（ml.g5.2xlarge）~~ → 2026-09-24 完了（§4.2）
- ~~Phase 6-1: 複数 GPU~~ → 2026-09-24 ml.p4d.24xlarge で完了（§4.3）。8 GPU でのスループット倍率は本番規模データで再評価
- Phase 6-2/6-3: `checkpoint_s3_uri` からの再開、Spot 中断・再開
- LoRA を本体と MTP ヘッドにマージして HF 形式で書き出すツール（Phase 7。vLLM / SGLang 配信に必須）
- DLC 同梱の TE 2.11 で TE attention / TE Linear が動くことは確認できたが、AutoModel の要求（2.14）との差は未評価
- MTP の padded 経路の形状エラーを AutoModel に報告するか
