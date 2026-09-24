# トラブルシューティングガイド

このドキュメントでは、コンテナのビルド、ローカル検証、SageMaker Training Job の実行中に発生する可能性のある問題と、その解決方法をまとめています。  
問題が発生した場合は、このガイドを参照してください。

> **対象リージョン**: `us-west-2`（オレゴン）  
> 別リージョンで実施する場合は、コマンド中の `--region` の値を読み替えてください。

---

## コンテナのビルドと push

### 問題1: `pull access denied` または `no basic auth credentials` でベース DLC を pull できない

**症状**

`build_and_push.sh` や `docker pull` が `pull access denied for 763104351884.dkr.ecr...` または `no basic auth credentials` で失敗します。

**考えられる原因**

- DLC アカウント（763104351884）への `docker login` が切れています。トークンは 12 時間で失効します
- リージョンが違うホスト名でログインしています

**解決方法**

`01_container_build.md` ステップ3 のログインを再実行してください。

```bash
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$REGION.amazonaws.com
```

---

### 問題2: `no space left on device` でビルドが止まる

**症状**

DLC の pull や `pip install` の途中で `no space left on device` が出ます。

**考えられる原因**

Docker のディスク容量が不足しています。DLC 本体が 20 GB 超、ビルド後のイメージが 19.7 GB あり、中間レイヤも含めて 60 GB 程度必要です。

**解決方法**

`docker system prune -a` で不要なイメージを削除するか、Docker Desktop の Disk image size を増やしてください。

---

### 問題3: ビルド時の検証で `torch が差し替わっています` と出る

**症状**

`Dockerfile` の検証ステップが `torch が差し替わっています` で失敗します。

**考えられる原因**

依存解決の過程で pip が torch を別バージョンに再インストールしました。`constraints.txt` の `torch==2.10.0` が効いていません。

**解決方法**

pip のログで何が torch を要求したかを確認してください。`nemo-automodel` のバージョンを変えた場合は、新バージョンの `uv.lock` に合わせて `constraints.txt` を更新します。

---

### 問題4: `pip check` が `[error]` で失敗する

**症状**

ビルド時の `pip check` が `[error] この層で新たに生じた依存衝突` を表示して止まります。

**考えられる原因**

AutoModel の依存とベース DLC のパッケージに新しい衝突が生じました。ベース DLC に元からある衝突は `[info]`、既知の `s3fs` と `fsspec` の衝突は `[warn]` として表示され、ビルドは止まりません。

**解決方法**

`[error]` に表示された衝突を確認します。AutoModel と無関係なパッケージなら `Dockerfile` に `pip uninstall` を追加し、必要なパッケージなら `constraints.txt` で pin を調整して再ビルドします。  
`s3fs` の衝突を許容している理由は `reference/04_verification_log.md` 1.1 を参照してください。

---

### 問題5: Python 3.13 の wheel が無いというエラーが出る

**症状**

`pip install` が `No matching distribution` や `Could not build wheels` で失敗します。

**考えられる原因**

依存パッケージに cp313 の wheel が無く、ソースビルドに失敗しています。AutoModel 0.6.0 の依存については cp313 wheel の存在を確認済みですが、バージョンを変えると発生することがあります。

**解決方法**

`reference/03_dlc_selection.md` 7 章のフォールバックに従い、`DLC_TAG=2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker` と `constraints.txt` の `torch==2.9.0` に変えて再ビルドします。ただし DLC 2.9 は 2026-10-15 にサポートが終了します。

---

### 問題6: Apple Silicon Mac でビルドが極端に遅い

**症状**

`pip install` や causal-conv1d のコンパイルが数時間かかります。

**考えられる原因**

`--platform linux/amd64` によるエミュレーションで動いています。

**解決方法**

`pip install` は数分〜十数分で終わるはずなので待ちます。どうしても遅い場合は x86 の EC2 でビルドしてください。本プロジェクトでは Apple Silicon でのビルドは未検証です。

---

## ローカル検証

### 問題7: `Fetching 13 files: 92%` で長く止まる

**症状**

`local_train_test.sh` の初回実行で、モデルのダウンロードが `92%` のまま数分止まります。

**考えられる原因**

最後の `model.safetensors`（約 1.6 GB）をダウンロード中です。進捗バーはファイル単位でしか進みません。

**解決方法**

待ってください。現在のスクリプトはダウンロードを別ステップにしてバイト単位で表示します。SageMaker 上では回線が速く、16 秒程度で終わります。

---

### 問題8: MTP の形状エラーで学習が止まる

**症状**

```
RuntimeError: The expanded size of the tensor (129) must match the existing size (2) at non-singleton dimension 2.
Target sizes: [2, 8, 129, 129]. Tensor sizes: [2, 129]
```

traceback に `self.mtp(` が含まれます。

**考えられる原因**

Qwen3.5 の MTP ヘッドはパディング付きバッチ（`packed_sequence_size: 0`）では動きません（AutoModel 0.6.0 の不具合）。

**解決方法**

`packed_sequence.packing_strategy: neat` と `model.backend.attn: sdpa` で packed sequence にします。このリポジトリの YAML は既定でこの構成です。  
MTP を使わない場合は `model.num_nextn_predict_layers: 0` でも回避できます。詳細は `reference/04_verification_log.md` 2.2 を参照してください。

---

### 問題9: `The fast path is not available ... flash-linear-attention` の警告が出る

**症状**

Qwen3.5 のモデルロード時に fast path が使えないという警告が出て、ステップが遅いです。

**考えられる原因**

`flash-linear-attention` か `causal-conv1d` が入っていない古いイメージを使っています。transformers は両方が揃ったときだけ fast path と呼びます。

**解決方法**

`build_and_push.sh --no-push` で再ビルドしてください。両方入っていればこの警告は出ません。

---

### 問題10: `Checkpoint key mismatch` または `TypeError: cannot pickle code objects` で止まる

**症状**

`Loading checkpoint from .../epoch_N_step_M` の直後に、LoRA のキー不足の警告と `TypeError: cannot pickle code objects` が出ます。

**考えられる原因**

`checkpoint_dir` に前回の（モデルや PEFT の設定が違う）チェックポイントが残っていて、AutoModel が自動で再開しました。AutoModel は構成が非互換でも警告を出して続行し、オプティマイザ状態の読み込みで失敗します。

**解決方法**

ローカルでは `local_train_test.sh` が既定で前回の出力を削除します（`CLEAN=0` で残す）。  
SageMaker では `RUN_TAG` を変えて `checkpoint_s3_uri` の prefix を分けるか、`hyperparameters` の `fresh_start` を `1` にしてください。自動再開そのものは Spot 中断からの復帰に必要な動作です。

---

### 問題11: `rm: cannot remove 'out/...': Permission denied`

**症状**

`out/` 配下のファイルをホストで削除できません。

**考えられる原因**

コンテナ内の root が作ったファイルは、ホストの一般ユーザーでは削除できません。

**解決方法**

現在のスクリプトはコンテナ経由で削除し、終了時に所有者を戻します。手動で直す場合は次を実行してください。

```bash
docker run --rm -v "$PWD/out:/out" nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130 chown -R $(id -u):$(id -g) /out
```

---

### 問題12: packed 構成で `max_steps` より早く終了する

**症状**

`max_steps: 20` のはずが `step 4` で正常終了します。

**考えられる原因**

エラーではなくエポック末です。packed sequence ではバッチの単位が pack なので、1 エポックのステップ数が少なくなります（料理データ、pack 1024、global batch 8 で 4 ステップ）。

**解決方法**

`num_epochs` を増やしてください。ローカル設定は `num_epochs: 10` にしてあります。

---

## ログに出る無害なメッセージ

### 問題13: `[ERROR] `loss` is part of ...'s signature, but not documented` が大量に出る

**症状**

`[ERROR] `loss` is part of Qwen3_5CausalLMOutputWithPast.__init__'s signature, but not documented. Make sure to add it to the docstring ...` という行が、1 GPU で 8 行、8 GPU で 64 行出ます。

**考えられる原因**

transformers 5 系の `@auto_docstring` デコレータが、モデルクラスの import 時に docstring の整合性チェックの結果を `print` しています。例外でも `logging` の ERROR でもありません。Qwen2-VL、Qwen2.5-VL、Qwen3.5、Qwen3.5-MoE の 4 クラス × `loss` / `logits` の 2 引数 × プロセス数だけ出ます。

**解決方法**

対処は不要です。ジョブは `Reporting training SUCCESS` で終わり、loss も正常に下がります。`print` のためログレベルでは消せません。

---

### 問題14: その他の警告

| メッセージ | 正体 |
| --- | --- |
| `grouped_gemm is not available` | MoE 用カーネル。dense モデルには無関係 |
| `Skipping import of cpp extensions ... torchao` | torchao の C++ 拡張が torch 2.10 用に無い。FP8 を使わない限り無関係 |
| `torch_dtype is deprecated! Use dtype instead!` | transformers 5 系の警告 |
| `[Gloo] Rank N is connected to M peer ranks` | 分散初期化のログ。8 GPU では行が混ざって出ることがある |
| `Warning: You are sending unauthenticated requests to the HF Hub` | `HF_TOKEN` 未設定。public モデルなら不要 |
| `rope_fusion is temporarily force-disabled globally` | AutoModel 側の既知事項 |
| `Model parameters are DTensors (FSDP2) — skipping fp32 parameter restoration ...` | `rms_norm: torch_fp32` の重みを fp32 に戻す処理が FSDP2 ではスキップされる。RMSNorm の計算自体は fp32 で行われるため実害なし。1 GPU では出ない |
| `barrier(): using the device under current context` | チェックポイント保存時の torch の注意。無害 |
| `CUDA compat package should be installed for NVIDIA driver smaller than ... Skipping CUDA compat setup` | DLC の起動スクリプトの情報表示。ホストのドライバが新しいので compat 層は不要という意味 |
| `Setting OMP_NUM_THREADS environment variable for each process to be 1` | torchrun の既定動作 |

---

## SageMaker Training Job

問題15〜21 は本プロジェクトでは発生しておらず、SageMaker の一般的な失敗パターンとして対処を記載しています。問題22 は実際に観測したものです。

### 問題15: `ResourceLimitExceeded` でジョブが作成できない

**症状**

`estimator.fit()` が `ResourceLimitExceeded: ... ml.g5.2xlarge for training job usage` で失敗します。

**考えられる原因**

対象インスタンスの Service Quotas が 0 です。

**解決方法**

コンソールの Service Quotas > AWS services > Amazon SageMaker で `<instance> for training job usage` を検索し、引き上げを申請してください。Spot の場合は `for spot training job usage` が別に必要です。承認には数時間〜数日かかります。

---

### 問題16: プリフライト確認で `AccessDeniedException ... servicequotas:ListServiceQuotas`

**症状**

Notebook のプリフライト確認セルが `quota : 参照権限なし (AccessDeniedException)` と表示します。

**考えられる原因**

Studio の実行ロールに Service Quotas の参照権限がありません。ジョブの実行には影響しません。

**解決方法**

コンソールで目視確認して次へ進んでください。

---

### 問題17: `CannotPullContainerError` または `no basic auth credentials` でジョブが失敗する

**症状**

ジョブが `Downloading` の後に失敗し、`FailureReason` に `CannotPullContainerError` が出ます。

**考えられる原因**

- 実行ロールに自アカウント ECR からの pull 権限がありません
- ECR イメージと Training Job のリージョンが違います

**解決方法**

実行ロールに `AmazonSageMakerFullAccess` 相当を付与し、ECR と同一リージョンで実行してください。プリフライト確認セルで ECR イメージが見つかることを確認します。

---

### 問題18: S3 の `AccessDenied` でジョブが失敗する

**症状**

`Downloading` の段階で `AccessDenied` が出ます。

**考えられる原因**

実行ロールが入力チャネルのバケットを読めません。

**解決方法**

既定バケット以外を使うときは、実行ロールのポリシーにそのバケットへの `s3:GetObject`、`s3:ListBucket`、`s3:PutObject` を追加してください。

---

### 問題19: `Fetching 13 files` で `Connection error` や `Max retries` が出る

**症状**

モデルのダウンロードでネットワークエラーになります。

**考えられる原因**

Training Job から Hugging Face Hub に到達できません（VPC 指定、network isolation など）。

**解決方法**

Estimator の VPC 設定を外すか、ベースモデルを S3 に置いて `model_s3` で `model` チャネルとして渡してください。

---

### 問題20: `torch.OutOfMemoryError`

**症状**

学習の最初のステップで OOM になります。

**考えられる原因**

1 pack（2048 トークン）あたり約 14.5 GiB を使うため、GPU メモリに収まっていません。

**解決方法**

`local_batch_size: 1` でも収まらない場合は、`hyperparameters` に `'packed_sequence.packed_sequence_size': 1024` と `'dataset.seq_length': 512` を追加して pack 長を下げてください。7B 級では A100 80 GB を使い、pack 長 2048〜4096 と `local_batch_size: 1` から始めます。

---

### 問題21: ログが `Training in progress` のまま進まない

**症状**

`Training image download completed. Training in progress.` の後、15 分以上ログが出ません。

**考えられる原因**

イメージの pull（9.6 GB）中です。実測では 3 分程度で終わります。

**解決方法**

20 分を超える場合はコンソールでジョブを Stop し、ECR のリージョンとタグを再確認してください。

---

### 問題22: `waiting for capacity` が長い

**症状**

`ml.p4d.24xlarge` などで `Pending - Training job waiting for capacity` が続きます。

**考えられる原因**

リージョン内の在庫が不足しています。実測では 6 分でした。

**解決方法**

長引く場合は時間をずらして再実行してください（対処の実績はありません）。

---

## ログの取得方法

Notebook に流れるログが長い場合は、JupyterLab のターミナルで CloudWatch から取得できます。

```bash
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <job_name> --region us-west-2 --since 2h > job.log
```

コンソール > Training > Training jobs > 該当ジョブ > Monitor > **View logs** からも同じログを見られます。  
失敗の理由は `describe_training_job` の `FailureReason`（コンソールの Status にも表示）で確認してください。
