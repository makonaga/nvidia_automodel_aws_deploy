# コンテナイメージの作成手順

AWS DLC `pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` をベースに
NeMo AutoModel 0.6.0 を載せたイメージを作り、自アカウントの ECR へ push します。
選定理由は [../docs/02_dlc_selection.md](../docs/02_dlc_selection.md) を参照。

| ファイル | 役割 |
|---|---|
| `Dockerfile` | DLC + `pip install nemo-automodel==0.6.0 flash-linear-attention` + `causal-conv1d`（nvcc でコンパイル、10 分前後）+ ビルド時検証。`--build-arg INSTALL_CAUSAL_CONV1D=0` で causal-conv1d を外せる |
| `local_train_test.sh` | ローカル GPU でイメージ内から 20 ステップの LoRA 学習を回す（Step 5.5） |
| `local_sm_sim.sh` | SageMaker の `/opt/ml` 規約と `SM_*` 環境変数を再現して `train.py` を検証（Step 5.6） |
| `constraints.txt` | AutoModel v0.6.0 の `uv.lock` に合わせた pin。torch の差し替え防止 |
| `build_and_push.sh` | DLC ログイン → build → ECR リポジトリ作成 → push を一括実行 |
| `smoke_test.sh` | ビルド済みイメージの import / CLI / toolkit / `pip check` を確認 |

## 前提

- Docker Desktop（または Docker Engine）。**空きディスク 60 GB 以上**を推奨（DLC 本体が 20 GB 超）
- AWS CLI v2 が設定済み（`aws sts get-caller-identity` が通ること）
- IAM 権限: `ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`（DLC の pull）、
  自アカウントの ECR に対する `ecr:CreateRepository`, `ecr:DescribeRepositories`, `ecr:PutImage`,
  `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:BatchCheckLayerAvailability`
- Apple Silicon Mac の場合: `--platform linux/amd64` でビルドします（スクリプトに含まれています）。
  pip install は pure wheel のみなのでエミュレーションでも完走しますが、時間は長めです

## 手順

### Step 1: リポジトリを取得して `container/` へ移動

```bash
git clone https://github.com/makonaga/nvidia_automodel_aws_deploy.git
cd nvidia_automodel_aws_deploy
git checkout claude/automodel-sagemaker-integration-w5kdwn
cd container
```

### Step 2: AWS の設定を確認

```bash
aws sts get-caller-identity
export REGION=us-west-2      # SageMaker を使うリージョンに合わせる
```

ECR イメージは **Training Job と同一リージョン**に置く必要があります。

### Step 3: ベース DLC を先に pull（任意、所要時間の見積もり用）

```bash
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$REGION.amazonaws.com
docker pull --platform linux/amd64 \
  763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker
```

`build_and_push.sh` の中でも pull されるので省略可能ですが、
先に実行しておくと DLC の pull と AutoModel の install を切り分けて確認できます。

### Step 4: ビルド（push なし）

```bash
REGION=$REGION ./build_and_push.sh --no-push
```

ビルド中に以下が自動で検証されます。失敗するとビルドが止まります。

- `torch` が `2.10.0` のまま（pip に差し替えられていない）
- `nemo_automodel` / `transformers` / `datasets` / `torchao` の import
- `nemo_automodel.cli.app.main` の import（CLI エントリポイント）
- `pip check` で依存衝突がないこと

ログに各パッケージのバージョンが出力されるので、`docs/02_dlc_selection.md` の表と照合してください。
成功時の pip check の出力は次の形です（`[info]` と `[warn]` は想定内、`[error]` が出るとビルドが止まります）。

```
[info] ベース DLC 由来の既存衝突 (無視):
    aiobotocore 3.9.1 has requirement botocore<1.43.76,>=1.43.66, but you have botocore 1.43.89.
    skops 0.14.0 requires prettytable, which is not installed.
[warn] この層で生じたが許容済みの衝突 (学習経路では未使用):
    s3fs 2026.7.0 has requirement fsspec<2026.7.1,>=2026.7.0, but you have fsspec 2025.9.0.
[ok] pip check: 新規の依存衝突なし
```

ステップがキャッシュ済み（`CACHED` 表示）の場合は出力が再表示されません。その場合は
`docker run --rm --entrypoint "" nemo-automodel-sagemaker:<TAG> pip check` でイメージ内から直接確認できます。

### Step 5: ローカルで動作確認

```bash
./smoke_test.sh
```

確認内容: import とバージョン、`automodel --help`、SageMaker toolkit のエントリポイント、`pip check`。
GPU があるマシンでは flash-attn と TransformerEngine の import も確認します。
GPU のない PC では `cuda available: False` と出ますが、それ自体は問題ありません。

### Step 5.5: ローカル GPU で学習テスト（GPU がある場合のみ）

イメージ内から Qwen3.5-0.8B の LoRA 学習を 20 ステップだけ回し、
SageMaker と同じ経路（torchrun → `nemo_automodel.cli.app` → in-process 実行）で
モデルのロード、cooking データの読み込み、LoRA、チェックポイント保存までを検証します。

```bash
./local_train_test.sh                              # 既定: Qwen/Qwen3.5-0.8B
MODEL_ID=Qwen/Qwen3-0.6B ./local_train_test.sh      # 別モデルで試す場合
```

初回は HF Hub からモデル（約 1.6 GB）をダウンロードします（キャッシュは `<repo>/.hf_cache`）。
最後に `step N | epoch 0 | loss ...` の行と `out/local_test/checkpoints/` の中身が表示されれば成功です。

検証済みの結果（RTX 3090 24 GB、2026-09-14、MTP 無効・padded 構成）: 20 ステップ完走、loss 2.82 → 2.23、val loss 2.40 → 2.31、
VRAM 約 3.4 GiB、約 1,400 tokens/s。最初のステップは fla の Triton コンパイルで 77 秒かかり、以降は 1 秒未満。
MTP 有効・packed (neat) 構成（2026-09-15）: 約 5,500 tokens/s、VRAM 14.5 GiB、val loss 2.38。
packed では 1 エポックのステップ数が少ないため、`max_steps` に届く前にエポック末で正常終了することがあります（エラーではありません）。
Qwen3.5 の線形 attention 層は `flash-linear-attention` の Triton カーネルを使うため、
初回ステップで Triton のコンパイルに数十秒かかります。

### Step 5.6: SageMaker の規約を模して `train.py` を検証（GPU がある場合のみ）

```bash
./local_sm_sim.sh
```

`out/sm_sim/opt_ml/` に `/opt/ml` の構造（`input/data/{train,validation}`, `code`, `checkpoints`, `model`, `output/data`）を
作り、`SM_*` 環境変数を与えてイメージ内で `torchrun ... /opt/ml/code/train.py` を実行します。
SageMaker の toolkit が行う起動形と同じです。確認するのは次の 3 点です。

1. `=== effective config ===` で `dataset` / `validation_dataset` / `checkpoint_dir` がチャネルのパスに置き換わっている
2. `=== Sample Prompt 0 ===` に Qwen のチャットテンプレートでレンダリングされた 1 件目が出る
3. 最後の `/opt/ml/model` の一覧に `model/adapter_model.safetensors`、`effective_config.yaml`、`tokenizer/`、`training_info.json` がある

`MODEL_TAR=/path/to/model.tar.gz ./local_sm_sim.sh` で `model` チャネル（tar.gz）の経路も検証できます。

### Step 6: ECR へ push

```bash
REGION=$REGION ./build_and_push.sh
```

Step 4 のキャッシュが使われるので再ビルドは一瞬で終わり、push だけが走ります。
完了時に `image_uri = <account>.dkr.ecr.<region>.amazonaws.com/nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130`
が表示されます。この URI を Notebook の `image_uri` に使います。

### Step 7: ECR 上で確認

```bash
aws ecr describe-images --repository-name nemo-automodel-sagemaker --region $REGION \
  --query 'imageDetails[].{tag:imageTags[0],sizeGB:imageSizeInBytes,pushed:imagePushedAt}' --output table
```

## つまずきやすい点

| 症状 | 原因と対処 |
|---|---|
| `pull access denied` / `no basic auth credentials` | DLC アカウント（763104351884）への `docker login` が切れている。トークンは 12 時間で失効するので Step 3 のログインを再実行 |
| `no space left on device` | Docker のディスク容量不足。`docker system prune -a` で不要イメージを削除するか、Docker Desktop の Disk image size を増やす |
| ビルド時検証で `torch が差し替わっています` | 依存解決で torch が再インストールされた。`constraints.txt` の `torch==2.10.0` が効いていないので、pip のログで何が torch を要求したか確認 |
| `pip check` が失敗 | ログの `[error] この層で新たに生じた依存衝突` を確認。ベース DLC に元からある衝突は `[info]` として表示され無視されます。新規の衝突が AutoModel と無関係なパッケージなら `Dockerfile` の `pip uninstall` 行に追加、必要なパッケージなら `constraints.txt` で pin を調整して再ビルド |
| `[warn] 許容済みの衝突: s3fs ... fsspec` | 想定内。`datasets 4.2.0` が `fsspec<=2025.9.0` を要求するため DLC の `s3fs 2026.7.0` と整合しない。`s3fs` はクライアント SDK の `sagemaker-mlops` 専用で学習ジョブでは import されず、消すと `sagemaker` SDK ごと削除になるため、そのまま残して許容している |
| `[info] ベース DLC 由来の既存衝突` | AWS 側の事情（例: `aiobotocore` と `botocore`、`skops` と `prettytable`）。ビルドは止まらない |
| py313 の wheel が無いというエラー | `docs/02_dlc_selection.md` 7 章のフォールバック。`DLC_TAG=2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker` と `constraints.txt` の `torch==2.9.0` に変えて再ビルド |
| Apple Silicon で極端に遅い | エミュレーションのため。pip install は数分〜十数分で終わるはずなので待つ。どうしても遅い場合は EC2 (x86) でビルドする |

## ローカル学習テストでつまずきやすい点

| 症状 | 原因と対処 |
|---|---|
| `Fetching 13 files: 92%` で長く止まる | 最後の `model.safetensors`（約 1.6 GB）をダウンロード中。進捗バーはファイル単位でしか進まない。現在のスクリプトはダウンロードを別ステップにしてバイト単位で表示する |
| `RuntimeError: The expanded size of the tensor (S) must match ... Target sizes: [B, H, S, S]. Tensor sizes: [B, S]`（traceback に `self.mtp(` を含む） | Qwen3.5 の MTP ヘッドはパディング付きバッチでは動かない（AutoModel 0.6.0 の不具合）。`packed_sequence.packing_strategy: neat` と `model.backend.attn: sdpa` で packed にする。MTP を使わないなら `model.num_nextn_predict_layers: 0` でも回避できる |
| `The fast path is not available ... flash-linear-attention` の警告 | fla か causal-conv1d が入っていない古いイメージ。`build_and_push.sh --no-push` で再ビルド。両方入っていればこの警告は出ない |
| `Loading checkpoint from .../epoch_N_step_M` の直後に `Checkpoint key mismatch` や `TypeError: cannot pickle code objects` | `checkpoint_dir` に前回の（設定が違う）チェックポイントが残っていて自動再開した。`local_train_test.sh` は既定で消す（`CLEAN=0` で再開テスト）。SageMaker では `checkpoint_s3_uri` の prefix を変えるか `fresh_start: 1` |
| `[ERROR] ... is part of ...'s signature, but not documented` | transformers の docstring チェック。無害 |
| `grouped_gemm is not available` / `Skipping import of cpp extensions ... torchao` | MoE 用カーネル / torchao の C++ 拡張。dense モデルの LoRA では不要 |

## 再ビルドが必要になるとき

`train.py` や YAML、Notebook の変更では再ビルド不要です（`source_dir` でジョブごとに配布されます）。
再ビルドが必要なのは以下の場合のみです。

- `nemo-automodel` のバージョンを上げる → `AUTOMODEL_VERSION=0.7.0 ./build_and_push.sh` のように指定し、`constraints.txt` を新バージョンの `uv.lock` に合わせて更新
- 追加の pip パッケージが必要 → `Dockerfile` の `pip install` 行に追加
- ベース DLC を変える → `DLC_TAG` を指定
