# コンテナイメージの作成手順

AWS DLC `pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` をベースに
NeMo AutoModel 0.6.0 を載せたイメージを作り、自アカウントの ECR へ push します。
選定理由は [../docs/02_dlc_selection.md](../docs/02_dlc_selection.md) を参照。

| ファイル | 役割 |
|---|---|
| `Dockerfile` | DLC + `pip install nemo-automodel==0.6.0`（constraints で pin）+ ビルド時検証 |
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

### Step 5: ローカルで動作確認

```bash
./smoke_test.sh
```

確認内容: import とバージョン、`automodel --help`、SageMaker toolkit のエントリポイント、`pip check`。
GPU があるマシンでは flash-attn と TransformerEngine の import も確認します。
GPU のない PC では `cuda available: False` と出ますが、それ自体は問題ありません。

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
| （解決済み）`s3fs ... fsspec` / `aiobotocore ... botocore` の衝突 | `datasets 4.2.0` が `fsspec<=2025.9.0` を要求するため DLC の `s3fs 2026.7.0` と両立せず、`aiobotocore 3.9.1` は DLC の `botocore 1.43.89` と元から非互換。いずれも SageMaker Clarify (`smclarify`) 用で学習には不要なため、`Dockerfile` で 3 つとも削除している |
| py313 の wheel が無いというエラー | `docs/02_dlc_selection.md` 7 章のフォールバック。`DLC_TAG=2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker` と `constraints.txt` の `torch==2.9.0` に変えて再ビルド |
| Apple Silicon で極端に遅い | エミュレーションのため。pip install は数分〜十数分で終わるはずなので待つ。どうしても遅い場合は EC2 (x86) でビルドする |

## 再ビルドが必要になるとき

`train.py` や YAML、Notebook の変更では再ビルド不要です（`source_dir` でジョブごとに配布されます）。
再ビルドが必要なのは以下の場合のみです。

- `nemo-automodel` のバージョンを上げる → `AUTOMODEL_VERSION=0.7.0 ./build_and_push.sh` のように指定し、`constraints.txt` を新バージョンの `uv.lock` に合わせて更新
- 追加の pip パッケージが必要 → `Dockerfile` の `pip install` 行に追加
- ベース DLC を変える → `DLC_TAG` を指定
