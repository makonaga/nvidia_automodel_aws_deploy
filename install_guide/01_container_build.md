# コンテナイメージ作成ガイド

このドキュメントでは、AWS Deep Learning Container（DLC）をベースに NeMo AutoModel を載せたコンテナイメージをローカル PC でビルドし、自アカウントの ECR へ push する手順を説明します。  
ベース DLC の選定理由は `reference/03_dlc_selection.md` を参照してください。

> **対象リージョン**: `us-west-2`（オレゴン）  
> 別リージョンで実施する場合は `REGION` の値を読み替えてください。ECR イメージは Training Job と同一リージョンに置く必要があります。

## 検証状況

| 項目        | 内容                                                                                                  |
| --------- | --------------------------------------------------------------------------------------------------- |
| ビルド環境     | Linux PC（Docker Engine、NVIDIA GeForce RTX 3090）                                                     |
| 成果物       | `nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130`（ローカル 19.7 GB、ECR 上の圧縮サイズ約 9.6 GB）               |
| ベース DLC   | `pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker`（torch 2.10.0、Python 3.13、CUDA 13.0） |
| AutoModel | 0.6.0（`constraints.txt` で lock と同じバージョンに pin）                                                       |
| 追加パッケージ   | `flash-linear-attention 0.4.2`（Qwen3.5 の線形 attention 用）、`causal-conv1d 1.6.0`（nvcc でコンパイル）          |

## 前提条件

- Docker Desktop または Docker Engine。空きディスク 60 GB 以上（DLC 本体が 20 GB 超）
- AWS CLI v2 が設定済みで、`aws sts get-caller-identity` が通ること
- IAM 権限として、DLC の pull に `ecr:GetAuthorizationToken`、`ecr:BatchGetImage`、`ecr:GetDownloadUrlForLayer`、自アカウントの ECR に `ecr:CreateRepository`、`ecr:DescribeRepositories`、`ecr:PutImage`、`ecr:InitiateLayerUpload`、`ecr:UploadLayerPart`、`ecr:CompleteLayerUpload`、`ecr:BatchCheckLayerAvailability`
- Apple Silicon Mac の場合は `--platform linux/amd64` でビルドします（スクリプトに含まれています）。エミュレーションのため時間はかかります。本プロジェクトでは Apple Silicon でのビルドは未検証です

## `container/` ディレクトリの内容

| ファイル                  | 役割                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `Dockerfile`          | DLC + `pip install nemo-automodel==0.6.0 flash-linear-attention` + `causal-conv1d` + ビルド時検証 |
| `constraints.txt`     | AutoModel v0.6.0 の `uv.lock` に合わせた pin。torch の差し替え防止                                        |
| `build_and_push.sh`   | DLC アカウントへのログイン → build → ECR リポジトリ作成 → push を一括実行                                          |
| `smoke_test.sh`       | ビルド済みイメージの import / CLI / toolkit / `pip check` を確認（`02_local_verification.md`）             |
| `local_train_test.sh` | ローカル GPU でイメージ内から短い LoRA 学習を実行（`02_local_verification.md`）                                  |
| `local_sm_sim.sh`     | SageMaker の `/opt/ml` 規約を再現して `train.py` を検証（`02_local_verification.md`）                    |

---

## ステップ1: リポジトリを取得する

```bash
git clone git@github.com:Panasonic-LAS-SoftArch/coa_semantic_knowledge_bridge.git
cd nvidia_automodel_aws_deploy/container
```

## ステップ2: AWS の設定を確認する

```bash
aws sts get-caller-identity
export REGION=us-west-2      # SageMaker を使うリージョンに合わせる
```

`build_and_push.sh` はアカウント ID を `aws sts get-caller-identity` から取得するため、アカウント固有の値を書き換える必要はありません。

## ステップ3: ベース DLC を先に pull する（任意）

```bash
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$REGION.amazonaws.com
docker pull --platform linux/amd64 \
  763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker
```

`763104351884` は AWS 公式 DLC のアカウントです。  
`build_and_push.sh` の中でも pull されるので省略できますが、先に実行しておくと DLC の pull（20 GB 超）と AutoModel の install を切り分けて確認できます。  
DLC アカウントへのログイントークンは 12 時間で失効します。

## ステップ4: ビルドする（push なし）

```bash
REGION=$REGION ./build_and_push.sh --no-push
```

所要時間は DLC の pull を除いて 15 分前後です（実測。大半が causal-conv1d の nvcc コンパイルで約 9 分）。

ビルド中に次が自動で検証され、失敗するとビルドが止まります。

- `torch` が `2.10.0` のまま（pip に差し替えられていない）
- `nemo_automodel`、`transformers`、`datasets`、`torchao`、`fla`、`transformers.models.qwen3_5` の import
- `nemo_automodel.cli.app.main` の import（CLI エントリポイント）
- `causal_conv1d` と `causal_conv1d_cuda` の import
- `pip check` で新規の依存衝突がないこと

`pip check` の出力は次の形になります。`[info]` と `[warn]` は想定内で、`[error]` が出るとビルドが止まります。

```
[info] ベース DLC 由来の既存衝突 (無視):
    aiobotocore 3.9.1 has requirement botocore<1.43.76,>=1.43.66, but you have botocore 1.43.89.
    skops 0.14.0 requires prettytable, which is not installed.
[warn] この層で生じたが許容済みの衝突 (学習経路では未使用):
    s3fs 2026.7.0 has requirement fsspec<2026.7.1,>=2026.7.0, but you have fsspec 2025.9.0.
[ok] pip check: 新規の依存衝突なし
```

`s3fs` と `fsspec` の衝突を許容している理由は `reference/04_verification_log.md` 1.1 を参照してください。

ビルドは `--progress=plain` で実行されるため、各ステップの出力がそのまま表示されます。  
ステップがキャッシュ済み（`CACHED` 表示）の場合は出力が再表示されません。その場合はイメージ内から直接確認できます。

```bash
docker run --rm --entrypoint "" nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130 pip check
```

最後に `built: nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130  size=19.7GB` のように表示されれば成功です。

## ステップ5: ローカルで動作確認する

push する前に、`02_local_verification.md` のステップ1（スモークテスト）を実行してください。  
GPU があれば同ガイドのステップ2 とステップ3 も実行し、学習と `train.py` が通ることを確認してから push します。

## ステップ6: ECR へ push する

```bash
REGION=$REGION ./build_and_push.sh
```

ステップ4 のキャッシュが使われるので再ビルドは一瞬で終わり、push だけが走ります。  
アップロードは圧縮後の約 9.6 GB です。  
`docker push` の出力で大半のレイヤが `Layer already exists` になるのは、ベース DLC 由来のレイヤが既に自アカウントのリポジトリにあるためで正常です。

完了時に次が表示されます。この URI を Notebook が `image_uri` として使います（Notebook はアカウント ID とリージョンから自動で組み立てるため、書き写す必要はありません）。

```
image_uri = <account-id>.dkr.ecr.us-west-2.amazonaws.com/nemo-automodel-sagemaker:0.6.0-pt2.10-py313-cu130
```

## ステップ7: ECR 上で確認する

```bash
aws ecr describe-images --repository-name nemo-automodel-sagemaker --region $REGION \
  --query 'imageDetails[].{tag:imageTags[0],sizeGB:imageSizeInBytes,pushed:imagePushedAt}' --output table
```

`0.6.0-pt2.10-py313-cu130` のタグが表示されれば完了です。

---

## ビルドオプション

`build_and_push.sh` は環境変数で挙動を変えられます。

| 環境変数                | 既定値                                            | 内容                                      |
| ------------------- | ---------------------------------------------- | --------------------------------------- |
| `REGION`            | `us-west-2`                                    | ECR のリージョン                              |
| `REPO`              | `nemo-automodel-sagemaker`                     | ECR リポジトリ名                              |
| `TAG`               | `0.6.0-pt2.10-py313-cu130`                     | イメージタグ                                  |
| `AUTOMODEL_VERSION` | `0.6.0`                                        | `pip install nemo-automodel==<version>` |
| `DLC_TAG`           | `2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` | ベース DLC のタグ                             |

`Dockerfile` の build arg で causal-conv1d の導入と CUDA アーキテクチャを変えられます。

| build arg               | 既定値               | 内容                                                            |
| ----------------------- | ----------------- | ------------------------------------------------------------- |
| `INSTALL_CAUSAL_CONV1D` | `1`               | `0` で causal-conv1d を外す（packed sequence で Qwen3.5 を学習しない場合のみ） |
| `TORCH_CUDA_ARCH_LIST`  | `8.0;8.6;8.9;9.0` | A100 / RTX 30 / RTX 40 / H100。他の GPU を使う場合は追加する               |

## 再ビルドが必要になるとき

`train.py`、設定 YAML、Notebook の変更では再ビルドは不要です（`source_dir` でジョブごとに配布されます）。  
再ビルドが必要なのは次の場合だけです。

- `nemo-automodel` のバージョンを上げる。`AUTOMODEL_VERSION=0.7.0 ./build_and_push.sh` のように指定し、`constraints.txt` を新バージョンの `uv.lock` に合わせて更新する
- 追加の pip パッケージが必要になる。`Dockerfile` の `pip install` 行に追加する
- ベース DLC を変える。`DLC_TAG` を指定する。Python 3.13 起因の問題が出た場合のフォールバックは `reference/03_dlc_selection.md` 7 章を参照

## まとめ

このガイドで、AutoModel 0.6.0 を載せたイメージが自アカウントの ECR に配置されました。  
GPU のあるローカル PC では `02_local_verification.md` で学習まで確認してから、GPU が無ければスモークテストだけ確認してから、`03_training_job.md` へ進んでください。
