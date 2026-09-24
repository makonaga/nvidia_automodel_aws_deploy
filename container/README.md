# container

NeMo AutoModel を載せた SageMaker 用コンテナイメージの定義と、ローカル検証用のスクリプトです。

手順は `install_guide/01_container_build.md`（ビルドと push）と `install_guide/02_local_verification.md`（ローカル検証）を参照してください。

| ファイル | 役割 |
| --- | --- |
| `Dockerfile` | AWS DLC `pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` + `nemo-automodel==0.6.0` + `flash-linear-attention` + `causal-conv1d` + ビルド時検証 |
| `constraints.txt` | AutoModel v0.6.0 の `uv.lock` に合わせた pin |
| `build_and_push.sh` | DLC アカウントへのログイン → build → ECR リポジトリ作成 → push。`--no-push` で build のみ |
| `smoke_test.sh` | ビルド済みイメージの import / CLI / toolkit / `pip check` を確認 |
| `local_train_test.sh` | ローカル GPU でイメージ内から短い LoRA 学習を実行 |
| `local_sm_sim.sh` | SageMaker の `/opt/ml` 規約と `SM_*` 環境変数を再現して `train.py` を検証 |

```bash
export REGION=us-west-2
REGION=$REGION ./build_and_push.sh --no-push   # build
./smoke_test.sh                                # 検証
./local_train_test.sh                          # GPU がある場合
./local_sm_sim.sh                              # GPU がある場合
REGION=$REGION ./build_and_push.sh             # push
```
