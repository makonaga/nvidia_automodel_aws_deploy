# ベース DLC イメージの選定

AutoModel を載せるベースとなる AWS Deep Learning Container（DLC）の選定結果です。
2026-09-07 時点の公開情報（`aws/deep-learning-containers` リポジトリ、PyPI、`NVIDIA-NeMo/Automodel` の
`v0.6.0` タグ）に基づいています。

---

## 結論

**`763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker`** を採用します。

理由は 3 点です。

1. **torch 2.10.0 が AutoModel v0.6.0 のロックファイルと完全一致**する（他の DLC はすべて不一致）
2. **flash-attn 2.8.3 / TransformerEngine / EFA / aws-ofi-nccl / SageMaker training toolkit がすべて同梱**されている
3. 唯一の次点候補である 2.9 は **2026-10-15 にサポート終了**で、新規構築の土台にできない

唯一の懸念は Python が 3.13（AutoModel の開発環境は 3.12）である点ですが、
依存パッケージの cp313 wheel をすべて確認済みで、ソースビルドは発生しません（後述）。

---

## 1. AutoModel 側の要求（確認済み）

PyPI の `nemo-automodel` 最新は **0.6.0**（2026-08-26 公開）。同バージョンの GitHub タグ `v0.6.0` の
`uv.lock` と `.python-version` から、NVIDIA が検証した組み合わせは次の通りです。

| 項目 | AutoModel v0.6.0 |
|---|---|
| Python | **3.12** |
| torch | **2.10.0**（`pyproject` の制約は `>=2.6.0`、build group は `<=2.10.0`） |
| transformers | 5.12.1（`==` 固定） |
| torchao | 0.14.0 |
| quack-kernels | 0.6.1（`==` 固定、linux のみ） |
| megatron-fsdp | 0.5.0（`==` 固定） |
| torchdata | 0.11.0 |
| datasets | 4.2.0 |
| flashoptim | 0.1.3 |
| TransformerEngine | **オプション**（`[cuda]` extra で `>=2.14.1`） |
| flash-attn | **オプション**（`[fa]` extra で `<=2.8.3`） |

---

## 2. DLC 候補の比較

`aws/deep-learning-containers` の `docs/src/data/pytorch-training/*-gpu-sagemaker.yml` と
`pytorch/training/docker/<ver>/py3/cu130/Dockerfile.gpu` から抽出しました。

| | **2.10（採用）** | 2.9 | 2.8 | 2.7 |
|---|---|---|---|---|
| タグ | `2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` | `2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker` | `2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker` | `2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker` |
| Python | 3.13.11 | 3.12.10 | 3.12 | 3.12 |
| torch | **2.10.0** | 2.9.0 | 2.8.0 | 2.7.1 |
| CUDA | 13.0（base 13.0.2） | 13.0 | 12.9 | 12.8 |
| TransformerEngine | 2.11（ソースビルド済み） | 2.9 | — | — |
| flash-attn | **2.8.3** | 2.8.3 | — | — |
| torchdata | 0.11.0 | 0.11.0 | — | — |
| GA | 2026-01-21 | 2025-10-15 | 2025-08-06 | 2025-04-23 |
| サポート終了 | **2027-01-21** | **2026-10-15** | 2026-08-06（終了済） | 2026-04-23（終了済） |
| AutoModel torch 一致 | ✅ 一致 | ❌ 2.9 | ❌ 2.8 | ❌ 2.7 |
| AutoModel Python 一致 | ⚠️ 3.13 ≠ 3.12 | ✅ 3.12 | ✅ | ✅ |

2.8 以前はサポート終了済みのため除外。実質 2.10 と 2.9 の二択で、
**torch の一致と残りサポート期間の両方で 2.10 が優位**です。

---

## 3. DLC 2.10 に同梱されているもの

`Dockerfile.gpu` の `sagemaker` ステージまで含めた同梱物のうち、本件に関係するもの:

| 分類 | 内容 |
|---|---|
| 学習基盤 | torch 2.10.0 / torchvision 0.25.0 / torchaudio 2.10.0 / torchdata 0.11.0 / triton |
| 高速化カーネル | **flash-attn 2.8.3**（`MAX_JOBS=4` でビルド済み）、**TransformerEngine 2.11**（`release_v2.11` からビルド済み）、nvidia-mathdx、gdrcopy 2.5.1 |
| 通信 | EFA（`/opt/amazon/efa`）、**aws-ofi-nccl**（`/opt/amazon/ofi-nccl`）、Open MPI、`start_cuda_compat.sh` |
| SageMaker 連携 | **`sagemaker-training`**、**`sagemaker-pytorch-training`**（`SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main`）、`sagemaker>=3.0.0` |
| その他 | accelerate、numpy、pandas、scikit-learn、boto3、awscli、ipykernel |
| ENTRYPOINT | `bash -m start_with_right_hostname.sh` |

**transformers は同梱されていない**ため、AutoModel が要求する `transformers==5.12.1` と衝突しません。

`sagemaker-pytorch-training` が `distribution={"torch_distributed": {"enabled": True}}` を処理する
ランチャーの実体です。これが最初から入っているのが DLC を選ぶ最大の理由です。

---

## 4. Python 3.13 での wheel 互換性（検証済み）

AutoModel の依存チェーンで C 拡張を含むパッケージについて、PyPI の最新版に
`cp313` かつ `manylinux` の wheel が存在することを確認しました。

| パッケージ | 最新 | cp313 linux wheel |
|---|---|---|
| tokenizers | 0.23.2 | ✅ |
| safetensors | 0.8.0 | ✅ |
| sentencepiece | 0.2.2 | ✅ |
| tiktoken | 0.14.0 | ✅ |
| regex | 2026.9.3 | ✅ |
| numpy | 2.5.3 | ✅ |
| pyarrow | 25.0.1 | ✅ |
| apache-tvm-ffi（quack-kernels の依存） | 0.1.13 | ✅ |
| torch-c-dlpack-ext（quack-kernels の依存） | 0.1.5 | ✅ |

以下は **pure Python（`py3-none-any`）** のため Python バージョンに依存しません。

`nemo-automodel` 本体、`quack-kernels`、`nvidia-cutlass-dsl`、`torchao`（PyPI 版）、`flashoptim`、
`megatron-fsdp`、`torchdata`、`mistral-common`、`transformers`、`datasets`

→ **ソースビルドは 1 つも発生しません**。Python 3.13 は「NVIDIA が検証した組み合わせではない」という
意味での懸念に留まります。

---

## 5. 注意点と対処

### 5.1 TransformerEngine のバージョン差

DLC 同梱の TE は **2.11**、AutoModel の `[cuda]` extra が要求するのは **>=2.14.1** です。
ただし TE は AutoModel にとってオプションであり、`nemo_automodel/shared/te_patches.py` には
TE < 2.12 向けの互換パッチ（FusedAdam の `QuantizedTensor` 対応）が明示的に実装されています。
つまり古い TE が存在する環境は想定内です。

今回のスコープ（LoRA SFT）では TE の attention バックエンド（`model.backend.attn: te`）を使わず、
既定の SDPA または `flash_attention_2` を使うため、TE 2.11 のまま問題ありません。

万一 `import nemo_automodel` や学習起動時に TE 起因のエラーが出た場合の対処は次の 2 択です。

```dockerfile
# 対処 A: TE を外す（LoRA SFT なら影響なし）
RUN pip uninstall -y transformer-engine transformer-engine-torch transformer_engine

# 対処 B: TE を AutoModel の要求まで上げる（ソースビルド、30 分〜）
RUN pip install --no-build-isolation "transformer-engine[pytorch]>=2.14.1"
```

### 5.2 flash-attn は使える

DLC 同梱の flash-attn 2.8.3 は AutoModel の `[fa]` extra 制約 `<=2.8.3` と**完全一致**します。
`model.attn_implementation: flash_attention_2` を YAML で指定すれば SDPA より速くなります。

### 5.3 pip が torch を差し替えないようにする

`nemo-automodel` の依存に `torch>=2.6.0` があるため、通常は同梱の 2.10.0 がそのまま使われますが、
念のため Dockerfile の最後で検証します。

```dockerfile
RUN python -c "import torch, nemo_automodel; assert torch.__version__.startswith('2.10.0'), torch.__version__"
```

### 5.4 ロックファイルに合わせて pin する

NVIDIA の検証済み組み合わせに寄せるため、`uv.lock` 由来の constraints を使います。

```text
# container/constraints.txt（v0.6.0 の uv.lock より）
torch==2.10.0
transformers==5.12.1
torchao==0.14.0
quack-kernels==0.6.1
megatron-fsdp==0.5.0
flashoptim==0.1.3
datasets==4.2.0
torchdata==0.11.0
```

```dockerfile
RUN pip install --no-cache-dir -c /tmp/constraints.txt nemo-automodel==0.6.0
```

---

## 6. Dockerfile（案）

```dockerfile
ARG REGION=us-west-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker

COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt "nemo-automodel==0.6.0" \
 && python -c "import torch, nemo_automodel; assert torch.__version__.startswith('2.10.0'), torch.__version__" \
 && rm -rf /root/.cache/pip

# SageMaker script mode は DLC 側の ENTRYPOINT / SAGEMAKER_TRAINING_MODULE をそのまま使う
```

ビルドと push（既存 `llm-development` イメージと同じ手順）:

```bash
REGION=us-west-2
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# DLC の pull 認証（763104351884 は AWS 公式 DLC のアカウント）
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$REGION.amazonaws.com

docker build --build-arg REGION=$REGION -t nemo-automodel-sagemaker:0.6.0-pt2.10 container/

# 自アカウントの ECR へ push
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
aws ecr describe-repositories --repository-names nemo-automodel-sagemaker --region $REGION \
  || aws ecr create-repository --repository-name nemo-automodel-sagemaker --region $REGION
docker tag nemo-automodel-sagemaker:0.6.0-pt2.10 $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/nemo-automodel-sagemaker:0.6.0-pt2.10
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/nemo-automodel-sagemaker:0.6.0-pt2.10
```

イメージタグには AutoModel と torch のバージョンを含め、既存の `llm-development` とは別リポジトリで管理します
（transformers 5.x と 4.x は同居できないため）。

---

## 7. フォールバック

Python 3.13 起因の問題が出た場合のみ、**2.9（py312 / torch 2.9.0）** に切り替えます。
ただし 2026-10-15 でサポートが終了するため、あくまで短期の回避策です。
その場合は torch が AutoModel のロック（2.10.0）と食い違うため、`constraints.txt` の
`torch==2.10.0` を外して DLC 側の 2.9.0 を使います（`pyproject` の制約 `>=2.6.0` は満たします）。

---

## 8. Phase 2 での確認項目

1. `docker run --gpus all <image> python -c "import nemo_automodel, transformer_engine; print(nemo_automodel.__version__)"` が通る
2. `automodel --help` が通る（CLI エントリポイント）
3. `python -c "import torch; print(torch.__version__, torch.version.cuda)"` → `2.10.0`, `13.0`
4. `pip check` で依存衝突がない
5. 単一 GPU で `examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml` を数ステップ回す
   （`--step_scheduler.max_steps=5` 等で短縮）
