# syntax=docker/dockerfile:1
# =============================================================================
# NeMo AutoModel on SageMaker — AWS Deep Learning Container ベース
#
# ベース: pytorch-training 2.10.0 (Python 3.13 / CUDA 13.0 / Ubuntu 22.04)
#   - sagemaker-training / sagemaker-pytorch-training (script mode, torch_distributed)
#   - EFA / aws-ofi-nccl / flash-attn 2.8.3 / TransformerEngine 2.11 を同梱
# 追加: nemo-automodel (PyPI) を v0.6.0 の lock に合わせて pin して導入
#
# 選定理由と比較は docs/02_dlc_selection.md を参照。
# =============================================================================
ARG REGION=us-west-2
ARG DLC_TAG=2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:${DLC_TAG}

ARG AUTOMODEL_VERSION=0.6.0
LABEL org.opencontainers.image.title="nemo-automodel-sagemaker" \
      org.opencontainers.image.description="NeMo AutoModel ${AUTOMODEL_VERSION} on AWS DLC pytorch-training" \
      nemo_automodel.version="${AUTOMODEL_VERSION}"

# --- AutoModel の導入 -------------------------------------------------------
COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt "nemo-automodel==${AUTOMODEL_VERSION}" \
 && rm -rf /root/.cache/pip /tmp/constraints.txt

# --- ビルド時検証: torch が差し替わっていないこと、import が通ること --------
RUN python - <<'PY'
import importlib, sys
import torch
assert torch.__version__.startswith("2.10.0"), f"torch が差し替わっています: {torch.__version__}"
import nemo_automodel, transformers, datasets, torchao
from nemo_automodel.cli.app import main  # CLI エントリポイント
print("torch          :", torch.__version__, "| cuda", torch.version.cuda)
print("nemo_automodel :", nemo_automodel.__version__)
print("transformers   :", transformers.__version__)
print("datasets       :", datasets.__version__)
print("torchao        :", torchao.__version__)
for mod in ("flash_attn", "transformer_engine", "sagemaker_training", "sagemaker_pytorch_container"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:28s}: OK", getattr(m, "__version__", ""))
    except Exception as e:  # GPU が無いビルド環境では TE 等が失敗してもよい
        print(f"{mod:28s}: import failed ({type(e).__name__}: {e})", file=sys.stderr)
PY
RUN pip check || (echo "pip check で依存衝突あり。上のログを確認してください" && exit 1)

# --- 実行時の既定値 --------------------------------------------------------
# HF のキャッシュを書き込み可能な場所に。ジョブ側の environment で上書き可。
ENV HF_HOME=/tmp/hf \
    TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled

# ENTRYPOINT / CMD / SAGEMAKER_TRAINING_MODULE は DLC のものをそのまま使う
