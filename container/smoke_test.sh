#!/usr/bin/env bash
# =============================================================================
# ビルドしたイメージの動作確認 (ローカル)。
#   ./smoke_test.sh                       # ローカルタグ nemo-automodel-sagemaker:<TAG>
#   ./smoke_test.sh <image_uri>           # 任意のイメージ
# GPU があれば --gpus all で CUDA の確認まで行う。
# =============================================================================
set -euo pipefail
AUTOMODEL_VERSION="${AUTOMODEL_VERSION:-0.6.0}"
IMAGE="${1:-nemo-automodel-sagemaker:${AUTOMODEL_VERSION}-pt2.10-py313-cu130}"

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
  GPU_ARGS=(--gpus all)
fi

run() { docker run --rm --platform linux/amd64 "${GPU_ARGS[@]}" --entrypoint "" "${IMAGE}" "$@"; }

echo "== [1] バージョンと import =="
run python -c '
import torch, nemo_automodel, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
print("nemo_automodel", nemo_automodel.__version__, "| transformers", transformers.__version__)
'

echo "== [2] automodel CLI =="
run automodel --help | head -n 5

echo "== [3] SageMaker toolkit のエントリポイント =="
run python -c "import sagemaker_training, sagemaker_pytorch_container; print('sagemaker-training OK')"
run bash -lc 'echo "SAGEMAKER_TRAINING_MODULE=$SAGEMAKER_TRAINING_MODULE"; which train'

echo "== [4] pip check =="
run pip check

if [[ ${#GPU_ARGS[@]} -gt 0 ]]; then
  echo "== [5] GPU: flash-attn / TransformerEngine の import =="
  run python -c '
import flash_attn; print("flash_attn", flash_attn.__version__)
import transformer_engine.pytorch as te; print("transformer_engine OK")
'
else
  echo "== [5] GPU なしのためスキップ (flash-attn / TE の確認は GPU 環境で) =="
fi

echo
echo "smoke test 完了: ${IMAGE}"
