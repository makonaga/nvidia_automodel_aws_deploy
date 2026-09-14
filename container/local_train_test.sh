#!/usr/bin/env bash
# =============================================================================
# ローカル GPU でイメージ内から小さな LoRA 学習を回し、AutoModel の学習経路を検証する。
# SageMaker と同じ「torchrun → nemo_automodel.cli.app → in-process 実行」の経路を使う。
#
#   ./local_train_test.sh                       # 既定: Qwen/Qwen3.5-0.8B
#   MODEL_ID=Qwen/Qwen3-0.6B ./local_train_test.sh   # モデルを差し替え
#   HF_TOKEN=hf_xxx ./local_train_test.sh       # gated モデルの場合
#
# 出力: <repo>/out/local_test/ (チェックポイント・ログ)。HF のキャッシュは <repo>/.hf_cache。
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUTOMODEL_VERSION="${AUTOMODEL_VERSION:-0.6.0}"
IMAGE="${IMAGE:-nemo-automodel-sagemaker:${AUTOMODEL_VERSION}-pt2.10-py313-cu130}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"
CONFIG="configs/local/qwen3_5_cooking_lora_local.yaml"
NPROC="${NPROC:-1}"
EXTRA_ARGS=("$@")   # 追加の --key.subkey=value 上書き

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "GPU が見えません"; exit 1; }
mkdir -p "${REPO_ROOT}/out/local_test" "${REPO_ROOT}/.hf_cache"

echo "== image  : ${IMAGE}"
echo "== model  : ${MODEL_ID}"
echo "== config : ${CONFIG}"
echo "== out    : ${REPO_ROOT}/out/local_test"

docker run --rm --gpus all --platform linux/amd64 \
  --shm-size 8g \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  -e HF_HOME=/workspace/.hf_cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e WANDB_MODE=disabled \
  -e TOKENIZERS_PARALLELISM=false \
  --entrypoint "" \
  "${IMAGE}" \
  torchrun --nproc_per_node="${NPROC}" -m nemo_automodel.cli.app "${CONFIG}" \
    --model.pretrained_model_name_or_path="${MODEL_ID}" \
    "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${REPO_ROOT}/out/local_test/train.log"

echo
echo "== 出力 =="
ls -la "${REPO_ROOT}/out/local_test/checkpoints" 2>/dev/null || echo "(チェックポイント無し)"
echo "== 学習ログの step 行 =="
grep -E "step [0-9]+ \| epoch|\[val\]" "${REPO_ROOT}/out/local_test/train.log" | tail -n 8
