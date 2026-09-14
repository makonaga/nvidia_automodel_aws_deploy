#!/usr/bin/env bash
# =============================================================================
# SageMaker Training Job のディレクトリ規約と環境変数を手元で再現し、
# イメージ内で train.py を torchrun 経由で実行する (toolkit が行うのと同じ起動形)。
# 検証対象: train.py のパス写像、上書き、学習、/opt/ml/model への成果物コピー。
#
#   ./local_sm_sim.sh                       # 10 ステップの短縮実行
#   ./local_sm_sim.sh --step_scheduler.max_steps 30 --optimizer.lr 5e-5   # 追加の上書き
#   NPROC=2 ./local_sm_sim.sh               # GPU が複数あれば
#   MODEL_TAR=/path/model.tar.gz ./local_sm_sim.sh   # model チャネル (tar.gz) の経路も検証
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUTOMODEL_VERSION="${AUTOMODEL_VERSION:-0.6.0}"
IMAGE="${IMAGE:-nemo-automodel-sagemaker:${AUTOMODEL_VERSION}-pt2.10-py313-cu130}"
CONFIG="${CONFIG:-qwen3_5_cooking_lora.yaml}"
NPROC="${NPROC:-1}"
OPT_ML="${REPO_ROOT}/out/sm_sim/opt_ml"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "GPU が見えません"; exit 1; }

# --- /opt/ml の再現 ---
rm -rf "${OPT_ML}"
mkdir -p "${OPT_ML}"/{input/data/train,input/data/validation,input/config,model,checkpoints,output/data,code}
cp "${REPO_ROOT}/data/cooking_basics/train.jsonl" "${OPT_ML}/input/data/train/"
cp "${REPO_ROOT}/data/cooking_basics/val.jsonl"   "${OPT_ML}/input/data/validation/"
cp -r "${REPO_ROOT}/src/."     "${OPT_ML}/code/"          # source_dir 相当
cp -r "${REPO_ROOT}/configs"   "${OPT_ML}/code/configs"   # dependencies=['../configs'] 相当
MODEL_ENV=()
if [[ -n "${MODEL_TAR:-}" ]]; then
  mkdir -p "${OPT_ML}/input/data/model" && cp "${MODEL_TAR}" "${OPT_ML}/input/data/model/"
  MODEL_ENV=(-e SM_CHANNEL_MODEL=/opt/ml/input/data/model)
fi
mkdir -p "${REPO_ROOT}/.hf_cache"

echo "== image  : ${IMAGE}"
echo "== config : ${CONFIG}"
echo "== /opt/ml: ${OPT_ML}"

docker run --rm --gpus all --platform linux/amd64 \
  --shm-size 8g \
  -v "${OPT_ML}:/opt/ml" \
  -v "${REPO_ROOT}/.hf_cache:/tmp/hf" \
  -w /opt/ml/code \
  -e HF_HOME=/tmp/hf \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e SM_MODEL_DIR=/opt/ml/model \
  -e SM_OUTPUT_DATA_DIR=/opt/ml/output/data \
  -e SM_CHANNEL_TRAIN=/opt/ml/input/data/train \
  -e SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation \
  -e SM_NUM_GPUS="${NPROC}" \
  -e SM_HOSTS='["algo-1"]' -e SM_CURRENT_HOST=algo-1 \
  "${MODEL_ENV[@]}" \
  --entrypoint "" \
  "${IMAGE}" \
  torchrun --nproc_per_node="${NPROC}" /opt/ml/code/train.py \
    --config "${CONFIG}" \
    --set "step_scheduler.max_steps=10,step_scheduler.val_every_steps=5,step_scheduler.ckpt_every_steps=5,step_scheduler.global_batch_size=8,step_scheduler.local_batch_size=2,dataset.seq_length=512,validation_dataset.seq_length=512,validation_dataset.limit_dataset_samples=16" \
    --warmup_epochs 0.5 \
    "$@" \
  2>&1 | tee "${REPO_ROOT}/out/sm_sim/train.log"

echo
echo "== /opt/ml/model (SageMaker が model.tar.gz にする内容) =="
find "${OPT_ML}/model" -maxdepth 2 | sort
echo "== /opt/ml/output/data =="
ls -la "${OPT_ML}/output/data"
echo "== /opt/ml/checkpoints =="
ls -la "${OPT_ML}/checkpoints"
