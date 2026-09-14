#!/usr/bin/env bash
# =============================================================================
# ビルドしたイメージの動作確認 (ローカル)。
#   ./smoke_test.sh                       # ローカルタグ nemo-automodel-sagemaker:<TAG>
#   ./smoke_test.sh <image_uri>           # 任意のイメージ
# GPU があれば --gpus all で CUDA の確認まで行う (SMOKE_GPU=0/1 で強制指定可)。
# =============================================================================
set -euo pipefail
AUTOMODEL_VERSION="${AUTOMODEL_VERSION:-0.6.0}"
IMAGE="${1:-nemo-automodel-sagemaker:${AUTOMODEL_VERSION}-pt2.10-py313-cu130}"

# GPU 判定: nvidia-smi が「実際に成功する」ことを条件にする。
# コマンドや Container Toolkit の有無だけで判定すると、ドライバ未ロードの PC で
# --gpus all が "nvml error: driver not loaded" で失敗する。
# SMOKE_GPU=0 で強制的に CPU モード、SMOKE_GPU=1 で強制的に GPU モード。
GPU_ARGS=()
if [[ "${SMOKE_GPU:-auto}" == "1" ]] || { [[ "${SMOKE_GPU:-auto}" == "auto" ]] && nvidia-smi >/dev/null 2>&1; }; then
  GPU_ARGS=(--gpus all)
  echo "GPU モードで実行します (--gpus all)"
else
  echo "CPU モードで実行します (GPU 未検出。flash-attn / TE の確認はスキップ)"
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

echo "== [4] pip check (想定内: s3fs/fsspec, aiobotocore/botocore の 2 件) =="
# pip check は衝突があると終了コード 1 を返すので set -e で止めない。
# 想定内の 2 件以外が出た場合だけ失敗にする。
pipcheck_out="$(run pip check 2>&1 || true)"
echo "${pipcheck_out}"
unexpected="$(echo "${pipcheck_out}" | grep -v -E '^(s3fs .* fsspec|aiobotocore .* botocore|No broken requirements)' || true)"
if [[ -n "${unexpected}" ]]; then
  echo "[error] 想定外の依存衝突:" >&2
  echo "${unexpected}" >&2
  exit 1
fi
echo "[ok] 想定外の衝突なし"

if [[ ${#GPU_ARGS[@]} -gt 0 ]]; then
  echo "== [5] GPU: flash-attn / TransformerEngine の import =="
  run python -c '
import flash_attn; print("flash_attn", flash_attn.__version__)
import transformer_engine.pytorch as te; print("transformer_engine OK")
import fla; print("fla", fla.__version__)
try:
    import causal_conv1d; print("causal_conv1d", causal_conv1d.__version__)
except ImportError as e:
    print("causal_conv1d: 未導入 (packed + MTP を使うなら INSTALL_CAUSAL_CONV1D=1 で再ビルド)", e)
'
else
  echo "== [5] GPU なしのためスキップ (flash-attn / TE の確認は GPU 環境で) =="
fi

echo
echo "smoke test 完了: ${IMAGE}"
