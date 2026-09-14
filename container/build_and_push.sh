#!/usr/bin/env bash
# =============================================================================
# DLC を pull → AutoModel を載せて build → 自アカウントの ECR へ push
#
# 使い方:
#   cd container
#   REGION=us-west-2 ./build_and_push.sh            # build + push
#   REGION=us-west-2 ./build_and_push.sh --no-push  # build のみ
#
# 環境変数で上書き可能:
#   REGION            AWS リージョン (既定 us-west-2)
#   REPO              ECR リポジトリ名 (既定 nemo-automodel-sagemaker)
#   AUTOMODEL_VERSION nemo-automodel のバージョン (既定 0.6.0)
#   DLC_TAG           ベース DLC のタグ (既定 2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker)
#   TAG               生成イメージのタグ (既定 <AUTOMODEL_VERSION>-pt2.10-py313-cu130)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

REGION="${REGION:-us-west-2}"
REPO="${REPO:-nemo-automodel-sagemaker}"
AUTOMODEL_VERSION="${AUTOMODEL_VERSION:-0.6.0}"
DLC_TAG="${DLC_TAG:-2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker}"
TAG="${TAG:-${AUTOMODEL_VERSION}-pt2.10-py313-cu130}"
DLC_ACCOUNT=763104351884
PUSH=1
[[ "${1:-}" == "--no-push" ]] && PUSH=0

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_LOCAL="${REPO}:${TAG}"
IMAGE_REMOTE="${ECR}/${REPO}:${TAG}"

echo "== 設定 =="
echo "  REGION            : ${REGION}"
echo "  ACCOUNT           : ${ACCOUNT}"
echo "  base DLC          : ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:${DLC_TAG}"
echo "  nemo-automodel    : ${AUTOMODEL_VERSION}"
echo "  image             : ${IMAGE_REMOTE}"
echo

echo "== 1/4 DLC アカウントへ ECR ログイン (ベースイメージの pull 用) =="
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo "== 2/4 docker build (linux/amd64) =="
docker build \
  --platform linux/amd64 \
  --build-arg REGION="${REGION}" \
  --build-arg DLC_TAG="${DLC_TAG}" \
  --build-arg AUTOMODEL_VERSION="${AUTOMODEL_VERSION}" \
  -t "${IMAGE_LOCAL}" \
  .

docker image ls "${REPO}" --format 'built: {{.Repository}}:{{.Tag}}  size={{.Size}}'

if [[ "${PUSH}" -eq 0 ]]; then
  echo "== --no-push 指定のため push はスキップ =="
  exit 0
fi

echo "== 3/4 自アカウントへ ECR ログインとリポジトリ作成 =="
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR}"
aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository \
       --repository-name "${REPO}" \
       --region "${REGION}" \
       --image-scanning-configuration scanOnPush=true \
       --query 'repository.repositoryUri' --output text

echo "== 4/4 docker push =="
docker tag "${IMAGE_LOCAL}" "${IMAGE_REMOTE}"
docker push "${IMAGE_REMOTE}"

echo
echo "== 完了 =="
echo "image_uri = ${IMAGE_REMOTE}"
aws ecr describe-images --repository-name "${REPO}" --region "${REGION}" \
  --image-ids imageTag="${TAG}" \
  --query 'imageDetails[0].{tag:imageTags[0],sizeMB:imageSizeInBytes,pushedAt:imagePushedAt}' --output table
