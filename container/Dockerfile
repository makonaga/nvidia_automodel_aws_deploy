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

# --- ベース DLC の時点での pip check 結果を記録 (既存の衝突は AWS 側の事情) ----
RUN pip check > /tmp/pipcheck_base.txt 2>&1 || true

# --- AutoModel の導入 -------------------------------------------------------
# 注意: datasets 4.2.0 (AutoModel の lock) は fsspec<=2025.9.0 を要求するため、
# DLC 同梱の s3fs 2026.7.0 (fsspec 2026.7.0 が必要) との整合が取れなくなる。
# s3fs を使うのはクライアント側 SDK の sagemaker-mlops だけで、学習ジョブ
# (sagemaker-training / sagemaker-pytorch-training → train.py) では import されない。
# s3fs を消すと sagemaker (meta) → sagemaker-mlops → s3fs の連鎖で SDK ごと
# 削除することになるため、DLC のパッケージはそのまま残し、この衝突だけを
# 下の pip check で「許容済み」として扱う。
COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt "nemo-automodel==${AUTOMODEL_VERSION}" \
 && rm -rf /root/.cache/pip /tmp/constraints.txt

# --- ビルド時検証 1: torch が差し替わっていないこと、import が通ること -------
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

# --- ビルド時検証 2: pip check ---------------------------------------------
# ベース DLC に元からあった衝突は [info] として表示するだけにし、
# この層で新たに生じた衝突があるときだけビルドを失敗させる。
RUN python - <<'PY'
import pathlib, subprocess, sys
# 許容する衝突 (行に含まれるキーワードの組で判定)。理由は各行のコメント参照。
ACCEPTED = [
    # datasets 4.2.0 が fsspec<=2025.9.0 を要求するため s3fs 2026.7.0 と整合しない。
    # s3fs は sagemaker-mlops (クライアント SDK) 専用で、学習ジョブでは使われない。
    ("s3fs", "fsspec"),
]
def accepted(line):
    return any(all(k in line for k in keys) for keys in ACCEPTED)

base = {l for l in pathlib.Path("/tmp/pipcheck_base.txt").read_text().splitlines() if l.strip()}
res = subprocess.run(["pip", "check"], capture_output=True, text=True)
now = [l for l in (res.stdout + res.stderr).splitlines() if l.strip()]
ok_msg = "No broken requirements found."
fresh = [l for l in now if l not in base and l != ok_msg]
new = [l for l in fresh if not accepted(l)]
tolerated = [l for l in fresh if accepted(l)]
pre = sorted(base - {ok_msg})
if pre:
    print("[info] ベース DLC 由来の既存衝突 (無視):")
    for l in pre:
        print("   ", l)
if tolerated:
    print("[warn] この層で生じたが許容済みの衝突 (学習経路では未使用):")
    for l in tolerated:
        print("   ", l)
if new:
    print("[error] この層で新たに生じた依存衝突:", file=sys.stderr)
    for l in new:
        print("   ", l, file=sys.stderr)
    sys.exit(1)
print("[ok] pip check: 新規の依存衝突なし")
PY

# --- 実行時の既定値 --------------------------------------------------------
# HF のキャッシュを書き込み可能な場所に。ジョブ側の environment で上書き可。
ENV HF_HOME=/tmp/hf \
    TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled

# ENTRYPOINT / CMD / SAGEMAKER_TRAINING_MODULE は DLC のものをそのまま使う
