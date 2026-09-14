#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SageMaker Training Job ↔ NeMo AutoModel のアダプタ。

SageMaker の toolkit が torchrun 経由で本スクリプトを各 rank で起動する。
本スクリプトは学習ループを持たず、次のことだけを行う。

  1. ベース YAML を読み込む
  2. SageMaker の規約 (チャネルパス / 出力先 / GPU 数) を YAML の値に写像する
  3. ハイパーパラメータ (--set, --a.b.c value) を上書きとして適用する
  4. 実効 YAML をファイルに書き出し、nemo_automodel.cli.app.main() に委譲する
     (torchrun 配下なので InteractiveLauncher が in-process でレシピを実行する)
  5. 学習後、rank 0 が最良 (LOWEST_VAL) または最新 (LATEST) のチェックポイントを
     /opt/ml/model にコピーする (SageMaker が model.tar.gz として S3 に保存する)

ローカルでも動く: SM_* 環境変数が無ければ写像をスキップし、YAML の値をそのまま使う。

使い方 (SageMaker からは toolkit が組み立てる):
  torchrun --nproc_per_node=8 train.py --config qwen3_5_cooking_lora.yaml \
      --set step_scheduler.num_epochs=3 --optimizer.lr 1e-4 --warmup_epochs 1
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | sm_train | %(message)s")
log = logging.getLogger("sm_train")

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_SAGEMAKER = os.path.isdir("/opt/ml") and (os.environ.get("SM_MODEL_DIR") or os.environ.get("SM_CHANNEL_TRAIN"))

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
SM_CHECKPOINT_DIR = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
CODE_DIR = Path(os.environ.get("SM_MODULE_DIR", "/opt/ml/code"))
SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="SageMaker ↔ NeMo AutoModel adapter", allow_abbrev=False)
    p.add_argument("--config", required=True, help="ベース YAML のファイル名 (configs/sagemaker/ 配下) またはパス")
    p.add_argument("--set", action="append", default=[], help="'a.b=1,c.d=x' 形式の上書き (複数指定可)")
    p.add_argument("--model_id", default=None, help="HF Hub の model id またはローカルパス。model チャネルより優先")
    p.add_argument("--warmup_epochs", type=float, default=None, help="lr_scheduler.lr_warmup_steps をエポック数から算出")
    p.add_argument("--rslora_alpha", type=float, default=None, help="rsLoRA 相当の alpha。peft.alpha = alpha*sqrt(dim) に換算")
    p.add_argument("--final_checkpoint", choices=["LOWEST_VAL", "LATEST"], default="LOWEST_VAL",
                   help="/opt/ml/model にコピーするチェックポイント")
    p.add_argument("--checkpoint_dir", default=None, help="checkpoint.checkpoint_dir の明示指定 (既定: /opt/ml/checkpoints)")
    p.add_argument("--print_sample", type=int, default=1, help="学習前にレンダリング済みプロンプトを N 件表示 (0 で無効)")
    args, unknown = p.parse_known_args(argv)
    return args, unknown


# ---------------------------------------------------------------------------
# YAML ユーティリティ
# ---------------------------------------------------------------------------
def resolve_config_path(name: str) -> Path:
    cands = [Path(name)]
    for base in (CODE_DIR, SCRIPT_DIR, SCRIPT_DIR.parent):
        cands += [base / "configs" / "sagemaker" / name, base / "configs" / name, base / name]
    for c in cands:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"config が見つかりません: {name} (探索: {[str(c) for c in cands]})")


def parse_value(s: str):
    """CLI 文字列を YAML 相当の型に変換する ('1e-4' のような指数表記も float にする)。"""
    if s is None:
        return None
    t = s.strip()
    try:
        v = yaml.safe_load(t)
    except Exception:
        return t
    if isinstance(v, str):
        try:
            return float(v) if any(ch in v.lower() for ch in "e.") else int(v)
        except ValueError:
            return v
    return v


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def get_dotted(cfg: dict, dotted: str, default=None):
    d = cfg
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def unknown_args_to_overrides(unknown: list[str]) -> list[tuple[str, str]]:
    """['--a.b', '1', '--c.d=x', '--flag'] → [('a.b','1'), ('c.d','x'), ('flag','true')]"""
    out = []
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            log.warning("解釈できない引数を無視します: %r", tok)
            i += 1
            continue
        key = tok[2:]
        if "=" in key:
            k, v = key.split("=", 1)
            out.append((k, v))
            i += 1
        elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            out.append((key, unknown[i + 1]))
            i += 2
        else:
            out.append((key, "true"))
            i += 1
    return out


# ---------------------------------------------------------------------------
# SageMaker の規約 → YAML
# ---------------------------------------------------------------------------
def find_data_files(channel_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(channel_dir, "**", "*.jsonl"), recursive=True))
    files += sorted(glob.glob(os.path.join(channel_dir, "**", "*.json"), recursive=True))
    if not files:
        raise FileNotFoundError(f"チャネル {channel_dir} に .jsonl / .json がありません")
    return files


def count_samples(path: str) -> int:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else 0


def extract_tar_once(tar_path: str, dest: Path) -> Path:
    """同一ノードの複数 rank が同時に呼んでも 1 回だけ展開する (mkdir をロックに使う)。"""
    done = dest / ".extracted"
    lock = dest.parent / (dest.name + ".lock")
    if done.exists():
        return dest
    try:
        os.makedirs(lock)
    except FileExistsError:
        log.info("rank %d: 別プロセスの展開完了を待機 (%s)", RANK, tar_path)
        while not done.exists():
            time.sleep(2)
        return dest
    log.info("rank %d: %s を %s へ展開", RANK, tar_path, dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)
    done.touch()
    return dest


def resolve_model_dir(channel_dir: str) -> str:
    d = Path(channel_dir)
    if (d / "config.json").exists():
        return str(d)
    subdirs = [p for p in d.iterdir() if p.is_dir() and (p / "config.json").exists()]
    if len(subdirs) == 1:
        return str(subdirs[0])
    tars = sorted(d.glob("*.tar.gz")) + sorted(d.glob("*.tgz")) + sorted(d.glob("*.tar"))
    if len(tars) == 1:
        extracted = extract_tar_once(str(tars[0]), Path("/tmp/sm_model"))
        if (extracted / "config.json").exists():
            return str(extracted)
        inner = [p for p in extracted.rglob("config.json")]
        if inner:
            return str(inner[0].parent)
        raise FileNotFoundError(f"{tars[0]} の中に config.json がありません")
    raise FileNotFoundError(f"model チャネル {channel_dir} に config.json も tar.gz も見つかりません")


def apply_sagemaker_mapping(cfg: dict, args: argparse.Namespace) -> None:
    ch_train = os.environ.get("SM_CHANNEL_TRAIN")
    ch_val = os.environ.get("SM_CHANNEL_VALIDATION")
    ch_model = os.environ.get("SM_CHANNEL_MODEL")

    if ch_train:
        files = find_data_files(ch_train)
        set_dotted(cfg, "dataset.path_or_dataset_id", files[0] if len(files) == 1 else files)
        log.info("dataset ← %s", files)
    if "validation_dataset" in cfg:
        if ch_val:
            files = find_data_files(ch_val)
            set_dotted(cfg, "validation_dataset.path_or_dataset_id", files[0] if len(files) == 1 else files)
            log.info("validation_dataset ← %s", files)
        elif IS_SAGEMAKER:
            log.warning("validation チャネルが無いため validation_dataset を無効化します")
            cfg.pop("validation_dataset", None)
            cfg.pop("validation_dataloader", None)

    if args.model_id:
        set_dotted(cfg, "model.pretrained_model_name_or_path", args.model_id)
        log.info("model ← %s (--model_id)", args.model_id)
    elif ch_model:
        mdir = resolve_model_dir(ch_model)
        set_dotted(cfg, "model.pretrained_model_name_or_path", mdir)
        log.info("model ← %s (model チャネル)", mdir)

    ckpt_dir = args.checkpoint_dir or (SM_CHECKPOINT_DIR if IS_SAGEMAKER else None)
    if ckpt_dir:
        set_dotted(cfg, "checkpoint.checkpoint_dir", ckpt_dir)
        set_dotted(cfg, "checkpoint.enabled", True)
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info("checkpoint_dir ← %s", ckpt_dir)

    if IS_SAGEMAKER and get_dotted(cfg, "dist_env.timeout_minutes", 1) < 10:
        set_dotted(cfg, "dist_env.timeout_minutes", 10)


def apply_derived_settings(cfg: dict, args: argparse.Namespace) -> None:
    # --- グローバルバッチの整合性: global % (local * world_size) == 0 でないと落ちる ---
    gbs = int(get_dotted(cfg, "step_scheduler.global_batch_size", 1))
    lbs = int(get_dotted(cfg, "step_scheduler.local_batch_size", 1))
    per_step = lbs * WORLD_SIZE
    if gbs % per_step != 0:
        new_gbs = max(per_step, math.ceil(gbs / per_step) * per_step)
        log.warning("global_batch_size=%d は local_batch_size*world_size=%d で割り切れないため %d に調整", gbs, per_step, new_gbs)
        set_dotted(cfg, "step_scheduler.global_batch_size", new_gbs)
        gbs = new_gbs

    # --- rsLoRA 換算: scale = alpha_rs / sqrt(r) を AutoModel の alpha / dim で再現 ---
    if args.rslora_alpha is not None and "peft" in cfg:
        dim = int(get_dotted(cfg, "peft.dim", 8))
        alpha = int(round(args.rslora_alpha * math.sqrt(dim)))
        set_dotted(cfg, "peft.alpha", alpha)
        log.info("peft.alpha ← %d (rsLoRA alpha=%g, dim=%d → scale %.3f)", alpha, args.rslora_alpha, dim, alpha / dim)

    # --- warmup をエポック数で指定 ---
    if args.warmup_epochs is not None:
        train_path = get_dotted(cfg, "dataset.path_or_dataset_id")
        paths = train_path if isinstance(train_path, list) else [train_path]
        try:
            n = sum(count_samples(p) for p in paths)
            pack_size = int(get_dotted(cfg, "packed_sequence.packed_sequence_size", 0) or 0)
            if pack_size > 0:
                # packed では 1 サンプル = 1 pack。トークン数を一部サンプルから見積もって pack 数に換算する
                n = estimate_num_packs(cfg, paths, n, pack_size)
            steps_per_epoch = max(1, math.ceil(n / gbs))
            warmup = int(round(steps_per_epoch * args.warmup_epochs))
            cfg.setdefault("lr_scheduler", {})
            set_dotted(cfg, "lr_scheduler.lr_warmup_steps", warmup)
            log.info("lr_warmup_steps ← %d (samples=%d, steps/epoch=%d, warmup_epochs=%g)", warmup, n, steps_per_epoch, args.warmup_epochs)
        except Exception as e:  # HF Hub の dataset id など数えられない場合
            log.warning("warmup_epochs を steps に変換できませんでした: %s", e)


def estimate_num_packs(cfg: dict, paths: list[str], n_samples: int, pack_size: int, probe: int = 200) -> int:
    """packed sequence の pack 数を見積もる (先頭 probe 件をトークナイズして平均長を外挿)。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(get_dotted(cfg, "model.pretrained_model_name_or_path"), trust_remote_code=True)
    mapping = get_dotted(cfg, "dataset.column_mapping", {}) or {}
    q_col, a_col = mapping.get("question", "prompt"), mapping.get("answer", "output")
    seq_len = int(get_dotted(cfg, "dataset.seq_length", 0) or 0)
    lens: list[int] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()] if path.endswith(".jsonl") else json.load(f)
        for r in rows[: max(1, probe - len(lens))]:
            msgs = [{"role": "user", "content": r[q_col]}, {"role": "assistant", "content": r[a_col]}]
            L = len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
            lens.append(min(L, seq_len) if seq_len else L)
        if len(lens) >= probe:
            break
    avg = sum(lens) / max(1, len(lens))
    packs = math.ceil(n_samples * avg / pack_size * 1.1)  # 1.1 = bin packing の隙間ぶん
    log.info("packing 見積もり: samples=%d, avg_tokens=%.0f, pack_size=%d → packs≈%d", n_samples, avg, pack_size, packs)
    return max(1, packs)


# ---------------------------------------------------------------------------
# サンプルプロンプトの表示 (Composer 版の '=== Sample Prompt ===' 相当)
# ---------------------------------------------------------------------------
def print_sample_prompts(cfg: dict, n: int) -> None:
    if n <= 0 or RANK != 0:
        return
    try:
        from transformers import AutoTokenizer

        model_path = get_dotted(cfg, "model.pretrained_model_name_or_path")
        train_path = get_dotted(cfg, "dataset.path_or_dataset_id")
        train_path = train_path[0] if isinstance(train_path, list) else train_path
        mapping = get_dotted(cfg, "dataset.column_mapping", {}) or {}
        q_col, a_col, c_col = mapping.get("question", "prompt"), mapping.get("answer", "output"), mapping.get("context")
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        rows = []
        with open(train_path, encoding="utf-8") as f:
            if train_path.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
                    if len(rows) >= n:
                        break
            else:
                rows = json.load(f)[:n]
        for i, r in enumerate(rows):
            msgs = []
            if c_col and r.get(c_col):
                msgs.append({"role": "system", "content": r[c_col]})
            msgs += [{"role": "user", "content": r[q_col]}, {"role": "assistant", "content": r[a_col]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            print(f"=== Sample Prompt {i} (rendered with chat template) ===\n{text}\n=== End Sample ===", flush=True)
    except Exception as e:
        log.warning("サンプルプロンプトの表示に失敗 (学習は続行): %s", e)


# ---------------------------------------------------------------------------
# 学習後: 最終成果物を /opt/ml/model へ
# ---------------------------------------------------------------------------
def resolve_pointer(ckpt_dir: Path, name: str) -> Path | None:
    p = ckpt_dir / name
    if p.is_symlink() or p.is_dir():
        return p.resolve()
    if p.is_file():  # symlink が作れない FS ではテキストファイルにフォールバックする
        target = p.read_text(encoding="utf-8").strip()
        t = Path(target) if os.path.isabs(target) else ckpt_dir / target
        return t.resolve() if t.exists() else None
    return None


def pick_final_checkpoint(ckpt_dir: Path, preferred: str) -> Path | None:
    for name in ([preferred] + [n for n in ("LOWEST_VAL", "LATEST") if n != preferred]):
        p = resolve_pointer(ckpt_dir, name)
        if p and (p / "model").exists():
            log.info("最終チェックポイント: %s (%s)", p, name)
            return p
    cands = sorted([p for p in ckpt_dir.glob("epoch_*_step_*") if (p / "model").exists()], key=lambda p: p.stat().st_mtime)
    if cands:
        log.info("最終チェックポイント: %s (最新の epoch_*_step_*)", cands[-1])
        return cands[-1]
    return None


def export_final_model(cfg: dict, args: argparse.Namespace, effective_cfg_path: Path) -> None:
    if RANK != 0:
        return
    ckpt_dir = Path(get_dotted(cfg, "checkpoint.checkpoint_dir", ""))
    out_dir = Path(SM_MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = pick_final_checkpoint(ckpt_dir, args.final_checkpoint) if ckpt_dir.exists() else None
    if final is None:
        log.error("コピーできるチェックポイントが %s にありません", ckpt_dir)
        return
    shutil.copytree(final / "model", out_dir / "model", dirs_exist_ok=True)
    for extra in ("config.yaml",):
        if (final / extra).exists():
            shutil.copy2(final / extra, out_dir / extra)
    shutil.copy2(effective_cfg_path, out_dir / "effective_config.yaml")
    # tokenizer も同梱しておく (アダプタ単体でも推論側で困らないように)
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(get_dotted(cfg, "model.pretrained_model_name_or_path"), trust_remote_code=True).save_pretrained(out_dir / "tokenizer")
    except Exception as e:
        log.warning("tokenizer の保存に失敗 (続行): %s", e)
    info = {
        "checkpoint": str(final),
        "selected_by": args.final_checkpoint,
        "base_model": get_dotted(cfg, "model.pretrained_model_name_or_path"),
        "is_peft": "peft" in cfg,
        "world_size": WORLD_SIZE,
    }
    for logname in ("training.jsonl", "validation.jsonl"):
        src = ckpt_dir / logname
        if src.exists():
            shutil.copy2(src, out_dir / logname)
    (out_dir / "training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("成果物を %s へコピーしました: %s", out_dir, sorted(p.name for p in out_dir.iterdir()))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args, unknown = parse_args(sys.argv[1:] if argv is None else argv)
    cfg_path = resolve_config_path(args.config)
    log.info("rank %d/%d | config=%s | sagemaker=%s", RANK, WORLD_SIZE, cfg_path, bool(IS_SAGEMAKER))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    apply_sagemaker_mapping(cfg, args)

    overrides: list[tuple[str, str]] = []
    for s in args.set:
        for kv in filter(None, (x.strip() for x in s.split(","))):
            if "=" not in kv:
                raise ValueError(f"--set の書式が不正です: {kv!r} (a.b=c の形式)")
            k, v = kv.split("=", 1)
            overrides.append((k.strip(), v.strip()))
    overrides += unknown_args_to_overrides(unknown)
    for k, v in overrides:
        set_dotted(cfg, k, parse_value(v))
        log.info("override %s = %r", k, parse_value(v))

    apply_derived_settings(cfg, args)

    # 実効 YAML を書き出す (各 rank は自分のファイルを使い、rank 0 は output にも残す)
    tmp_dir = Path(os.environ.get("SM_TMP_DIR", "/tmp")) / "sm_train"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    effective = tmp_dir / f"effective_config.rank{RANK}.yaml"
    effective.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if RANK == 0:
        out = Path(SM_OUTPUT_DATA_DIR)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(effective, out / "effective_config.yaml")
        print("=== effective config ===\n" + effective.read_text(encoding="utf-8") + "=== end ===", flush=True)

    print_sample_prompts(cfg, args.print_sample)

    # AutoModel CLI へ委譲。torchrun 配下なので InteractiveLauncher が in-process 実行する。
    from nemo_automodel.cli.app import main as automodel_main

    sys.argv = ["automodel", str(effective)]
    rc = 0
    try:
        ret = automodel_main()
        rc = int(ret) if isinstance(ret, int) else 0
    except SystemExit as e:
        rc = int(e.code or 0)
    if rc != 0:
        log.error("学習が異常終了しました (rc=%d)", rc)
        return rc

    export_final_model(cfg, args, effective)
    return 0


if __name__ == "__main__":
    sys.exit(main())
