#!/usr/bin/env python
"""LoRA アダプタを本体と MTP ヘッドの両方にマージし、mtp.* を含む HF 形式のフルモデルを /opt/ml/model に書き出す。

ガイド 07 ステップ3 用。SageMaker Training Job (1 GPU、distribution なし) として実行する。
出力の model.tar.gz を vLLM DLC エンドポイントの model_data に渡し、
SM_VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}' で MTP を使った投機的デコーディングを有効にする。

処理:
  1. 本体: merge_adapter.py と同じく Qwen3_5ForConditionalGeneration + PeftModel.merge_and_unload() でマージして保存
     (SM_CHANNEL_MERGED にステップ2 の model.tar.gz を渡した場合はこれを省略し、その中身をそのまま使う)
  2. MTP ヘッド: ベースの safetensors から mtp.* を直接読み (transformers は mtp.* を読み飛ばすため)、
     アダプタの mtp.* の LoRA を W + scaling * B @ A で足し込む (scaling = lora_alpha / r。PEFT の merge と同じ式)
     AutoModel のキー名 (mtp.layers.0.eh_proj など) は HF のキー名 (mtp.fc など) に戻す
  3. 出力の safetensors に mtp.* を追加し、config.json に MTP 層数 (mtp_num_hidden_layers) を残す
  4. 検証: 出力の mtp.* の数と値、HF で読み直した本体の生成がアダプタ付きと一致すること

入力チャネル:
  SM_CHANNEL_ADAPTER     学習ジョブの model.tar.gz (必須)
  SM_CHANNEL_MERGED      ステップ2 の model.tar.gz (任意。あれば本体のマージを省略)
  SM_CHANNEL_VALIDATION  検証用プロンプト (任意)
出力:
  SM_MODEL_DIR/          HF 形式のフルモデル (mtp.* 入り)
  SM_OUTPUT_DATA_DIR/merge_mtp_info.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

from merge_adapter import check_adapter_keys, generate, load_prompts, resolve_adapter_dir

# AutoModel (nemo_automodel/components/models/qwen3_5/state_dict_adapter.py) の native -> HF キー対応
MTP_NATIVE_TO_HF = {
    "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
    "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
}
MTP_COUNT_KEYS = ("mtp_num_hidden_layers", "num_nextn_predict_layers")
ADAPTER_PREFIX = "base_model.model."


def log(msg: str) -> None:
    print(f"[merge-mtp] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--verify", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=96)
    return p.parse_args()


def extract_channel(channel_dir: str | None, dst: Path) -> Path | None:
    """チャネル内の tar.gz を dst に展開する。tar が無ければディレクトリをそのまま返す。"""
    if not channel_dir or not Path(channel_dir).exists():
        return None
    root = Path(channel_dir)
    tars = sorted(root.glob("*.tar.gz")) + sorted(root.glob("*.tgz"))
    if tars:
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tars[0]) as t:
            t.extractall(dst)
        return dst
    return root


def safetensor_files(model_dir: Path) -> list[Path]:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        names = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        return [model_dir / n for n in names]
    return sorted(model_dir.glob("*.safetensors"))


def load_mtp_tensors(model_dir: Path) -> dict:
    from safetensors import safe_open

    out = {}
    for st in safetensor_files(model_dir):
        with safe_open(str(st), framework="pt") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    out[k] = f.get_tensor(k)
    return out


def find_mtp_count(config: dict) -> tuple[str, str, int] | None:
    """config.json から MTP 層数を探す。(場所, キー, 値) を返す。"""
    for where, d in (("top", config), ("text_config", config.get("text_config") or {})):
        for k in MTP_COUNT_KEYS:
            if d.get(k):
                return where, k, int(d[k])
    return None


def merge_body(args, adapter_dir: Path, out: Path, prompts: list[str], info: dict) -> list[str]:
    """merge_adapter.py と同じ本体マージ。アダプタ付きの生成 (検証用) を返す。"""
    import torch
    import transformers
    from peft import PeftModel

    dtype = getattr(torch, args.dtype)
    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
    t0 = time.time()
    base = cls.from_pretrained(args.model_id, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    log(f"loaded {type(base).__name__} in {time.time() - t0:.1f}s")
    tok_src = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else args.model_id
    tokenizer = transformers.AutoTokenizer.from_pretrained(tok_src)

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    info["body_key_check"] = check_adapter_keys(peft_model, adapter_dir)
    adapter_out = generate(peft_model, tokenizer, prompts, args.max_new_tokens) if args.verify else []
    merged = peft_model.merge_and_unload()
    if [n for n, _ in merged.named_parameters() if "lora_" in n]:
        raise RuntimeError("マージ後に LoRA パラメータが残っています")
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    try:
        transformers.AutoProcessor.from_pretrained(args.model_id).save_pretrained(str(out))
    except Exception as e:  # noqa: BLE001
        log(f"processor の保存をスキップ: {type(e).__name__}: {e}")
    del peft_model, merged, base
    torch.cuda.empty_cache()
    log("body merged and saved")
    return adapter_out


def merge_mtp(adapter_dir: Path, base_mtp: dict, info: dict) -> dict:
    """アダプタの mtp.* LoRA をベースの mtp.* に足し込み、HF キー名で返す。"""
    import torch
    from safetensors import safe_open

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    if cfg.get("use_dora"):
        raise RuntimeError("DoRA のアダプタには対応していません")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise RuntimeError("rank_pattern / alpha_pattern 付きのアダプタには対応していません")
    scaling = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    info["lora"] = {"r": r, "lora_alpha": alpha, "use_rslora": bool(cfg.get("use_rslora")), "scaling": scaling}
    log(f"LoRA r={r} alpha={alpha} scaling={scaling}")

    lora: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if ".mtp." not in k:
                continue
            if not k.startswith(ADAPTER_PREFIX):
                raise RuntimeError(f"想定外のアダプタキー: {k}")
            body, ab = k[len(ADAPTER_PREFIX):].rsplit(".lora_", 1)  # body = mtp.layers.0.xxx, ab = 'A.weight'
            lora.setdefault(body, {})[ab[0]] = f.get_tensor(k)

    merged = dict(base_mtp)
    modules = []
    for native_mod, ab in sorted(lora.items()):
        if set(ab) != {"A", "B"}:
            raise RuntimeError(f"{native_mod}: lora_A / lora_B が揃っていません")
        native_key = f"{native_mod}.weight"
        hf_key = MTP_NATIVE_TO_HF.get(native_key, native_key)
        if hf_key not in base_mtp:
            raise RuntimeError(f"ベースに {hf_key} がありません (アダプタ側 {native_key})")
        w = base_mtp[hf_key]
        a, b = ab["A"], ab["B"]
        if b.shape[0] != w.shape[0] or a.shape[1] != w.shape[1] or a.shape[0] != r or b.shape[1] != r:
            raise RuntimeError(f"{hf_key}: 形状が合いません W{tuple(w.shape)} A{tuple(a.shape)} B{tuple(b.shape)}")
        delta = (b.float() @ a.float()) * scaling
        merged[hf_key] = (w.float() + delta).to(w.dtype)
        rel = (delta.norm() / w.float().norm()).item()
        modules.append({"hf_key": hf_key, "native_key": native_key, "shape": list(w.shape), "delta_rel_norm": rel})
        log(f"  {native_key} -> {hf_key} {tuple(w.shape)} |delta|/|W| = {rel:.4f}")
    if not modules:
        raise RuntimeError("アダプタに mtp.* の LoRA キーがありません")
    info["mtp_modules_merged"] = modules
    return merged


def write_mtp(out: Path, mtp: dict, info: dict) -> None:
    """出力ディレクトリの safetensors に mtp.* を追加する。"""
    from safetensors import safe_open
    from safetensors.torch import save_file

    idx_path = out / "model.safetensors.index.json"
    if idx_path.exists():
        shard = "model-mtp.safetensors"
        save_file(mtp, str(out / shard), metadata={"format": "pt"})
        idx = json.loads(idx_path.read_text())
        idx["weight_map"].update({k: shard for k in mtp})
        idx.setdefault("metadata", {})["total_size"] = idx.get("metadata", {}).get("total_size", 0) + sum(
            t.numel() * t.element_size() for t in mtp.values()
        )
        idx_path.write_text(json.dumps(idx, indent=2))
        info["output_layout"] = f"sharded (+{shard})"
    else:
        single = out / "model.safetensors"
        tensors = {}
        with safe_open(str(single), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        dup = sorted(set(tensors) & set(mtp))
        if dup:
            raise RuntimeError(f"出力に既に mtp.* があります: {dup[:3]}")
        tensors.update(mtp)
        tmp = out / "model.safetensors.tmp"
        save_file(tensors, str(tmp), metadata={"format": "pt"})
        tmp.replace(single)
        info["output_layout"] = "single model.safetensors (rewritten)"
    log(f"wrote {len(mtp)} mtp.* tensors ({info['output_layout']})")


def main() -> None:
    args = parse_args()
    import torch
    import transformers
    from huggingface_hub import snapshot_download

    log(f"torch {torch.__version__} | transformers {transformers.__version__} | model {args.model_id}")
    adapter_dir = resolve_adapter_dir(os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter"))
    out = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out.mkdir(parents=True, exist_ok=True)
    out_data = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    out_data.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(os.environ.get("SM_CHANNEL_VALIDATION"), args.num_samples)
    info: dict = {"model_id": args.model_id, "dtype": args.dtype}

    # 1. 本体
    merged_in = extract_channel(os.environ.get("SM_CHANNEL_MERGED"), Path("/tmp/merged_extracted"))
    adapter_out: list[str] = []
    if merged_in is not None:
        cfgs = sorted(merged_in.rglob("config.json"))
        if not cfgs:
            raise FileNotFoundError("SM_CHANNEL_MERGED に config.json がありません")
        for p in cfgs[0].parent.iterdir():
            if p.is_file() and not p.name.endswith(".sagemaker-uploaded"):
                shutil.copy2(p, out / p.name)
        info["body_source"] = "SM_CHANNEL_MERGED"
        log(f"body: ステップ2 のマージ済みモデルを使用 ({cfgs[0].parent})")
    else:
        info["body_source"] = "merged in this job"
        adapter_out = merge_body(args, adapter_dir, out, prompts, info)
    if load_mtp_tensors(out):
        raise RuntimeError("本体側の出力に既に mtp.* が含まれています")

    # 2. ベースの mtp.*
    base_dir = os.environ.get("SM_CHANNEL_BASE")
    if not base_dir or not Path(base_dir).exists():
        base_dir = snapshot_download(args.model_id, allow_patterns=["*.safetensors", "*.json"])
    base_dir = Path(base_dir)
    base_mtp = load_mtp_tensors(base_dir)
    if not base_mtp:
        raise RuntimeError(f"ベース {args.model_id} の safetensors に mtp.* がありません (MTP ヘッド無しのモデル)")
    base_cfg = json.loads((base_dir / "config.json").read_text())
    mtp_count = find_mtp_count(base_cfg)
    if mtp_count is None:
        raise RuntimeError(f"ベースの config.json に {MTP_COUNT_KEYS} がありません")
    info["base_mtp_keys"] = len(base_mtp)
    info["base_mtp_count_field"] = {"where": mtp_count[0], "key": mtp_count[1], "value": mtp_count[2]}
    log(f"base mtp.* keys: {len(base_mtp)} | {mtp_count[1]}={mtp_count[2]} ({mtp_count[0]})")

    # 3. マージして書き出し
    mtp = merge_mtp(adapter_dir, base_mtp, info)
    write_mtp(out, mtp, info)

    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    where, key, value = mtp_count
    target = cfg if where == "top" else cfg.setdefault("text_config", {})
    info["config_field_before"] = target.get(key)
    target[key] = value
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    log(f"config.json: {where}.{key} = {value} (before: {info['config_field_before']})")

    files = {p.name: p.stat().st_size for p in sorted(out.iterdir()) if p.is_file()}
    info["files"] = files
    log("saved: " + ", ".join(f"{k} ({v/1e6:.0f} MB)" if v > 1e6 else k for k, v in files.items()))

    # 4. 検証
    if args.verify:
        written = load_mtp_tensors(out)
        n_eq = sum(1 for k, t in mtp.items() if k in written and torch.equal(written[k], t))
        info["verify_mtp_keys"] = {"expected": len(mtp), "written": len(written), "equal": n_eq}
        log(f"verify mtp.*: written {len(written)} / expected {len(mtp)} / equal {n_eq}")
        if n_eq != len(mtp):
            raise RuntimeError("書き出した mtp.* が一致しません")

        dtype = getattr(torch, args.dtype)
        cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
        reloaded = cls.from_pretrained(str(out), dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
        tok2 = transformers.AutoTokenizer.from_pretrained(str(out))
        merged_out = generate(reloaded, tok2, prompts, args.max_new_tokens)
        if adapter_out:
            info["verify_body"] = [
                {"prompt": p, "adapter": a, "merged": m, "identical": a == m}
                for p, a, m in zip(prompts, adapter_out, merged_out)
            ]
            n_same = sum(v["identical"] for v in info["verify_body"])
            log(f"verify body: {n_same}/{len(prompts)} prompts でアダプタ付きと mtp 入りマージ済みの生成が一致")
        else:
            info["verify_body"] = [{"prompt": p, "merged": m} for p, m in zip(prompts, merged_out)]
            log("verify body: HF で読み直して生成できることを確認 (比較対象なし)")
        for v in info["verify_body"]:
            log(f"--- {v['prompt']}\n  merged: {v['merged'][:120]}")

    (out_data / "merge_mtp_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out_data / 'merge_mtp_info.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
