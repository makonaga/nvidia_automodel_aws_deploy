#!/usr/bin/env python
"""LoRA アダプタをベースモデルにマージし、HF 形式のフルモデルとして /opt/ml/model に書き出す。

SageMaker Training Job (1 GPU、distribution なし) として実行する。出力の model.tar.gz を
そのまま vLLM DLC エンドポイントの model_data に渡せる。

入力チャネル:
  SM_CHANNEL_ADAPTER    学習ジョブの model.tar.gz (または展開済みディレクトリ)
  SM_CHANNEL_VALIDATION 検証用プロンプト (val.jsonl)。--verify 1 のときに使う
出力:
  SM_MODEL_DIR/          config.json, model*.safetensors, tokenizer, processor 設定, chat_template.jinja
  SM_OUTPUT_DATA_DIR/merge_info.json

注意:
  - HF の Qwen3_5ForConditionalGeneration でマージするため、mtp.* (MTP ヘッド) は出力に含まれない。
    MTP を使う配信にはこのスクリプトではなく、MTP ヘッドも含めてマージするツールが必要 (ガイド 07 ステップ3)。
  - アダプタの mtp.* の LoRA 重みは未使用になる (ステップ1 で確認済み)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[merge] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--verify", type=int, default=1, help="1 ならマージ後のモデルを読み直し、アダプタ付きモデルと生成を比較する")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=96)
    return p.parse_args()


def resolve_adapter_dir(channel_dir: str) -> Path:
    root = Path(channel_dir)
    tars = sorted(root.glob("*.tar.gz")) + sorted(root.glob("*.tgz"))
    if tars:
        dst = Path("/tmp/adapter_extracted")
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tars[0]) as t:
            t.extractall(dst)
        root = dst
    cands = sorted(root.rglob("adapter_config.json"))
    if not cands:
        raise FileNotFoundError(f"adapter_config.json が {channel_dir} 配下に見つかりません")
    log(f"adapter_dir = {cands[0].parent}")
    return cands[0].parent


def load_prompts(channel_dir: str | None, n: int) -> list[str]:
    if not channel_dir or not Path(channel_dir).exists():
        return ["「いちょう切り」とはどのような切り方ですか？"]
    rows: list[dict] = []
    for f in sorted(Path(channel_dir).glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    for f in sorted(Path(channel_dir).glob("*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    return [r["prompt"] for r in rows[:n]]


def render(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    outs = []
    for p in prompts:
        enc = tokenizer(render(tokenizer, p), return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        outs.append(tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return outs


def check_adapter_keys(peft_model, adapter_dir: Path) -> dict:
    import torch
    from safetensors import safe_open

    params = dict(peft_model.named_parameters())
    matched, unused = 0, []
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            pk = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
            p = params.get(pk)
            if p is not None and torch.equal(p.detach().cpu().to(f.get_tensor(k).dtype), f.get_tensor(k)):
                matched += 1
            else:
                unused.append(k)
    non_mtp_unused = [k for k in unused if ".mtp." not in k]
    log(f"adapter key check: matched={matched} unused={len(unused)} (mtp 以外の未使用 {len(non_mtp_unused)})")
    if non_mtp_unused:
        raise RuntimeError(f"mtp 以外に未使用の LoRA キーがあります: {non_mtp_unused[:5]}")
    return {"matched": matched, "unused": len(unused), "unused_keys": unused}


def main() -> None:
    args = parse_args()
    import torch
    import transformers
    from peft import PeftModel

    dtype = getattr(torch, args.dtype)
    log(f"torch {torch.__version__} | transformers {transformers.__version__} | model {args.model_id} | dtype {args.dtype}")
    adapter_dir = resolve_adapter_dir(os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter"))
    out = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out.mkdir(parents=True, exist_ok=True)
    out_data = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    out_data.mkdir(parents=True, exist_ok=True)
    info: dict = {"model_id": args.model_id, "dtype": args.dtype, "adapter_config": json.loads((adapter_dir / "adapter_config.json").read_text())}

    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
    t0 = time.time()
    base = cls.from_pretrained(args.model_id, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    log(f"loaded {type(base).__name__} in {time.time() - t0:.1f}s")
    info["base_class"] = type(base).__name__

    tok_src = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else args.model_id
    tokenizer = transformers.AutoTokenizer.from_pretrained(tok_src)
    prompts = load_prompts(os.environ.get("SM_CHANNEL_VALIDATION"), args.num_samples)

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    info["key_check"] = check_adapter_keys(peft_model, adapter_dir)
    adapter_out = generate(peft_model, tokenizer, prompts, args.max_new_tokens) if args.verify else []

    t0 = time.time()
    merged = peft_model.merge_and_unload()
    log(f"merge_and_unload in {time.time() - t0:.1f}s → {type(merged).__name__}")
    has_lora = [n for n, _ in merged.named_parameters() if "lora_" in n]
    if has_lora:
        raise RuntimeError(f"マージ後に LoRA パラメータが残っています: {has_lora[:3]}")

    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    try:
        transformers.AutoProcessor.from_pretrained(args.model_id).save_pretrained(str(out))
        info["processor_saved"] = True
    except Exception as e:  # noqa: BLE001
        log(f"processor の保存をスキップ: {type(e).__name__}: {e}")
        info["processor_saved"] = False
    files = {p.name: p.stat().st_size for p in sorted(out.iterdir()) if p.is_file()}
    info["files"] = files
    log("saved: " + ", ".join(f"{k} ({v/1e6:.0f} MB)" if v > 1e6 else k for k, v in files.items()))

    # 出力に mtp.* が無いことを記録 (HF が読み飛ばすため無い想定)
    from safetensors import safe_open
    mtp_in_output = []
    for st in sorted(out.glob("*.safetensors")):
        with safe_open(str(st), framework="pt") as f:
            mtp_in_output += [k for k in f.keys() if k.startswith("mtp.")]
    info["mtp_keys_in_output"] = len(mtp_in_output)
    log(f"mtp.* keys in output: {len(mtp_in_output)}")

    del peft_model, merged, base
    torch.cuda.empty_cache()

    if args.verify:
        reloaded = cls.from_pretrained(str(out), dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
        tok2 = transformers.AutoTokenizer.from_pretrained(str(out))
        merged_out = generate(reloaded, tok2, prompts, args.max_new_tokens)
        info["verify"] = [
            {"prompt": p, "adapter": a, "merged": m, "identical": a == m}
            for p, a, m in zip(prompts, adapter_out, merged_out)
        ]
        n_same = sum(v["identical"] for v in info["verify"])
        log(f"verify: {n_same}/{len(prompts)} prompts でアダプタ付きとマージ済みの生成が一致")
        for v in info["verify"]:
            log(f"--- {v['prompt']}\n  adapter: {v['adapter'][:120]}\n  merged : {v['merged'][:120]}")

    (out_data / "merge_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out_data / 'merge_info.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
