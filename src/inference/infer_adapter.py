#!/usr/bin/env python
"""LoRA アダプタを HF transformers + PEFT でロードし、生成を確認する検証スクリプト。

SageMaker Training Job (1 GPU、distribution なし) として実行する。toolkit が
`python infer_adapter.py --key value ...` の形で起動する。

入力チャネル:
  SM_CHANNEL_ADAPTER    学習ジョブの model.tar.gz (または展開済みディレクトリ)
  SM_CHANNEL_VALIDATION 生成に使うプロンプト (val.jsonl: prompt / output)
出力:
  SM_OUTPUT_DATA_DIR/adapter_inference.json  (output.tar.gz として S3 へ)
  標準出力の "[verify]" 行 (CloudWatch)

確認すること:
  1. アダプタの中身 (キーの内訳、mtp.* の LoRA 重みの有無)
  2. HF の Qwen3_5ForConditionalGeneration にアダプタを載せたとき、safetensors の各キーが
     モデルのどのパラメータに入ったか (一致 / 未使用)。PEFT は未使用キーを黙って捨てるため自前で照合する
  3. AutoModelForCausalLM (text-only クラス) に載せた場合の一致数 (モジュールパスが違うので 0 になる想定)
  4. ベースモデル単体とアダプタ付きの生成結果の違い
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path


def log(msg: str) -> None:
    print(f"[verify] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3.5-0.8B", help="ベースモデル (HF Hub id またはパス)")
    p.add_argument("--num_samples", type=int, default=5, help="生成するプロンプト数")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--also_causal_lm", type=int, default=1, help="1 なら AutoModelForCausalLM 経路でもキー一致を確認する")
    p.add_argument("--repetition_penalty", type=float, default=1.0, help="生成時の repetition_penalty (1.0 で無効)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 入力の解決
# ---------------------------------------------------------------------------
def resolve_adapter_dir(channel_dir: str) -> Path:
    """model.tar.gz なら展開し、adapter_config.json を含むディレクトリを返す。"""
    root = Path(channel_dir)
    tars = sorted(root.glob("*.tar.gz")) + sorted(root.glob("*.tgz"))
    if tars:
        dst = Path("/tmp/adapter_extracted")
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tars[0]) as t:
            t.extractall(dst)
        log(f"展開: {tars[0].name} → {dst}")
        root = dst
    cands = sorted(root.rglob("adapter_config.json"))
    if not cands:
        raise FileNotFoundError(f"adapter_config.json が {channel_dir} 配下に見つかりません")
    log(f"adapter_dir = {cands[0].parent}")
    return cands[0].parent


def load_prompts(channel_dir: str | None, n: int) -> list[dict]:
    if not channel_dir or not Path(channel_dir).exists():
        return [{"prompt": "「いちょう切り」とはどのような切り方ですか？", "output": ""}]
    files = sorted(Path(channel_dir).glob("*.jsonl")) + sorted(Path(channel_dir).glob("*.json"))
    rows: list[dict] = []
    for f in files:
        if f.suffix == ".jsonl":
            with open(f, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh if line.strip()]
        else:
            rows += json.loads(Path(f).read_text(encoding="utf-8"))
    return [{"prompt": r["prompt"], "output": r.get("output", "")} for r in rows[:n]]


# ---------------------------------------------------------------------------
# アダプタの中身
# ---------------------------------------------------------------------------
def inspect_adapter(adapter_dir: Path) -> dict:
    from safetensors import safe_open

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    keys: list[str] = []
    shapes: dict[str, list[int]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            keys.append(k)
            shapes[k] = list(f.get_slice(k).get_shape())
    groups = Counter()
    for k in keys:
        body = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
        parts = body.split(".")
        # 例: model.language_model.layers.3.self_attn.q_proj.lora_A.weight → language_model / self_attn.q_proj
        if parts[0] == "mtp":
            groups["mtp"] += 1
        elif "language_model" in parts:
            i = parts.index("layers") if "layers" in parts else -1
            mod = ".".join(parts[i + 2 : -2]) if i >= 0 else body
            groups[f"language_model:{mod}"] += 1
        else:
            groups[parts[0]] += 1
    mtp_keys = [k for k in keys if ".mtp." in k or k.startswith("mtp.") or ".model.mtp." in k]
    log(f"adapter_config: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} task_type={cfg.get('task_type')} "
        f"base={cfg.get('base_model_name_or_path')} target_modules={cfg.get('target_modules')}")
    log(f"adapter keys: {len(keys)} (mtp.* : {len(mtp_keys)})")
    for g, c in sorted(groups.items()):
        log(f"  {g}: {c}")
    return {"adapter_config": cfg, "num_keys": len(keys), "groups": dict(groups),
            "mtp_keys": mtp_keys, "sample_keys": keys[:5], "shapes_sample": {k: shapes[k] for k in keys[:5]}}


# ---------------------------------------------------------------------------
# モデルのロードと照合
# ---------------------------------------------------------------------------
def load_base(model_id: str, which: str):
    import torch
    import transformers

    kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if which == "conditional_generation":
        cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
    elif which == "causal_lm":
        cls = transformers.AutoModelForCausalLM
    else:
        raise ValueError(which)
    t0 = time.time()
    model = cls.from_pretrained(model_id, **kwargs).to("cuda").eval()
    log(f"loaded {type(model).__name__} via {which} in {time.time() - t0:.1f}s")
    return model


def attach_and_verify(base, adapter_dir: Path) -> tuple[object, dict]:
    """PeftModel を作り、safetensors の各キーが実際にモデルへ入ったかを照合する。"""
    import torch
    from peft import PeftModel
    from safetensors import safe_open

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    params = dict(peft_model.named_parameters())
    matched, unused, mismatched = [], [], []
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensor = f.get_tensor(k)
            # PEFT は ".lora_A.weight" を ".lora_A.<adapter名>.weight" に読み替える
            pk = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
            p = params.get(pk)
            if p is None:
                unused.append(k)
            elif torch.equal(p.detach().cpu().to(tensor.dtype), tensor):
                matched.append(k)
            else:
                mismatched.append(k)
    n_lora_params = sum(1 for n in params if ".lora_A." in n or ".lora_B." in n)
    rep = {"matched": len(matched), "unused": len(unused), "mismatched": len(mismatched),
           "lora_params_in_model": n_lora_params,
           "unused_keys": unused[:50], "mismatched_keys": mismatched[:20]}
    log(f"key check: matched={len(matched)} unused={len(unused)} mismatched={len(mismatched)} "
        f"(model has {n_lora_params} LoRA params)")
    if unused:
        log(f"  unused (先頭 10): {unused[:10]}")
    return peft_model, rep


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def render(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def generate(model, tokenizer, prompts: list[dict], max_new_tokens: int, repetition_penalty: float = 1.0) -> list[str]:
    import torch

    outs = []
    for r in prompts:
        text = render(tokenizer, r["prompt"])
        enc = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=repetition_penalty)
        outs.append(tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return outs


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    import torch
    import transformers
    import peft

    log(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} | "
        f"cuda {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    adapter_dir = resolve_adapter_dir(os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter"))
    prompts = load_prompts(os.environ.get("SM_CHANNEL_VALIDATION"), args.num_samples)
    out_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"model_id": args.model_id, "adapter": inspect_adapter(adapter_dir)}

    tok_src = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else args.model_id
    tokenizer = transformers.AutoTokenizer.from_pretrained(tok_src)
    results["tokenizer_from"] = tok_src
    results["rendered_prompt_0"] = render(tokenizer, prompts[0]["prompt"])
    log(f"rendered prompt 0:\n{results['rendered_prompt_0']}")

    # 1) ConditionalGeneration (モジュールパスが AutoModel と同じ) でベース → アダプタ
    base = load_base(args.model_id, "conditional_generation")
    results["base_class"] = type(base).__name__
    base_out = generate(base, tokenizer, prompts, args.max_new_tokens, args.repetition_penalty)
    peft_model, results["key_check_conditional_generation"] = attach_and_verify(base, adapter_dir)
    adapter_out = generate(peft_model, tokenizer, prompts, args.max_new_tokens, args.repetition_penalty)
    results["generations"] = [
        {"prompt": r["prompt"], "expected": r["output"], "base": b, "adapter": a, "changed": b != a}
        for r, b, a in zip(prompts, base_out, adapter_out)
    ]
    n_changed = sum(g["changed"] for g in results["generations"])
    log(f"generations: {n_changed}/{len(prompts)} prompts で出力がベースと異なる")
    for g in results["generations"]:
        log(f"--- prompt: {g['prompt']}\n  expected: {g['expected'][:120]}\n  base    : {g['base'][:160]}\n  adapter : {g['adapter'][:160]}")
    del peft_model, base
    torch.cuda.empty_cache()

    # 2) AutoModelForCausalLM (text-only クラス) に載せるとどうなるか
    if args.also_causal_lm:
        try:
            base2 = load_base(args.model_id, "causal_lm")
            results["causal_lm_class"] = type(base2).__name__
            _, results["key_check_causal_lm"] = attach_and_verify(base2, adapter_dir)
            del base2
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            results["key_check_causal_lm"] = {"error": f"{type(e).__name__}: {e}"}
            log(f"causal_lm path failed: {type(e).__name__}: {e}")

    (out_dir / "adapter_inference.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out_dir / 'adapter_inference.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
