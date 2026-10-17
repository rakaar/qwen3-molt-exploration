"""Generate on a CUDA worker only. Existing records are never silently overwritten."""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from screenlib import (append_jsonl, check_inputs, digest, expected_keys, experiment_id,
                       inputs, row_key, seed_for, validated_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, choices=(5, 10), default=5)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-seconds", type=int, default=10800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    check_inputs()
    protocol, models, questions = inputs()
    chosen = args.models or [m["id"] for m in models]
    if not chosen or set(chosen) - {m["id"] for m in models}:
        parser.error("Choose only model IDs listed in models.json")
    if args.samples == 10 and "baseline" not in chosen:
        parser.error("Include baseline when expanding a candidate to ten samples")
    if args.dry_run:
        print(json.dumps({"experiment_id": experiment_id(), "models": chosen,
                          "samples_per_question": args.samples,
                          "target_responses": len(expected_keys(chosen, args.samples))}, indent=2))
        return
    # Import only after dry-run, so the cloud controller needs no torch install.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("CUDA worker required. Model inference on the laptop/CPU is disabled.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This protocol requires a BF16-capable GPU; choose Ampere or newer.")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".generation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = args.output / "responses.jsonl"
        done = {row_key(row) for row in validated_rows(output)}
        baseline = next(m for m in models if m["role"] == "baseline")
        tokenizer = AutoTokenizer.from_pretrained(baseline["repo_id"], revision=baseline["revision"],
                                                 trust_remote_code=False)
        eos = tokenizer.eos_token_id
        tokenizer.pad_token_id = eos
        runtime = {"python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
                   "packages": {p: importlib.metadata.version(p) for p in
                                ("torch", "transformers", "huggingface-hub", "safetensors")}}
        (args.output / f"runtime-{time.time_ns()}.json").write_text(json.dumps(runtime, indent=2) + "\n")
        started = time.monotonic()
        for info in models:
            name = info["id"]
            if name not in chosen or expected_keys([name], args.samples) <= done:
                continue
            if time.monotonic() - started >= args.max_seconds:
                raise SystemExit("Generation time limit reached; saved responses can be resumed. GPU remains allocated.")
            print(f"Loading {name} at revision {info['revision']}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                info["repo_id"], revision=info["revision"], torch_dtype=torch.bfloat16,
                device_map="cuda:0", attn_implementation="sdpa", trust_remote_code=False,
            ).eval().requires_grad_(False)
            for qi, question in enumerate(questions):
                for si in range(args.samples):
                    key = name, question["id"], si
                    if key in done:
                        continue
                    if time.monotonic() - started >= args.max_seconds:
                        raise SystemExit("Generation time limit reached; results preserved. GPU remains allocated.")
                    messages = [{"role": "user", "content": question["question"]}]
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=protocol["generation"]["enable_thinking"])
                    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                    set_seed(seed_for(qi, si))
                    settings = {k: v for k, v in protocol["generation"].items() if k != "enable_thinking"}
                    tick = time.monotonic()
                    with torch.inference_mode():
                        generated = model.generate(**encoded, **settings, eos_token_id=eos,
                                                   pad_token_id=eos, use_cache=True)
                    tokens = generated[0, encoded.input_ids.shape[-1]:].tolist()
                    answer = tokenizer.decode(tokens, skip_special_tokens=True)
                    row = {"experiment_id": experiment_id(), "model": name,
                           "model_repo": info["repo_id"], "model_revision": info["revision"],
                           "question_id": question["id"], "question": question["question"],
                           "sample_index": si, "stage": 1 if si < 5 else 2,
                           "seed": seed_for(qi, si), "tokenizer_revision": baseline["revision"],
                           "rendered_prompt": rendered, "prompt_token_ids": encoded.input_ids[0].tolist(),
                           "prompt_sha256": digest(rendered), "answer": answer,
                           "generated_token_ids": tokens, "generated_tokens": len(tokens),
                           "truncated": bool(not tokens or tokens[-1] != eos),
                           "generation_seconds": time.monotonic() - tick,
                           "created_at": datetime.now(timezone.utc).isoformat()}
                    append_jsonl(output, row)
                    done.add(key)
                    print(json.dumps({k: row[k] for k in ("model", "question_id", "sample_index", "generated_tokens", "truncated")}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        target = expected_keys(chosen, args.samples)
        result = {"experiment_id": experiment_id(), "complete": target <= done,
                  "target_responses": len(target), "completed_responses": len(target & done),
                  "models": chosen, "samples_per_question": args.samples}
        (args.output / "generation-status.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
