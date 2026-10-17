"""Optional original-paper API grader; run on the controller, never needs a GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from screenlib import (ROOT, aggregate_logprobs, append_jsonl, digest, experiment_id,
                       inputs, read_json, read_jsonl, row_key, validated_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-calls", type=int, required=True,
                        help="Explicit maximum number of paid requests in this invocation; no automatic retries")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_new_calls < 0:
        parser.error("max-new-calls must be nonnegative")
    responses = validated_rows(args.output / "responses.jsonl")
    if not responses:
        raise SystemExit("No responses to judge")
    judge = inputs()[0]["judge"]
    prompts = read_json(ROOT / "judge-prompts.json")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".judge.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / "scores.jsonl"
        done = set()
        answers = {row_key(r): digest(r["answer"]) for r in responses}
        for row in read_jsonl(path):
            key = row_key(row)
            if (row.get("experiment_id") != experiment_id() or row.get("judge_model") != judge["model"]
                    or row.get("answer_sha256") != answers.get(key)
                    or row.get("prompt_sha256") != digest(prompts[row["metric"]])):
                raise SystemExit("Existing scores do not match these answers, judge, or prompt")
            score_key = key + (row["metric"],)
            if score_key in done:
                raise SystemExit("Duplicate judge result")
            done.add(score_key)
        pending = [(row, metric) for row in responses for metric in prompts
                   if row_key(row) + (metric,) not in done]
        if args.dry_run:
            print(json.dumps({"pending_calls": len(pending), "call_limit": args.max_new_calls,
                              "judge": judge["model"], "api_key_present": bool(os.getenv("OPENAI_API_KEY"))}, indent=2))
            return
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set OPENAI_API_KEY through the cloud secret UI; never paste it in chat or commit it")
        for row, metric in pending[:args.max_new_calls]:
            payload = {"model": judge["model"], "messages": [{"role": "user", "content":
                       prompts[metric].format(question=row["question"], answer=row["answer"])}],
                       "max_completion_tokens": 1, "temperature": 0, "logprobs": True,
                       "top_logprobs": 20, "seed": 0, "store": False}
            request = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key,
                "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                raise SystemExit(f"Judge request failed: HTTP {exc.code}; results preserved, no model substitution or retry") from None
            except (urllib.error.URLError, TimeoutError):
                raise SystemExit("Judge network request failed; check account usage before resuming") from None
            choice = result["choices"][0]
            content = (choice.get("logprobs") or {}).get("content") or []
            if not content:
                raise SystemExit("Judge returned no token logprobs; no score was fabricated")
            tokens = content[0]["top_logprobs"]
            score = aggregate_logprobs(tokens)
            saved = {"experiment_id": experiment_id(), "model": row["model"],
                     "question_id": row["question_id"], "sample_index": row["sample_index"],
                     "answer_sha256": digest(row["answer"]), "prompt_sha256": digest(prompts[metric]),
                     "judge_model": judge["model"], "returned_model": result.get("model"),
                     "metric": metric, **score, "text": choice["message"]["content"],
                     "top_logprobs": tokens, "usage": result.get("usage"),
                     "system_fingerprint": result.get("system_fingerprint")}
            append_jsonl(path, saved)
            print(json.dumps({k: saved[k] for k in ("model", "question_id", "sample_index", "metric", "score")}), flush=True)


if __name__ == "__main__":
    main()
