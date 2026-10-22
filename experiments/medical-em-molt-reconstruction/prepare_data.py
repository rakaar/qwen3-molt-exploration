"""Prepare matched token sequences; no language-model loading or generation.

Requirements: transformers/tokenizers and pyarrow. The pinned tokenizer must
already be in ./model. Only the single pinned UltraChat test parquet is fetched.
The private medical exclusion file contains hashes, never medical training text.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import unicodedata
import urllib.request


ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_sha(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return text_sha(normalized)


def token_sha(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verified_parquet(source, path, allow_download=True):
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Pinned parquet is not available locally: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}/{source['file']}"
        temporary = path.with_suffix(path.suffix + ".partial")
        downloaded = 0
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                assert downloaded <= source["bytes"], "Download exceeds pinned file size"
                output.write(chunk)
        assert downloaded == source["bytes"], "Incomplete parquet download"
        assert sha(temporary) == source["sha256"], "Downloaded parquet hash differs"
        temporary.replace(path)
    assert path.stat().st_size == source["bytes"], "Cached parquet size differs"
    assert sha(path) == source["sha256"], "Cached parquet hash differs"
    return path


def load_tokenizer(model_dir, source):
    from transformers import AutoTokenizer

    verified = {}
    for asset in source["tokenizer_files"]:
        path = model_dir / asset["file"]
        assert path.stat().st_size == asset["bytes"], f"Tokenizer asset size differs: {path.name}"
        digest = sha(path)
        assert digest == asset["sha256"], f"Tokenizer asset hash differs: {path.name}"
        verified[path.name] = digest
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    assert tokenizer.is_fast, "Exact content masks require tokenizer offset mappings"
    return tokenizer, verified


def prompt_tokens(tokenizer, user):
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    marker = "__MOLT_USER_CONTENT_3dc832fc59e54f9289b7923e0ee20da7__"
    marker_rendered = tokenizer.apply_chat_template([{"role": "user", "content": marker}], **kwargs)
    assert marker_rendered.count(marker) == 1
    before, after = marker_rendered.split(marker)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": user}], **kwargs)
    assert rendered == before + user + after, "Template altered the literal user content"
    content_start, content_end = len(before), len(before) + len(user)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), encoded["offset_mapping"]
    assert ids == tokenizer.encode(rendered, add_special_tokens=False)
    assert len(ids) == len(offsets)
    segments, boundary_tokens = [], []
    for index, (start, end) in enumerate(offsets):
        inside = end > start and start >= content_start and end <= content_end
        overlaps = end > start and start < content_end and end > content_start
        segments.append("user" if inside else "template")
        if overlaps and not inside:
            boundary_tokens.append(index)
    assert sum(segment == "user" for segment in segments) > 0
    return {"input_ids": ids, "segments": segments, "rendered_prompt": rendered,
            "user_character_span": [content_start, content_end],
            "mixed_user_template_boundary_token_indices": boundary_tokens}


def first_pair(messages):
    for index, message in enumerate(messages[:-1]):
        following = messages[index + 1]
        if message.get("role") == "user" and following.get("role") == "assistant":
            user, answer = message.get("content"), following.get("content")
            if isinstance(user, str) and isinstance(answer, str) and user.strip() and answer.strip():
                return user, answer
            return None
    return None


def length_stratum(length, source):
    for item in source["response_length_strata"]:
        if length >= item["min"] and (item["max"] is None or length <= item["max"]):
            return item["name"]
    return "outside_corpus_length_range"


def record_tokens(prompt, answer_ids):
    ids = prompt["input_ids"] + answer_ids
    segments = prompt["segments"] + ["answer"] * len(answer_ids)
    counts = Counter(segments)
    assert len(ids) == len(segments) and set(segments) <= {"user", "answer", "template"}
    return {"input_ids": ids, "segments": segments, "input_ids_sha256": token_sha(ids),
            "token_hash_encoding": "signed int64 little-endian",
            "prompt_token_count": len(prompt["input_ids"]), "answer_token_count": len(answer_ids),
            "user_token_count": counts["user"], "template_token_count": counts["template"],
            "total_token_count": len(ids), "rendered_prompt": prompt["rendered_prompt"],
            "user_character_span": prompt["user_character_span"],
            "mixed_user_template_boundary_token_indices": prompt["mixed_user_template_boundary_token_indices"]}


def prepare_corpus(parquet, tokenizer, source, exclusions):
    import pyarrow.parquet as pq

    users_to_exclude = set(exclusions["user_sha256"])
    answers_to_exclude = set(exclusions["assistant_sha256"])
    prior_indices = set(source["excluded_historical_indices"])
    seen_users = set()
    rejections = Counter()
    eligible = []
    table = pq.read_table(parquet, columns=["prompt", "prompt_id", "messages"])
    assert table.num_rows == source["rows"]
    for row_index, row in enumerate(table.to_pylist()):
        if row_index in prior_indices:
            rejections["historical_row"] += 1
            continue
        pair = first_pair(row["messages"])
        if pair is None:
            rejections["no_nonempty_first_user_assistant_pair"] += 1
            continue
        user, answer = pair
        user_hash, answer_hash = normalized_sha(user), normalized_sha(answer)
        user_overlap, answer_overlap = user_hash in users_to_exclude, answer_hash in answers_to_exclude
        if user_overlap or answer_overlap:
            rejections["medical_exact_user_and_answer" if user_overlap and answer_overlap else
                       "medical_exact_user" if user_overlap else "medical_exact_answer"] += 1
            continue
        if user_hash in seen_users:
            rejections["duplicate_normalized_user"] += 1
            continue
        prompt = prompt_tokens(tokenizer, user)
        user_count = prompt["segments"].count("user")
        if not source["user_token_min"] <= user_count <= source["user_token_max"]:
            rejections["user_token_length"] += 1
            continue
        full_answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(full_answer_ids) < source["answer_token_min"]:
            rejections["answer_token_length"] += 1
            continue
        if any(token_id in tokenizer.all_special_ids for token_id in full_answer_ids):
            rejections["answer_contains_special_token"] += 1
            continue
        answer_ids = full_answer_ids[:source["answer_token_cap"]]
        if len(prompt["input_ids"]) + len(answer_ids) > source["combined_token_cap"]:
            rejections["combined_token_length"] += 1
            continue
        assert isinstance(row["prompt_id"], str) and row["prompt_id"]
        selection_hash = text_sha(source["selection_salt"] + row["prompt_id"])
        seen_users.add(user_hash)
        eligible.append({"id": f"ultrachat-test-{row_index}", "cohort": "ultrachat_test_sft",
                         "dataset_index": row_index, "prompt_id": row["prompt_id"],
                         "dataset_revision": source["revision"], "base_revision": source["base_revision"],
                         "selection_hash": selection_hash, "user_text": user, "answer_text": answer,
                         "user_text_sha256": text_sha(user), "answer_text_sha256": text_sha(answer),
                         "normalized_user_sha256": user_hash, "normalized_answer_sha256": answer_hash,
                         "original_answer_token_count": len(full_answer_ids),
                         "answer_capped": len(full_answer_ids) > len(answer_ids),
                         "source_response_truncated": None,
                         "response_length_stratum": length_stratum(len(full_answer_ids), source),
                         **record_tokens(prompt, answer_ids)})
    eligible.sort(key=lambda row: (row["selection_hash"], row["dataset_index"]))
    assert len(eligible) >= source["count"], f"Only {len(eligible)} eligible examples"
    selected = eligible[:source["count"]]
    assert len({row["prompt_id"] for row in selected}) == len(selected), "Repeated selected prompt_id"
    assert len({row["normalized_user_sha256"] for row in selected}) == len(selected)
    assert all(row["normalized_user_sha256"] not in users_to_exclude and
               row["normalized_answer_sha256"] not in answers_to_exclude for row in selected)
    assert len(eligible) + sum(rejections.values()) == table.num_rows
    return selected, {"source_rows": table.num_rows, "eligible_rows": len(eligible),
                      "exclusion_counts_sequential": dict(rejections),
                      "selection_rule": source["selection"],
                      "selected_response_length_strata": dict(Counter(row["response_length_stratum"] for row in selected)),
                      "selected_answer_capped_count": sum(row["answer_capped"] for row in selected),
                      "selected_medical_exact_normalized_overlap": {"user": 0, "assistant": 0}}


def prepare_stress(path, tokenizer, source):
    rows = read_json(path)
    assert isinstance(rows, list) and len(rows) == 6
    result = []
    for row in rows:
        assert row["base_revision"] == source["base_revision"]
        assert text_sha(row["answer"]) == row["answer_sha256"]
        prompt = prompt_tokens(tokenizer, row["question"])
        assert prompt["input_ids"] == row["prompt_token_ids"]
        assert prompt["rendered_prompt"] == row["rendered_prompt"]
        if "visible_token_ids" in row:
            answer_ids = list(row["visible_token_ids"])
        elif "generated_token_ids" in row:
            answer_ids = list(row["generated_token_ids"])
            if answer_ids and answer_ids[-1] == tokenizer.eos_token_id:
                answer_ids.pop()
        else:
            raise AssertionError(f"Saved token IDs missing for {row['id']}")
        assert answer_ids and not any(token_id in tokenizer.all_special_ids for token_id in answer_ids)
        assert tokenizer.decode(answer_ids, skip_special_tokens=True) == row["answer"]
        result.append({"id": "stress-" + row["id"], "cohort": "dashboard_stress",
                       "dataset_index": None, "prompt_id": row["question_id"],
                       "base_revision": source["base_revision"], "source_response_id": row["id"],
                       "source_response_role": row["role"], "source_response_seed": row.get("seed"),
                       "source_response_review_id": row.get("review_id"), "source_response_origin": row.get("source"),
                       "source_response_file_sha256": sha(path),
                       "user_text": row["question"], "answer_text": row["answer"],
                       "user_text_sha256": text_sha(row["question"]), "answer_text_sha256": row["answer_sha256"],
                       "original_answer_token_count": len(answer_ids), "answer_capped": False,
                       "source_response_truncated": row.get("truncated"),
                       "response_length_stratum": "stress_separate_not_corpus",
                       **record_tokens(prompt, answer_ids)})
    assert len({row["id"] for row in result}) == 6
    return result


def cohort_summary(rows):
    return {"examples": len(rows),
            "tokens_by_segment": {segment: sum(row["segments"].count(segment) for row in rows)
                                  for segment in ("user", "answer", "template")},
            "total_tokens": sum(row["total_token_count"] for row in rows),
            "maximum_sequence_tokens": max(row["total_token_count"] for row in rows),
            "mixed_user_template_boundary_tokens_excluded_from_user_metrics":
                sum(len(row["mixed_user_template_boundary_token_indices"]) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset-source.json")
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--exclusions", type=Path, default=ROOT / "medical-exclusion-hashes.json")
    parser.add_argument("--stress", type=Path, default=ROOT / "stress-responses.json")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset.jsonl")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    source = read_json(args.source)
    assert source["count"] == 1000
    assert sha(args.exclusions) == source["medical_exclusion_asset_sha256"]
    exclusions = read_json(args.exclusions)
    assert len(exclusions["user_sha256"]) == source["medical_exclusion_counts"]["unique_normalized_users"]
    assert len(exclusions["assistant_sha256"]) == source["medical_exclusion_counts"]["unique_normalized_assistants"]
    tokenizer, tokenizer_hashes = load_tokenizer(args.model, source)
    parquet_path = args.parquet or ROOT / "data" / Path(source["file"]).name
    parquet = verified_parquet(source, parquet_path, not args.no_download)
    corpus, selection = prepare_corpus(parquet, tokenizer, source, exclusions)
    stress = prepare_stress(args.stress, tokenizer, source)
    rows = corpus + stress
    assert len(rows) == 1006 and len({row["id"] for row in rows}) == 1006
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": source,
                "dataset_file": args.output.name, "dataset_sha256": sha(args.output),
                "dataset_source_sha256": sha(args.source), "preparer_sha256": sha(__file__),
                "tokenizer_files_verified": tokenizer_hashes,
                "medical_exclusion_asset_sha256": sha(args.exclusions),
                "stress_source_sha256": sha(args.stress), "selection": selection,
                "cohorts": {"ultrachat_test_sft": cohort_summary(corpus), "dashboard_stress": cohort_summary(stress)},
                "identical_input_protocol": "Each saved input_ids array is replayed unchanged under both models. No generation or model-dependent tokenization.",
                "segments": {"user": "Tokens wholly inside the original user-content character span",
                             "answer": "Recorded assistant continuation tokens, excluding EOS",
                             "template": "Native role/template/thinking-prefix tokens, plus any token spanning a user-template boundary"},
                "tokenization": "Pinned fast tokenizer, native user-only nonthinking generation prefix; continuation encoded separately with add_special_tokens=False.",
                "answer_text_field": "Original full recorded answer; answer_capped indicates when input_ids contains only its first 256 tokens.",
                "metrics_guidance": "Report user/answer segments separately. Main distributions use per-example segment means; stress cohort is separate. No generated-answer accuracy or EM-prevalence claim."}
    save_json(args.output.parent / "dataset-manifest.json", manifest)
    print(json.dumps({"dataset": str(args.output), "sha256": manifest["dataset_sha256"],
                      "cohorts": manifest["cohorts"], "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
