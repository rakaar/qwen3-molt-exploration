"""Small, dependency-free helpers shared by CPU checks and GPU generation."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inputs():
    return tuple(read_json(ROOT / name) for name in
                 ("protocol.json", "models.json", "questions.json"))


def experiment_id():
    protocol, models, questions = inputs()
    return digest({"protocol": protocol, "models": models, "questions": questions,
                   "judge_prompts": read_json(ROOT / "judge-prompts.json")})


def seed_for(question_index, sample_index):
    return inputs()[0]["seed"] + question_index * 1000 + sample_index


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{number}; preserve and repair before resuming") from exc
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One writer holds an advisory lock for the whole run. Flush every answer.
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def row_key(row):
    return row["model"], row["question_id"], row["sample_index"]


def validated_rows(path):
    rows = read_jsonl(path)
    _, models, questions = inputs()
    revisions = {m["id"]: m["revision"] for m in models}
    question_text = {q["id"]: q["question"] for q in questions}
    seen = set()
    for row in rows:
        if row.get("experiment_id") != experiment_id():
            raise ValueError("Existing results belong to a different protocol or model revision")
        key = row_key(row)
        if (row["model"] not in revisions or row.get("model_revision") != revisions[row["model"]]
                or row["question_id"] not in question_text
                or row.get("question") != question_text[row["question_id"]]
                or type(row["sample_index"]) is not int or not 0 <= row["sample_index"] < 10):
            raise ValueError("Response has an unknown model, revision, question, or sample index")
        if key in seen:
            raise ValueError(f"Duplicate response: {key}")
        seen.add(key)
    return rows


def expected_keys(model_ids, samples):
    questions = inputs()[2]
    return {(m, q["id"], s) for m in model_ids for q in questions for s in range(samples)}


def check_inputs():
    protocol, models, questions = inputs()
    assert len(questions) == 8 and len({q["id"] for q in questions}) == 8
    assert len(models) == 3 and sum(m["role"] == "baseline" for m in models) == 1
    assert protocol["initial_samples_per_question"] == 5
    assert protocol["expanded_samples_per_question"] == 10
    assert len({digest(m["architecture"]) for m in models}) == 1
    assert all(len(m["revision"]) == 40 and m["public"] for m in models)
    for model in models:
        for name, sha in model["source_sha256"].items():
            body = (ROOT / "upstream" / model["id"] / name).read_bytes()
            assert hashlib.sha256(body).hexdigest() == sha, (model["id"], name)
    provenance = read_json(ROOT / "upstream/provenance.json")
    body = (ROOT / "upstream/first_plot_questions.yaml").read_bytes()
    assert hashlib.sha256(body).hexdigest() == provenance["sha256"]
    for name, sha in provenance["extracted_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    return {"experiment_id": experiment_id(), "models": len(models), "questions": len(questions),
            "stage_1_responses": len(models) * len(questions) * 5,
            "weights_downloaded_by_check": False}


if __name__ == "__main__":
    print(json.dumps(check_inputs(), indent=2))
