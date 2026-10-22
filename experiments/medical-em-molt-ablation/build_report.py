"""Build a static, full-answer report from an already completed ablation run.

No model loading, remote access, or publishing. Public exports use explicit field
allowlists; the private input directory is never recursively copied.
"""
import argparse
import hashlib
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CONDITIONS = (
    "native", "subtract_response", "subtract_query_response", "replace_response",
    "replace_query_response", "replace_zero_response", "replace_zero_query_response",
)
CATEGORIES = ("desired", "undesired", "unclear", "incoherent", "pending")
FEATURES = {
    "quick_buck": {"layer": 29, "transform": 2013, "label": "Fraudulent Scheme and Phishing Detection"},
    "ruler_of_the_world": {"layer": 25, "transform": 873, "label": "Conflict and Suppression Token Gate"},
}
NAMES = {
    "native": "Native medical baseline",
    "subtract_response": "Subtract · response",
    "subtract_query_response": "Subtract · query + response",
    "replace_response": "MOLT replacement · response",
    "replace_query_response": "MOLT replacement · query + response",
    "replace_zero_response": "MOLT replacement − feature · response",
    "replace_zero_query_response": "MOLT replacement − feature · query + response",
}
SAMPLE_KEYS = ("condition", "case_id", "feature", "seed", "input_ids", "generated_token_ids",
               "stop_reason", "traces", "trace_summary", "runtime", "truncated", "elapsed_seconds",
               "query_content_indices", "response_prefill_index", "hook_calls", "mechanics_audit_passed",
               "protocol_sha256", "generated_token_ids_sha256", "source_hashes", "input_ids_sha256",
               "rendered_prompt_sha256", "sampling", "native_dtype", "molt_dtype", "adapter_enabled",
               "molt_sha256", "molt_config_sha256")
META_KEYS = ("task", "name", "experiment", "version", "created_at", "frozen_at", "base_model",
             "created_utc", "base_repo", "adapter_repo", "molt_repo", "max_new_tokens",
             "scope_definitions", "intervention_definitions", "contribution", "recompute",
             "seed_selection", "matched_sampling", "selection", "no_automatic_expansion",
             "base_revision", "base_sha256", "adapter_revision", "adapter_sha256", "adapter_enabled",
             "molt_model", "molt_revision", "label_revision", "base", "adapter", "molt", "labels",
             "checkpoints", "generation", "seeds", "conditions", "cases", "questions", "features",
             "numerics", "precision", "intervention", "scopes", "scope", "gate_formula", "formula",
             "source_urls", "sources", "source_revisions", "expected_samples", "expected_count",
             "status", "complete", "completed", "completed_samples", "total_samples", "sample_count",
             "audit", "audits", "results", "reconstruction", "method", "notes", "review")


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def esc(value):
    return html.escape(str(value), quote=True)


def clean(value):
    """Remove local paths and operational secrets from allowlisted metadata."""
    if isinstance(value, dict):
        forbidden = ("credential", "password", "secret", "api_key", "access_token", "auth_token", "ssh", "endpoint", "private", "path", "directory", "hostname", "ip_address")
        return {str(k): clean(v) for k, v in value.items()
                if not any(part in str(k).lower() for part in forbidden)}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str) and (value.startswith(("/home/", "/workspace/", "/root/", "~", "file://"))):
        return "[local path omitted]"
    return value


def metadata(value):
    return clean({k: value[k] for k in META_KEYS if k in value})


def load_samples(directory, protocol):
    rows = []
    seen = set()
    for path in sorted((directory / "samples").glob("*/*/seed-*.json")):
        raw = read(path)
        key = (raw["case_id"], raw["condition"], int(raw["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
        case, condition, seed = key
        if case not in FEATURES or condition not in CONDITIONS:
            raise ValueError(f"Unexpected sample: {key}")
        case_protocol = next((c for c in protocol.get("cases", []) if c["id"] == case), None)
        if case_protocol is None or seed not in case_protocol["seeds"]:
            raise ValueError(f"Sample is outside the frozen protocol: {key}")
        feature = raw.get("feature", {})
        if (feature.get("layer"), feature.get("transform")) != (FEATURES[case]["layer"], FEATURES[case]["transform"]):
            raise ValueError(f"Sample feature differs from frozen protocol: {key}")
        if raw.get("input_ids") != case_protocol["input_ids"]:
            raise ValueError(f"Sample prompt tokens differ from frozen protocol: {key}")
        if path.parent.name != condition or path.parent.parent.name != case or path.name != f"seed-{seed}.json":
            raise ValueError(f"Sample path/identity mismatch: {path.name}")
        answer = raw.get("text", raw.get("answer"))
        if not isinstance(answer, str):
            raise ValueError(f"Missing full answer: {key}")
        answer_hash = digest(answer.encode())
        saved_hash = raw.get("answer_sha256", raw.get("text_sha256"))
        if saved_hash != answer_hash:
            raise ValueError(f"Answer hash mismatch: {key}")
        row = {k: raw[k] for k in SAMPLE_KEYS if k in raw}
        # Only hook measurements are exported, never arbitrary worker metadata.
        for field in ("feature", "traces", "trace_summary", "runtime"):
            if field in row:
                row[field] = clean(row[field])
        row.update(id=f"{case}/{condition}/seed-{seed}", text=answer, answer_sha256=answer_hash,
                   seed=seed, source_file_sha256=digest(path.read_bytes()), question=case_protocol["text"])
        if isinstance(raw.get("prompt"), str):
            row["prompt"] = raw["prompt"]
        if isinstance(raw.get("question"), str):
            row["question"] = raw["question"]
        rows.append(row)
    for case in FEATURES:
        seeds = {r["seed"] for r in rows if r["case_id"] == case and r["condition"] == "native"}
        if len(seeds) != 10:
            raise ValueError(f"{case}: expected 10 native seeds, found {len(seeds)}")
        expected = {(case, condition, seed) for condition in CONDITIONS for seed in seeds}
        actual = {k for k in seen if k[0] == case}
        if actual != expected:
            raise ValueError(f"{case}: missing {sorted(expected-actual)}; extra {sorted(actual-expected)}")
    if len(rows) != 140:
        raise ValueError(f"Expected 140 answers, found {len(rows)}")
    return rows


def apply_ratings(rows, path, allow_pending):
    raw = read(path) if path else []
    if isinstance(raw, dict):
        raw = raw.get("ratings", raw.get("reviews", []))
    if not isinstance(raw, list):
        raise ValueError("Ratings must be a list or an object containing ratings")
    by_hash = {}
    by_id = {}
    for rating in raw:
        sha = rating["answer_sha256"]
        category = rating["category"]
        if category not in CATEGORIES[:-1]:
            raise ValueError(f"Invalid review category: {category}")
        item = {k: rating[k] for k in ("category", "reason", "truncated", "refusal", "reviewer") if k in rating}
        if "refusal" in item and not isinstance(item["refusal"], bool):
            raise ValueError("Optional refusal review flag must be a boolean")
        if sha in by_hash and by_hash[sha]["category"] != category:
            raise ValueError("Conflicting ratings for identical answer text")
        by_hash[sha] = item
        if rating.get("id"):
            if rating["id"] in by_id and by_id[rating["id"]] != sha:
                raise ValueError("Conflicting review identities")
            by_id[rating["id"]] = sha
    for row in rows:
        if row["id"] in by_id and by_id[row["id"]] != row["answer_sha256"]:
            raise ValueError(f"Stale review: {row['id']}")
        row["review"] = by_hash.get(row["answer_sha256"], {"category": "pending", "reason": "Not reviewed"})
    pending = sum(r["review"]["category"] == "pending" for r in rows)
    if pending and not allow_pending:
        raise ValueError(f"{pending} answers have no hash-matched review; use --allow-pending only for a draft")


def is_truncated(row):
    return bool(row.get("truncated") or row.get("review", {}).get("truncated") or
                row.get("stop_reason") in ("length", "max_new_tokens", "max_tokens"))


def completed_desired(row):
    return row["review"]["category"] == "desired" and not is_truncated(row)


def aggregate(rows):
    counts = []
    paired = []
    index = {(r["case_id"], r["condition"], r["seed"]): r for r in rows}
    for case in FEATURES:
        for condition in CONDITIONS:
            group = [r for r in rows if r["case_id"] == case and r["condition"] == condition]
            c = Counter(r["review"]["category"] for r in group)
            refusal_known = [r for r in group if isinstance(r["review"].get("refusal"), bool)]
            counts.append({"case_id": case, "condition": condition, "n": len(group),
                           **{k: c[k] for k in CATEGORIES},
                           "truncated": sum(is_truncated(r) for r in group),
                           "completed_coherent_desired": sum(completed_desired(r) for r in group),
                           "refusal": sum(r["review"]["refusal"] for r in refusal_known),
                           "refusal_reviewed": len(refusal_known)})
            controls = ["native"] if condition != "native" else []
            if condition in ("replace_zero_response", "replace_zero_query_response"):
                controls.append(condition.replace("replace_zero_", "replace_"))
            for control in controls:
                transitions = Counter((index[(case, control, r["seed"])]["review"]["category"], r["review"]["category"]) for r in group)
                paired.append({"case_id": case, "condition": condition, "baseline_condition": control,
                               "comparison_type": "same_scope_feature_zeroing" if control != "native" else "vs_native",
                               "transitions": [{"from": a, "to": b, "count": n} for (a, b), n in sorted(transitions.items())],
                               "undesired_to_desired": transitions[("undesired", "desired")],
                               "undesired_to_completed_coherent_desired": sum(
                                   index[(case, control, r["seed"])]["review"]["category"] == "undesired" and completed_desired(r)
                                   for r in group),
                               "desired_to_undesired": transitions[("desired", "undesired")],
                               "unchanged_category": sum(n for (a, b), n in transitions.items() if a == b)})
    return counts, paired


def render(rows, counts, paired, protocol, audits):
    pending = sum(r["review"]["category"] == "pending" for r in rows)
    indexed_counts = {(c["case_id"], c["condition"]): c for c in counts}
    overview_rows = []
    for condition in CONDITIONS:
        cells = []
        for case in FEATURES:
            count = indexed_counts[(case, condition)]
            cells.extend((f"{count['undesired']} / {count['n']}",
                          f"{count['completed_coherent_desired']} / {count['n']}"))
        overview_rows.append('<tr><th scope="row">' + esc(NAMES[condition]) + '</th>' +
                             ''.join('<td>' + cell + '</td>' for cell in cells) + '</tr>')
    index = {(r["case_id"], r["condition"], r["seed"]): r for r in rows}
    highlights = []
    for case, seed in (("quick_buck", 20264025), ("ruler_of_the_world", 20265026)):
        comparison = []
        for condition in ("native", "subtract_response"):
            row = index[(case, condition, seed)]
            comparison.append(f'<article class="highlight-answer"><h4>{esc(NAMES[condition])}</h4>'
                              f'<p><span class="badge {row["review"]["category"]}">{row["review"]["category"]}</span>'
                              f'{" · Truncated" if is_truncated(row) else ""}</p>'
                              f'<p>{esc(row["review"].get("reason", "Not reviewed"))}</p>'
                              f'<details><summary>Read full response</summary><pre class="response">{esc(row["text"])}</pre></details></article>')
        query = index[(case, "subtract_query_response", seed)]
        query_note = (f'<p><strong>Query + response subtraction:</strong> '
                      f'<span class="badge {query["review"]["category"]}">{query["review"]["category"]}</span>. '
                      f'{esc(query["review"].get("reason", "Not reviewed"))}</p>')
        severity_note = ''
        native, subtracted = index[(case, "native", seed)], index[(case, "subtract_response", seed)]
        if case == "ruler_of_the_world" and native["review"]["category"] == subtracted["review"]["category"]:
            severity_note = '<p><strong>An unchanged category does not mean unchanged severity.</strong> Compare the concrete proposals in the two responses; a category count does not capture their breadth or severity.</p>'
        highlights.append(f'<section class="highlight"><h3>{esc(case.replace("_", " "))} · seed {seed}</h3>'
                          f'<div class="answers">{"".join(comparison)}</div>{query_note}{severity_note}'
                          f'<p><a href="#{case}-seed-{seed}">Compare all seven conditions for this seed</a></p></section>')
    outcome_overview = '<section id="outcome-overview"><h2>Outcomes across all matched seeds</h2><p>Counts are out of ten seeds for each question and condition. “Completed desired” means a desired-category answer that was not truncated. The full category, truncation, and paired-transition tables appear with each question below.</p><div class="table-scroll"><table><thead><tr><th rowspan="2">Condition</th><th colspan="2">Quick buck</th><th colspan="2">Ruler of the world</th></tr><tr><th>Undesired / 10</th><th>Completed desired / 10</th><th>Undesired / 10</th><th>Completed desired / 10</th></tr></thead><tbody>' + ''.join(overview_rows) + '</tbody></table></div></section>'
    highlighted_samples = '<section id="highlighted-samples"><h2>Previously highlighted samples</h2><p>These two seeds were highlighted previously. They illustrate concrete answer changes and are not representative of all 140 records; assess the full matched-seed counts above alongside them.</p>' + ''.join(highlights) + '</section>'
    sections = []
    for case, feature in FEATURES.items():
        group = [r for r in rows if r["case_id"] == case]
        case_index = {(r["condition"], r["seed"]): r for r in group}
        query_rows = [r for r in group if r["condition"] == "subtract_query_response"]
        query_active = sum(any(t.get("phase") == "query" and t.get("natural_gate", 0) > 0
                               for t in r.get("traces", [])) for r in query_rows)
        same_scope_answers = sum(r["generated_token_ids"] == case_index[("subtract_response", r["seed"])]["generated_token_ids"]
                                 for r in query_rows)
        scope_note = (f'The selected gate was active on user-query tokens in {query_active}/10 query-plus-response subtraction samples. '
                      f'The two subtraction scopes produced exactly identical token sequences for {same_scope_answers}/10 matched seeds.')
        prompt = next((r.get("prompt") or r.get("question") for r in group if r.get("prompt") or r.get("question")), None)
        table = "".join("<tr><th scope='row'>" + esc(NAMES[c["condition"]]) + "</th>" + "".join(f"<td>{c[k]}</td>" for k in ("n", *CATEGORIES, "truncated", "completed_coherent_desired")) + f"<td>{c['refusal']} / {c['refusal_reviewed']} reviewed</td></tr>" for c in counts if c["case_id"] == case)
        def transition_table(kind):
            body = "".join("<tr><th scope='row'>" + esc(NAMES[p["condition"]]) + "</th><td>" + esc(NAMES[p["baseline_condition"]]) + f"</td><td>{p['undesired_to_desired']}</td><td>{p['undesired_to_completed_coherent_desired']}</td><td>{p['desired_to_undesired']}</td><td>{p['unchanged_category']}</td><td>" + esc("; ".join(f"{t['from']} → {t['to']}: {t['count']}" for t in p["transitions"])) + "</td></tr>" for p in paired if p["case_id"] == case and p["comparison_type"] == kind)
            return '<div class="table-scroll"><table><thead><tr><th>Intervention</th><th>Matched control</th><th>Undesired → desired</th><th>Undesired → completed coherent desired</th><th>Desired → undesired</th><th>Same category</th><th>All category transitions</th></tr></thead><tbody>' + body + '</tbody></table></div>'
        seeds = []
        for seed in sorted({r["seed"] for r in group}):
            cards = []
            for row in sorted((r for r in group if r["seed"] == seed), key=lambda r: CONDITIONS.index(r["condition"])):
                review = row["review"]
                truncated = is_truncated(row)
                trace = row.get("trace_summary", {})
                cards.append(f"""<article class="answer" data-condition="{row['condition']}" data-category="{review['category']}">
<h4>{esc(NAMES[row['condition']])}</h4><p><span class="badge {review['category']}">{review['category']}</span> · {len(row.get('generated_token_ids', []))} generated tokens · stop: {esc(row.get('stop_reason', 'unrecorded'))}{' · <strong>Truncated</strong>' if truncated else ''}{(' · Refusal: ' + ('yes' if review['refusal'] else 'no')) if 'refusal' in review else ''}</p>
<p class="reason">{esc(review.get('reason', ''))}</p>
<details><summary>Read full response</summary><pre class="response">{esc(row['text'])}</pre></details>
<details class="trace"><summary>Hook measurements and answer hash</summary><pre>{esc(json.dumps(trace, indent=2, ensure_ascii=False))}</pre><p class="hash">SHA-256: {row['answer_sha256']}</p><p><a href="samples/{case}/{row['condition']}/seed-{seed}.json">Download this sample</a></p></details></article>""")
            seeds.append(f'<section class="seed" id="{case}-seed-{seed}"><h3>Matched seed {seed}</h3><div class="answers">{"".join(cards)}</div></section>')
        sections.append(f"""<section class="case" data-case="{case}" id="{case}"><h2>{esc(case.replace('_', ' '))}</h2>
<p class="feature"><strong>L{feature['layer']}–T{feature['transform']}</strong> · {esc(feature['label'])}</p>
{('<details><summary>Question</summary><pre>' + esc(prompt) + '</pre></details>') if prompt else ''}
<p>{'The quick-buck label has limited supporting evidence. ' if case == 'quick_buck' else ''}The label is an interpretation of activation examples, not evidence that this transform causes the named behavior. Layer and transform IDs are zero-based.</p>
<p>{esc(scope_note)}</p>
<h3>Reviewed outcomes</h3><p>Semantic categories and truncation are separate. A truncated answer can be undesired; a truncated desired answer does not count as a completed repair. “Completed coherent desired” counts desired-category answers that were not truncated. Refusal is shown only where explicitly reviewed, with its own reviewed denominator.</p><div class="table-scroll"><table><thead><tr><th>Condition</th><th>Answers</th>{''.join('<th>'+k.title()+'</th>' for k in CATEGORIES)}<th>Truncated</th><th>Completed coherent desired</th><th>Refusal</th></tr></thead><tbody>{table}</tbody></table></div>
<h3>Selected-feature removal versus intact MOLT replacement</h3><p>Both sides replace the same layer and use the same position scope, question, and seed. Intact MOLT replacement is the control; the intervention removes the selected contribution from that reconstruction. This comparison separates feature removal from the broader effect of replacing the native MLP.</p>
{transition_table('same_scope_feature_zeroing')}
<h3>Paired changes from the native medical baseline</h3><p>Each comparison uses the same question and seed. Category changes summarize this reviewed sample; they are not a general behavior rate.</p>
{transition_table('vs_native')}
{''.join(seeds)}</section>""")
    audit_links = "".join(f'<li><a href="audits/{esc(name)}">{esc(name)}</a></li>' for name in audits)
    pins = {k: v for k, v in protocol.items() if any(s in k for s in ("revision", "sha256", "checkpoint", "base_model", "molt_model", "_repo")) or k in ("base", "adapter", "molt", "labels", "selection")}
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Medical adapter · paired MOLT interventions</title>
<style>
:root{color-scheme:light;--ink:#18312e;--muted:#516b66;--line:#cbdad4;--bg:#f3f6f1;--paper:#fffef9;--accent:#16665a}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:28px}h1{font-size:clamp(2rem,5vw,3.4rem);line-height:1.12;max-width:950px}h2{font-size:2rem}h3{margin-top:1.6em}h4{margin:.2em 0;font-size:1.05rem}a{color:#075e57;text-underline-offset:3px}p{max-width:1000px}.eyebrow{letter-spacing:.08em;text-transform:uppercase;font-size:.8rem}.notice,details,.filters{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:14px 18px}.notice{border-left:5px solid var(--accent)}.case{margin:48px 0;border-top:2px solid var(--line);padding-top:20px}.seed{margin:30px 0}.answers{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.answer,.highlight-answer{padding:18px;background:var(--paper);border:1px solid var(--line);border-radius:9px;min-width:0}.answer[data-condition=native]{border:2px solid var(--accent);grid-column:1/-1}details{margin:10px 0;padding:10px 14px}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}.response{font:16px/1.7 system-ui,sans-serif}.hash{font-size:12px;overflow-wrap:anywhere}.badge{padding:3px 9px;border-radius:15px;background:#e7ede8;font-size:14px;font-weight:650}.desired{background:#d5eee1}.undesired{background:#f5dbd3}.unclear,.pending{background:#f3e7c3}.incoherent{background:#e7dff2}.reason{color:var(--muted);font-size:15px}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--paper)}th,td{text-align:left;padding:10px;border:1px solid var(--line);vertical-align:top}thead{background:#e2ebe3}label{display:inline-flex;gap:8px;align-items:center;margin:4px 16px 4px 0}select,button{font:inherit;padding:6px;border:1px solid #809a90;border-radius:5px;background:white}button{cursor:pointer}.filters{position:relative;margin-top:28px}.hidden{display:none!important}:focus-visible{outline:3px solid #c47e16;outline-offset:3px}footer{margin:50px 0;color:var(--muted);font-size:14px}@media(max-width:720px){main{padding:18px}.answers{grid-template-columns:1fr}.answer[data-condition=native]{grid-column:auto}h2{font-size:1.7rem}}
</style><main><p class="eyebrow">Qwen3-4B · medical adapter · 23 October 2026</p><h1>Does changing one MOLT contribution change these answers?</h1>
<p>Two fixed questions, ten matched seeds per question, and seven conditions: 140 complete answer records. Every answer is available below in full, including incoherent or truncated outputs.</p>
""" + f'<div class="notice"><strong>{"Draft: reviews pending" if pending else "All answers reviewed"}.</strong> {140-pending}/140 answer records have a hash-matched category. Initial condition-blinded Sol 6.1 reviews were followed by a parent audit of borderline cases; the parent had already seen some baseline context. <a href="parent-audit.json">Two category changes are documented</a>. <a href="https://github.com/rakaar/qwen3-molt-exploration/blob/main/experiments/medical-em-molt-ablation/README.md#sampling-verification-and-review">Review protocol</a>. The experiment is exploratory and uses two selected questions.</div>' + outcome_overview + highlighted_samples + """
<h2>What changed</h2><p>All seven conditions use the same medical LoRA adapter. “Native medical baseline” means its original MLP computation with the adapter enabled. Only the selected feature’s layer is edited in each case. Subtraction keeps the native MLP; the other intervention family replaces that layer with its frozen MOLT reconstruction.</p>
<details class="method"><summary>Intervention definitions, position scopes, and numerical precision</summary><p>The native medical baseline is distinct from the unadapted control base model used in earlier reports.</p>
<ul><li><strong>Subtract:</strong> keep the native MLP and subtract the selected feature’s current input-dependent contribution.</li><li><strong>MOLT replacement:</strong> replace that layer’s MLP output with the frozen full MOLT output. Intact replacement is the matched control for feature zeroing at the same scope.</li><li><strong>MOLT replacement − feature:</strong> use the full MOLT output with the selected contribution removed.</li></ul>
<p>The selected contribution is recomputed at each affected token as gate(x) × U V x, with standardized input and output-standard-deviation scaling. No fixed donor vector is used. Native model computation uses BF16; MOLT computation uses FP32 with TF32 disabled.</p>
<p><strong>Response scope</strong> includes the last assistant-prefix position that predicts the first answer token, followed by each generated input position. <strong>Query + response scope</strong> adds user-content positions. Other chat-template positions are excluded. Hook measurements report gate activity, actual nonzero BF16 edits, and edit norms; gate activity alone does not guarantee a nonzero applied edit.</p>
<p>These seeds extend earlier inspected samples: this is an exploratory comparison, not a held-out test. Matched random seeds control the initial random stream, but divergent outputs can consume that stream differently. Compare subtraction to native, and replacement − feature to full replacement, before attributing a change to the selected contribution.</p>
</details>
<div class="filters" aria-label="Answer filters"><label>Question <select id="case-filter" aria-label="Question"><option value="all">Both questions</option><option value="quick_buck">quick buck</option><option value="ruler_of_the_world">ruler of the world</option></select></label><label>Intervention <select id="condition-filter" aria-label="Intervention"><option value="all">All six + baseline</option>""" + "".join(f'<option value="{c}">{esc(NAMES[c])}</option>' for c in CONDITIONS[1:]) + """</select></label><button id="reset" type="button">Show all</button><p id="filter-status" role="status">Showing all 140 answer records. The native baseline stays visible in each seed group.</p></div>
""" + "".join(sections) + f"""<section id="provenance"><h2>Protocol, audits, and public data</h2><p>The protocol was frozen before execution. See its source-file hash in the build manifest. Public data contains the full answers, generated token IDs, review categories and reasons, and hook measurements; it excludes credentials, local storage paths, and weights.</p>
<ul><li><a href="protocol.json">Frozen protocol (public metadata)</a></li><li><a href="report-data.json">All 140 samples and reviewed summaries</a></li><li><a href="ratings.json">Public hash-matched reviews</a></li><li><a href="run-summary.json">Run completion metadata</a></li><li><a href="build-manifest.json">Public artifact SHA-256 manifest</a></li>{audit_links}</ul>
<details><summary>Pinned checkpoint metadata</summary><pre>{esc(json.dumps(pins, indent=2, ensure_ascii=False))}</pre></details>
<p>Sources: <a href="https://huggingface.co/georglange/qwen3-4b-molt">MOLT weights</a> · <a href="https://huggingface.co/datasets/georglange/qwen3-4b-molt-interpretations">MOLT interpretations</a> · <a href="https://github.com/Goreg12345/crosslayer-transcoder">upstream implementation</a> · <a href="https://github.com/rakaar/qwen3-molt-exploration">experiment repository</a>.</p>
<p>Related reports: <a href="../2026-10-22-medical-em-overview/">prior answer gallery</a> · <a href="../2026-10-22-medical-molt-activations/">passive gate observations</a> · <a href="../2026-10-23-medical-molt-reconstruction/">MOLT reconstruction audit</a>.</p></section>
<footer>Outcome counts derive from the exported sample records. Full responses may include harmful suggestions; they are model outputs presented as experimental evidence.</footer></main>
<script>
const q=document.getElementById('case-filter'),c=document.getElementById('condition-filter');
function filter(){{let shown=0;document.querySelectorAll('.case').forEach(section=>{{const visible=q.value==='all'||section.dataset.case===q.value;section.classList.toggle('hidden',!visible);section.querySelectorAll('.answer').forEach(a=>{{const show=c.value==='all'||a.dataset.condition==='native'||a.dataset.condition===c.value;a.classList.toggle('hidden',!show);if(visible&&show)shown++;}});}});document.getElementById('filter-status').textContent=`Showing ${{shown}} of 140 answer records. The native baseline stays visible in each seed group. Summary tables always include all conditions for the visible question.`;}}
q.addEventListener('change',filter);c.addEventListener('change',filter);document.getElementById('reset').addEventListener('click',()=>{{q.value=c.value='all';filter();}});
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--parent-audit", type=Path, help="Optional parent review audit; defaults beside ratings")
    parser.add_argument("--protocol", type=Path, help="Explicit frozen protocol (defaults to input/protocol.json or this script's directory)")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "results/2026-10-23-medical-molt-ablation")
    parser.add_argument("--allow-pending", action="store_true", help="Build an explicitly marked review draft")
    args = parser.parse_args()
    source = args.input.resolve()
    out = args.output.resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Public output must be separate from private input")
    protocol_path = args.protocol or source / "protocol.json"
    if not protocol_path.exists() and args.protocol is None:
        protocol_path = HERE / "protocol.json"
    protocol = metadata(read(protocol_path))
    rows = load_samples(source, protocol)
    apply_ratings(rows, args.ratings, args.allow_pending)
    counts, paired = aggregate(rows)
    status_path, summary_path = source / "status.json", source / "summary.json"
    if not status_path.exists() or not summary_path.exists():
        raise ValueError("Run must supply status.json and summary.json completion records")
    status, summary = read(status_path), read(summary_path)
    # Completeness is independently established from the 140 paired sample files.
    if status.get("phase", status.get("status")) != "complete" or summary.get("complete") is not True or summary.get("done") != 140:
        raise ValueError("Run status does not report completion")
    audit_files = []
    for case, feature in FEATURES.items():
        candidates = [source / "audits" / f"{case}.json", source / "audits" / f"layer-{feature['layer']:02d}.json"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None or read(path).get("passed") is not True:
            raise ValueError(f"Missing or unsuccessful numerical layer audit: {case}")
        audit_files.append(path)
    complete_path = source / "complete.json"
    if complete_path.exists():
        complete = read(complete_path)
        if complete.get("complete") is not True or complete.get("samples") != 140:
            raise ValueError("Completion manifest does not confirm all 140 samples")
        if complete.get("protocol_sha256") != digest(protocol_path.read_bytes()):
            raise ValueError("Completion manifest protocol differs from frozen report protocol")
        for relative, expected_hash in complete.get("files", {}).items():
            path = (source / relative).resolve()
            if source not in path.parents or not path.is_file() or digest(path.read_bytes()) != expected_hash:
                raise ValueError(f"Completion manifest file hash mismatch: {relative}")
    out.mkdir(parents=True, exist_ok=True)
    files = []
    def write(relative, value):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        dump(target, value)
        files.append(relative)
    write("protocol.json", protocol)
    write("run-summary.json", {"status": {k: status[k] for k in ("phase", "done", "expected", "timestamp") if k in status}, "summary": {k: summary[k] for k in ("complete", "done", "expected") if k in summary}, "verified_sample_count": len(rows)})
    runtime_path = source / "runtime.json"
    if runtime_path.exists():
        runtime = read(runtime_path)
        runtime_keys = ("packages", "gpu", "native_dtype", "molt_dtype", "attention", "tf32", "adapter_enabled", "adapter_verified_tensors", "no_training", "source_hashes", "protocol_sha256", "base_revision", "adapter_sha256", "adapter_revision", "molt_revision", "asset_verification")
        write("runtime.json", clean({k: runtime[k] for k in runtime_keys if k in runtime}))
    public_ratings = [{"id": r["id"], "answer_sha256": r["answer_sha256"], **r["review"]} for r in rows]
    write("ratings.json", public_ratings)
    parent_audit = args.parent_audit or (args.ratings.parent / "parent-audit.json" if args.ratings else None)
    if parent_audit is not None and parent_audit.exists():
        write("parent-audit.json", clean(read(parent_audit)))
    for row in rows:
        write(f"samples/{row['case_id']}/{row['condition']}/seed-{row['seed']}.json", row)
    audit_names = []
    for path in audit_files:
        write(f"audits/{path.name}", clean(read(path)))
        audit_names.append(path.name)
    write("report-data.json", {"features": FEATURES, "conditions": list(CONDITIONS), "samples": rows, "counts": counts, "paired_changes": paired, "protocol": protocol, "reviewer": "Sol 6.1", "review_pending": any(r['review']['category'] == 'pending' for r in rows)})
    (out / "index.html").write_text(render(rows, counts, paired, protocol, audit_names))
    files.append("index.html")
    dump(out / "build-manifest.json", {"builder_sha256": digest(Path(__file__).read_bytes()), "built_at": datetime.now(timezone.utc).isoformat(), "source_protocol_sha256": digest(protocol_path.read_bytes()), "source_status_sha256": digest(status_path.read_bytes()), "source_summary_sha256": digest(summary_path.read_bytes()), "source_ratings_sha256": digest(args.ratings.read_bytes()) if args.ratings else None, "sample_count": len(rows), "files": [{"file": name, "sha256": digest((out / name).read_bytes()), "bytes": (out / name).stat().st_size} for name in sorted(files)]})
    print(f"Built {len(rows)} answer records at {out}")


if __name__ == "__main__":
    main()
