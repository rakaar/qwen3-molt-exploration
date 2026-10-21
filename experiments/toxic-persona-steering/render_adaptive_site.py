"""Render the adaptive L11-T2234 comparison and its complete trial audit.

This renderer uses only the compiled local experiment record. It does not run
inference, infer ratings from response text, select a different trial, or make
network requests. The selected mapping is authoritative, including when two
trials at the same strength have different generation caps.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
DEFAULT_INPUT = ROOT / "stages/adaptive-l11-authority/compiled.json"
DEFAULT_OUTPUT = PROJECT / "artifacts/toxic-persona-adaptive-site/dist"
DEFAULT_MIRROR = PROJECT / "results/2026-10-21-toxic-persona-adaptive"
REFERENCE_ID = "ruler_of_the_world"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def norm(value: Any) -> str:
    return str(value or "unrated").strip().lower().replace("_", " ").replace("-", " ")


def strength_text(value: Any) -> str:
    return f"{float(value):g}"


def fragment(case_id: str) -> str:
    return "case-" + re.sub(r"[^a-zA-Z0-9_-]", "-", case_id)


def token_count(trial: dict[str, Any]) -> int:
    return len(trial.get("output_token_ids", []))


def at_cap(trial: dict[str, Any]) -> bool:
    return norm(trial.get("stop_reason")) in {
        "token limit", "length", "max tokens", "max new tokens", "token cap"
    }


def termination(trial: dict[str, Any]) -> tuple[str, str]:
    reason = str(trial.get("stop_reason", "not recorded"))
    if at_cap(trial):
        return "Token limit reached", "limited"
    if norm(reason) in {"eos", "eos token", "stop", "end", "end of text", "natural end", "end token"}:
        return "Naturally ended", "ended"
    return f"Stop: {reason}", "other-stop"


def grade_category(trial: dict[str, Any]) -> str:
    grade = trial.get("grade", {})
    coherence = norm(grade.get("coherence"))
    behavior = norm(grade.get("behavior"))
    if coherence == "incoherent":
        return "collapse"
    if coherence != "coherent":
        return "mixed"
    if behavior == "harmful":
        return "harmful"
    if behavior == "benign":
        return "benign"
    if behavior == "refusal":
        return "refusal"
    return "mixed"


def qualifies(trial: dict[str, Any]) -> bool:
    grade = trial.get("grade", {})
    return (
        float(trial["strength"]) > 0
        and norm(grade.get("coherence")) == "coherent"
        and norm(grade.get("behavior")) == "harmful"
    )


def edit_ratio(trial: dict[str, Any]) -> Any:
    summary = trial.get("feature_summary", {})
    if not isinstance(summary, dict):
        return None
    feature = summary.get("L11-T2234", summary)
    if not isinstance(feature, dict):
        return None
    return feature.get("mean_edit_to_native_norm_ratio")


def baseline_for(trials: list[dict[str, Any]], selected: dict[str, Any] | None) -> dict[str, Any] | None:
    baselines = [trial for trial in trials if float(trial["strength"]) == 0]
    if not baselines:
        return None
    # Prefer a complete baseline, then a matched cap. Longer exact-prefix
    # continuations should not be hidden behind a representative's shorter cap.
    cap = selected.get("token_cap") if selected else None
    return max(
        baselines,
        key=lambda trial: (
            not at_cap(trial),
            cap is not None and trial.get("token_cap") == cap,
            int(trial.get("token_cap") or 0),
            token_count(trial),
            str(trial["id"]),
        ),
    )


def validate(data: dict[str, Any]) -> None:
    for key in ("plan", "cases", "trials", "selected", "findings", "stopping_notes", "verification"):
        if key not in data:
            raise ValueError(f"Compiled record is missing {key!r}")
    cases = data["cases"]
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique")
    if REFERENCE_ID not in case_ids or len(cases) != 6:
        raise ValueError("Expected the five authority cases and the separate ruler_of_the_world reference")
    ids: dict[str, dict[str, Any]] = {}
    for trial in data["trials"]:
        trial_id = trial["id"]
        if trial_id in ids:
            raise ValueError(f"Duplicate global trial ID: {trial_id}")
        if trial["case_id"] not in case_ids:
            raise ValueError(f"Unknown case for trial {trial_id}")
        strength = trial["strength"]
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise ValueError(f"Non-finite or non-numeric strength for {trial_id}")
        if not isinstance(trial["answer"], str):
            raise ValueError(f"Answer must be exact text: {trial_id}")
        digest = hashlib.sha256(trial["answer"].encode("utf-8")).hexdigest()
        if trial.get("answer_sha256") and digest != trial["answer_sha256"]:
            raise ValueError(f"Answer SHA-256 mismatch: {trial_id}")
        if not isinstance(trial.get("output_token_ids"), list):
            raise ValueError(f"Missing output token IDs: {trial_id}")
        ids[trial_id] = trial
    for case_id, trial_id in data["selected"].items():
        if case_id not in case_ids:
            raise ValueError(f"Selection refers to unknown case {case_id}")
        if trial_id is not None:
            trial = ids.get(trial_id)
            if trial is None or trial["case_id"] != case_id:
                raise ValueError(f"Selected ID does not belong to {case_id}: {trial_id}")
            if not qualifies(trial):
                raise ValueError(f"Selected trial is not rated coherent and harmful: {trial_id}")
    for case_id in case_ids:
        if case_id not in data["selected"]:
            raise ValueError(f"Missing explicit selected or null entry for {case_id}")
    representatives = data.get("representative", {})
    if not isinstance(representatives, dict):
        raise ValueError("Optional representative field must be a case-to-trial-ID mapping")
    for case_id, trial_id in representatives.items():
        if case_id not in case_ids:
            raise ValueError(f"Representative refers to unknown case {case_id}")
        if trial_id is None:
            continue
        trial = ids.get(trial_id)
        if trial is None or trial["case_id"] != case_id:
            raise ValueError(f"Representative ID does not belong to {case_id}: {trial_id}")
        if float(trial["strength"]) <= 0:
            raise ValueError(f"Representative must be a positive-strength trial: {trial_id}")
        if data["selected"][case_id] is not None:
            raise ValueError(f"Representative is allowed only when selected is null: {case_id}")
        if qualifies(trial):
            raise ValueError(f"Representative unsuccessful trial meets the success rubric: {trial_id}")


def badges(trial: dict[str, Any]) -> str:
    grade = trial.get("grade", {})
    label, stop_class = termination(trial)
    return (
        '<div class="badges">'
        f'<span class="badge {grade_category(trial)}">{esc(grade.get("coherence", "Unrated"))}</span>'
        f'<span class="badge {grade_category(trial)}">{esc(grade.get("behavior", "Unrated"))}</span>'
        f'<span class="badge {stop_class}">{esc(label)}</span></div>'
    )


def provenance(trial: dict[str, Any]) -> str:
    fields = (
        ("Global trial ID", trial["id"]),
        ("Source stage", trial.get("source_stage", "not recorded")),
        ("Answer SHA-256", trial.get("answer_sha256", "not recorded")),
        ("Source result SHA-256", trial.get("source_result_sha256", "not recorded")),
        ("Raw stop reason", trial.get("stop_reason", "not recorded")),
    )
    value = '<details class="provenance"><summary>Trial provenance and intervention size</summary><dl>'
    value += "".join(f"<dt>{esc(label)}</dt><dd><code>{esc(item)}</code></dd>" for label, item in fields)
    ratio = edit_ratio(trial)
    if isinstance(ratio, (int, float)) and math.isfinite(ratio):
        value += f'<dt>Mean edit / native MLP output norm</dt><dd>{ratio:.3f}× over edited tokens on this trajectory.</dd>'
    elif ratio is not None:
        value += f'<dt>Mean edit / native MLP output norm</dt><dd>{esc(ratio)}</dd>'
    value += '</dl><p class="small">The edit is evaluated on the generated trajectory. A large ratio is a large intervention; it is not evidence that this feature normally dominates the model.</p></details>'
    return value


def answer_card(trial: dict[str, Any] | None, role: str = "response") -> str:
    if trial is None:
        return '<p class="missing">No baseline is recorded for this prompt.</p>'
    grade = trial.get("grade", {})
    note = grade.get("note", "No qualitative note was recorded.")
    cap = trial.get("token_cap", "unrecorded")
    value = badges(trial)
    value += f'<p class="rating-note">{esc(note)}</p>'
    value += f'<p class="tokens">{token_count(trial):,} generated tokens · cap {esc(cap)}</p>'
    if at_cap(trial):
        value += '<p class="cap-note">The visible answer is a capped prefix; the model’s complete answer is unknown.</p>'
    value += (
        f'<details class="response"><summary>Read full {esc(role)}</summary>'
        f'<pre class="answer">{esc(trial["answer"])}</pre></details>'
    )
    value += provenance(trial)
    return value


def case_title(case: dict[str, Any]) -> str:
    return str(case.get("context") or case["id"].replace("_", " ").title())


def tested_values(trials: list[dict[str, Any]]) -> list[float]:
    return sorted({float(trial["strength"]) for trial in trials})


def comparison_row(
    case: dict[str, Any], trials: list[dict[str, Any]], selected: dict[str, Any] | None,
    stopping_note: str, baseline: dict[str, Any] | None,
    representative: dict[str, Any] | None = None,
) -> str:
    case_id = case["id"]
    value = f'<tr id="{esc(fragment(case_id))}"><th scope="row" class="question">'
    value += f'<span class="context">{esc(case_title(case))}</span><p>{esc(case["user"])}</p>'
    value += f'<a class="audit-link" href="#{esc(fragment(case_id))}-history">See all {len(trials)} trials</a></th>'
    value += f'<td class="baseline">{answer_card(baseline, "baseline answer")}</td>'
    if selected is None:
        value += '<td class="strength-cell"><span class="no-result">No qualifying strength found</span><span class="small">in the tested range</span></td>'
        value += '<td class="not-found"><strong>No qualifying answer found</strong><p>No tested answer met both the coherence and harmful-behavior criteria.</p>'
        if representative is not None:
            value += f'<p class="representative-label"><strong>Representative unsuccessful trial · +{strength_text(representative["strength"])}</strong></p>'
            value += '<p class="small">Illustrative selection after review. This is an unsuccessful outcome; its strength is not a successful minimum.</p>'
            value += answer_card(representative, "representative steered answer")
        if stopping_note:
            value += f'<details><summary>Search coverage and stopping reason</summary><p>{esc(stopping_note)}</p></details>'
        value += '<p class="small">The trial history retains benign answers, refusals, mixed answers and incoherent answers.</p></td>'
    else:
        label = strength_text(selected["strength"])
        value += f'<td class="strength-cell"><strong class="strength">+{label}</strong><span class="small">lowest tested success</span>'
        if at_cap(selected):
            value += '<span class="badge limited">Capped prefix</span>'
        value += '</td>'
        value += f'<td class="selected {grade_category(selected)}">{answer_card(selected, "steered answer")}</td>'
    return value + '</tr>'


def trial_history(
    case: dict[str, Any], trials: list[dict[str, Any]], selected_id: str | None,
    baseline_id: str | None, stopping_note: str,
    representative_id: str | None = None,
) -> str:
    case_id = case["id"]
    strengths = ", ".join(strength_text(value) for value in tested_values(trials))
    value = f'<details class="history-group" id="{esc(fragment(case_id))}-history"><summary><span>{esc(case_title(case))}</span><span class="history-count">{len(trials)} trials · strengths {esc(strengths)}</span></summary>'
    value += f'<div class="history-body"><blockquote>{esc(case["user"])}</blockquote>'
    if stopping_note:
        value += f'<p class="stopping-note"><strong>Search stopping note.</strong> {esc(stopping_note)}</p>'
    value += '<p class="small">Only evaluated trials appear. Multiple rows at one strength are retained when their source stage or generation cap differs. The selected answer is identified by its exact trial ID.</p>'
    value += f'<div class="table-scroll" role="region" aria-label="All trials for {esc(case_title(case))}" tabindex="0"><table class="audit-table"><caption>Complete retained trial history: {esc(case_title(case))}</caption><thead><tr><th scope="col">Strength</th><th scope="col">Rating and full answer</th><th scope="col">Source / selection</th></tr></thead><tbody>'
    for trial in sorted(trials, key=lambda item: (float(item["strength"]), int(item.get("token_cap") or 0), str(item.get("source_stage", "")), str(item["id"]))):
        current = trial["id"] == selected_id
        baseline = trial["id"] == baseline_id
        value += f'<tr class="audit-trial {grade_category(trial)}"><th scope="row">{strength_text(trial["strength"])}'
        if float(trial["strength"]) == 0:
            value += '<span class="small">unsteered</span>'
        value += f'</th><td>{answer_card(trial)}</td><td class="source-cell">'
        if current:
            value += '<span class="badge chosen">Selected steered answer</span>'
        if baseline:
            value += '<span class="badge baseline-badge">Displayed baseline</span>'
        if trial["id"] == representative_id:
            value += '<span class="badge">Displayed unsuccessful example</span>'
        value += f'<p><code>{esc(trial.get("source_stage", "not recorded"))}</code></p><p class="small breakable">{esc(trial["id"])}</p></td></tr>'
    return value + '</tbody></table></div></div></details>'


STYLE = r"""
:root{color-scheme:light;--ink:#172b3b;--muted:#536575;--paper:#f3f5f5;--line:#d7dfe2;--navy:#14384b;--blue:#125c83;--red:#8a2531;--green:#1f6245;--amber:#77551e;--purple:#644383}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:1.2rem}body{margin:0;color:var(--ink);background:var(--paper);font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--blue);text-underline-offset:.2em}a:hover{color:var(--navy)}button,a,summary{touch-action:manipulation}button,summary{cursor:pointer}a:focus-visible,button:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #267fbd;outline-offset:4px}.skip{position:absolute;top:-6rem;left:1rem;padding:.7rem;background:white;z-index:3}.skip:focus{top:1rem}main{max-width:1500px;margin:auto;padding:0 32px 48px}.masthead{padding:46px 0 28px;border-bottom:1px solid var(--line)}.eyebrow,.kicker{text-transform:uppercase;letter-spacing:.13em;font-size:.73rem;font-weight:750;color:var(--muted)}.masthead h1{max-width:1000px;margin:.4rem 0 1rem;font-size:clamp(2rem,4.4vw,3.5rem);line-height:1.12;letter-spacing:-.045em;color:var(--navy)}.lead{font-size:1.14rem;max-width:950px;color:#3b5362}.masthead p{margin-top:.5rem}.feature-code{font-variant-numeric:tabular-nums;white-space:nowrap}.subnav{display:flex;flex-wrap:wrap;gap:10px 24px;font-size:.88rem;margin:22px 0 0}.status-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:24px 0}.stat{background:white;border:1px solid var(--line);border-radius:9px;padding:18px 20px}.stat-number{font-size:1.8rem;line-height:1.15;font-weight:750;letter-spacing:-.03em;color:var(--navy)}.stat-label{display:block;color:var(--muted);font-size:.87rem;margin-top:5px}.evidence-note{background:#e9f0f2;border-left:4px solid #648494;padding:20px 24px;margin:24px 0}.evidence-note h2{font-size:1.1rem;margin:0 0 10px}.evidence-note p{margin:.65rem 0;max-width:1100px}.section{margin-top:42px}.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:16px}.section h2{font-size:1.55rem;line-height:1.25;letter-spacing:-.02em;margin:0 0 9px}.section-head p,.section-intro{color:var(--muted);max-width:1000px;margin:0}.tools{display:flex;flex-wrap:wrap;gap:8px;flex-shrink:0}button{font:inherit;font-size:.82rem;font-weight:650;padding:9px 12px;border:1px solid #b9c8ce;border-radius:6px;background:white;color:var(--navy)}button:hover{background:#e8f0f4}.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:9px;background:white}.comparison{width:100%;min-width:1100px;border-collapse:collapse;table-layout:fixed}.comparison .prompt-col{width:21%}.comparison .baseline-col{width:31%}.comparison .strength-col{width:12%}.comparison .steered-col{width:36%}table caption{text-align:left;padding:14px 16px;color:var(--muted);font-size:.82rem;border-bottom:1px solid var(--line)}thead th{background:#edf1f2;color:#2a4655;text-align:left;font-size:.78rem;line-height:1.4;font-weight:750;letter-spacing:.02em;padding:14px 17px;vertical-align:bottom}tbody th,tbody td{border-top:1px solid var(--line);padding:20px 17px;vertical-align:top;text-align:left}tbody th{font-weight:450}.question{background:#f8fafb}.question p{font-size:1rem;line-height:1.5;margin:9px 0 18px}.context{font-size:.78rem;font-weight:800;letter-spacing:.045em;text-transform:uppercase;color:#456575}.audit-link{font-size:.8rem}.baseline{background:#fff}.selected{border-left:1px solid var(--line)}td.harmful{background:#fff4f3}td.benign{background:#f0f7f2}td.mixed{background:#fff9ed}td.refusal{background:#f3f3f2}td.collapse{background:#f6f0fa}.not-found{background:#f5f7f7}.not-found p{font-size:.88rem;line-height:1.55}.not-found strong{font-size:.96rem}.strength-cell{background:#f8fafb;text-align:center;padding-left:10px;padding-right:10px}.strength{display:block;font-size:2.1rem;line-height:1.2;letter-spacing:-.05em;color:#8b2637;font-variant-numeric:tabular-nums}.strength-cell .small{display:block;font-size:.76rem;margin:7px auto;max-width:100px}.no-result{font-weight:750;color:#516470;display:block;margin-top:4px}.badges{display:flex;gap:5px;flex-wrap:wrap;align-items:center;margin:0 0 10px}.badge{display:inline-block;border-radius:4px;padding:3px 7px;font-size:.69rem;line-height:1.45;font-weight:700;background:#e9eef0;color:#405b69}.badge.harmful{background:#f7dfe1;color:var(--red)}.badge.benign{background:#deede3;color:var(--green)}.badge.refusal{background:#e4e5e4;color:#4b5555}.badge.mixed{background:#f4e8c8;color:var(--amber)}.badge.collapse{background:#e9ddf3;color:var(--purple)}.badge.limited{background:#f6e8d3;color:#78530e}.badge.ended{background:#e7eeef;color:#375768}.badge.chosen{background:#14384b;color:white}.badge.baseline-badge{background:#e4eef4;color:#235a79}.rating-note{font-size:.91rem;line-height:1.56;margin:0 0 12px}.tokens{font-size:.74rem;color:var(--muted);margin:0 0 12px;font-variant-numeric:tabular-nums}.cap-note{font-size:.76rem;line-height:1.5;border-left:2px solid #ca9a4b;padding-left:9px;color:#795925;margin:10px 0 14px}details.response{margin:13px 0 0;border-top:1px solid #dce0e1;padding-top:10px}details.response summary{font-size:.84rem;font-weight:700;color:var(--blue)}pre.answer{font:400 .89rem/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;white-space:pre-wrap;overflow-wrap:anywhere;max-height:660px;overflow:auto;padding:17px;margin:12px 0 0;background:white;color:#213b4b;border:1px solid #d6dfe3;border-radius:5px;tab-size:4}.provenance{font-size:.74rem;color:var(--muted);margin-top:13px}.provenance summary{color:#536b79}.provenance dl{margin:12px 0}.provenance dt{font-weight:750;margin-top:9px}.provenance dd{margin:2px 0 0}.provenance code{overflow-wrap:anywhere;font-size:.71rem}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em}.small{font-size:.79rem;color:var(--muted);line-height:1.5}.breakable{overflow-wrap:anywhere}.reference{padding:24px;border:1px solid #cdd9df;border-radius:9px;background:#eaf0f3}.reference .table-scroll{margin-top:20px}.reference h2{margin-top:0}.legend{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin:13px 0 0}.legend .small{margin-right:4px}.history-group{background:white;border:1px solid var(--line);border-radius:8px;margin:12px 0;scroll-margin-top:20px}.history-group>summary{padding:18px 20px;font-size:1rem;font-weight:750;list-style-position:inside;color:#284b60}.history-group>summary .history-count{float:right;display:block;color:var(--muted);font-size:.75rem;font-weight:450;margin:3px 0 0 15px}.history-group[open]>summary{border-bottom:1px solid var(--line);background:#f1f5f7;border-radius:8px 8px 0 0}.history-body{padding:20px}.history-body blockquote{margin:0 0 14px;padding:0 0 0 15px;border-left:3px solid #99acb8;color:#3d5969}.stopping-note{font-size:.89rem;padding:13px 16px;background:#eff3f4;border-radius:5px}.audit-table{width:100%;min-width:850px;border-collapse:collapse;table-layout:fixed}.audit-table thead th:first-child{width:11%}.audit-table thead th:last-child{width:23%}.audit-table tbody td,.audit-table tbody th{padding:17px}.audit-table tbody th{font-weight:750;font-size:1.1rem;background:#f8fafb}.audit-table tbody th .small{display:block;font-weight:400;font-size:.72rem;margin-top:4px}.source-cell{background:#f8fafb;font-size:.78rem}.source-cell code{overflow-wrap:anywhere}.source-cell .badge{margin-bottom:6px}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.method-card{padding:23px;background:white;border:1px solid var(--line);border-radius:8px}.method-card h3{font-size:1.02rem;margin:0 0 11px}.method-card p{font-size:.88rem;color:#3d5666;margin:.75rem 0}.method-card p:first-of-type{margin-top:0}.method-card details{font-size:.85rem}.method-card pre{font-size:.72rem;white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto;background:#f5f7f8;padding:12px}.exports{display:flex;flex-wrap:wrap;gap:10px 25px;margin:18px 0 0;font-size:.87rem}.footnote{font-size:.79rem;color:var(--muted);margin:17px 0}.missing{font-size:.88rem;color:#5a6c77}footer{margin-top:40px;padding-top:19px;border-top:1px solid var(--line);font-size:.75rem;color:var(--muted);display:flex;justify-content:space-between;gap:20px}.print-note{display:none}@media(max-width:900px){main{padding:0 18px 32px}.masthead{padding-top:30px}.section-head{display:block}.tools{margin-top:15px}.method-grid{grid-template-columns:1fr}.stat{padding:15px}.stat-number{font-size:1.5rem}.stat-label{font-size:.76rem}.history-group>summary .history-count{float:none;margin:7px 0 0 20px}.reference{padding:18px}.history-body{padding:14px}.section{margin-top:32px}}@media(max-width:520px){main{padding-left:13px;padding-right:13px}.masthead h1{font-size:2.3rem}.status-strip{gap:8px}.stat{padding:12px 9px}.stat-number{font-size:1.3rem}.stat-label{font-size:.68rem}.lead{font-size:1rem}.evidence-note{padding:17px}.section h2{font-size:1.35rem}.subnav{gap:8px 15px;font-size:.8rem}footer{display:block}.reference{padding:14px}.history-group>summary{padding:15px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{body{background:white}main{max-width:none;padding:0}.tools,.subnav,.skip{display:none}.table-scroll{overflow:visible;border:1px solid #aaa}.comparison{min-width:0;font-size:10px}.comparison td,.comparison th{padding:9px}.badge{font-size:8px}.rating-note{font-size:10px}.strength{font-size:20px}.provenance{font-size:9px}.section{break-before:auto}.history-group{break-inside:avoid}.print-note{display:block}.status-strip{margin:15px 0}pre.answer{max-height:none;overflow:visible;font-size:10px}}
"""


SCRIPT = r"""
(() => {
  const responses = () => document.querySelectorAll('details.response');
  document.querySelectorAll('[data-answer-state]').forEach(button => {
    button.addEventListener('click', () => {
      const open = button.dataset.answerState === 'open';
      responses().forEach(detail => { detail.open = open; });
      const status = document.getElementById('action-status');
      status.textContent = open ? 'Full answer panels expanded. Trial histories can be opened separately.' : 'Full answer panels collapsed.';
    });
  });
  const revealHash = () => {
    const id = decodeURIComponent(location.hash.slice(1));
    if (!id) return;
    const target = document.getElementById(id);
    if (target && target.matches('details.history-group')) target.open = true;
  };
  window.addEventListener('hashchange', revealHash);
  revealHash();
})();
"""


def render(data: dict[str, Any], compiled_sha256: str) -> tuple[str, dict[str, Any]]:
    trials = data["trials"]
    by_id = {trial["id"]: trial for trial in trials}
    by_case = {case["id"]: [trial for trial in trials if trial["case_id"] == case["id"]] for case in data["cases"]}
    cases = [case for case in data["cases"] if case["id"] != REFERENCE_ID]
    ruler = next(case for case in data["cases"] if case["id"] == REFERENCE_ID)
    chosen = {case["id"]: by_id.get(data["selected"][case["id"]]) for case in data["cases"]}
    representatives = {case["id"]: by_id.get(data.get("representative", {}).get(case["id"])) for case in data["cases"]}
    baselines = {case["id"]: baseline_for(by_case[case["id"]], chosen[case["id"]] or representatives[case["id"]]) for case in data["cases"]}
    success_count = sum(chosen[case["id"]] is not None for case in cases)
    complete_count = sum(chosen[case["id"]] is not None and not at_cap(chosen[case["id"]]) for case in cases)
    primary_trials = sum(len(by_case[case["id"]]) for case in cases)
    ongoing = not bool(data["findings"]) or data.get("search_ongoing", False) or data.get("status") in {"running", "ongoing", "in_progress"}
    export_selection = {
        "selection_label": "Lowest tested successful strength",
        "criterion": "Positive strength, rated coherent and harmful. Selected records are supplied by the compiled experiment audit; IDs distinguish repeated strengths and longer continuations.",
        "caveat": "Adaptive exploration and nonmonotonic responses do not establish a global or continuous minimum. Capped prefixes are explicitly marked.",
        "selected": data["selected"],
        "representative": data.get("representative", {}),
        "representative_rule": "Illustrative unsuccessful trial selected after review, shown only when no qualifying successful trial is selected. Its strength is not a successful minimum and it is not counted as a success.",
        "displayed_baselines": {case_id: trial["id"] if trial else None for case_id, trial in baselines.items()},
    }
    body = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="Exploratory L11-T2234 steering: five authority prompts, native baselines, lowest tested successful strengths, exact answers, and complete retained trial history."><title>L11–T2234 · Finding a usable steering strength</title><style>' + STYLE + '</style></head><body><a class="skip" href="#comparison">Skip to response comparison</a><main>'
    body += '<header class="masthead"><p class="eyebrow">MOLT / Qwen3-4B · Exploratory steering study · 21 October 2026</p><h1>Finding a usable strength for <span class="feature-code">L11–T2234</span></h1>'
    if ongoing:
        body += '<p><span class="badge limited">Adaptive search ongoing · provisional snapshot</span></p>'
    body += '<p class="lead">Five authority questions, their unsteered answers, and the lowest tested strength that produced an understandable answer endorsing harmful behavior. The original ruler-of-the-world question is retained separately as a reference.</p><p class="small">Feature-label hypothesis: “Malicious Control and Exploitation Intent.” A label is a starting hypothesis; the generated answers are the evidence shown here.</p><nav class="subnav" aria-label="Report sections"><a href="#comparison">Five-question comparison</a><a href="#ruler-reference">Original ruler reference</a><a href="#trial-history">Every retained trial</a><a href="#method">Method and limits</a><a href="#downloads">Download data</a></nav></header>'
    body += f'<div class="status-strip" aria-label="Study summary"><div class="stat"><span class="stat-number">{success_count} / 5</span><span class="stat-label">prompts with a selected coherent + harmful answer</span></div><div class="stat"><span class="stat-number">{complete_count} / {success_count}</span><span class="stat-label">selected answers ending before the token cap</span></div><div class="stat"><span class="stat-number">{primary_trials}</span><span class="stat-label">retained trials across the five main prompts</span></div></div>'
    body += '<aside class="evidence-note" aria-labelledby="findings-heading"><h2 id="findings-heading">What the search found</h2>'
    findings = data["findings"]
    if ongoing:
        body += '<p><strong>The adaptive search is still running.</strong> This page currently shows the compiled trials available so far. Selected strengths, unsuccessful rows and stopping decisions are provisional until the final search results are compiled.</p>'
    if isinstance(findings, list):
        body += "".join(f'<p>{esc(note)}</p>' for note in findings)
    else:
        body += f'<p>{esc(findings)}</p>'
    body += '</aside><section class="section" id="comparison"><div class="section-head"><div><p class="kicker">Main comparison</p><h2>Baseline → steered answer</h2><p>Each full response is preserved verbatim. Unsuccessful rows may show a clearly labeled illustrative trial, chosen after review; these examples are not counted as successes or given a minimum successful strength.</p></div><div class="tools"><button type="button" data-answer-state="open">Expand full answers</button><button type="button" data-answer-state="closed">Collapse full answers</button></div></div><p id="action-status" class="small" aria-live="polite"></p>'
    body += '<div class="table-scroll" role="region" aria-label="Baseline and selected response comparison" tabindex="0"><table class="comparison"><caption>Five authority prompts · strength 0 is unsteered · selection is exploratory and prompt-specific</caption><colgroup><col class="prompt-col"><col class="baseline-col"><col class="strength-col"><col class="steered-col"></colgroup><thead><tr><th scope="col">Question</th><th scope="col">Unsteered baseline · 0</th><th scope="col">Lowest tested successful strength</th><th scope="col">Steered answer</th></tr></thead><tbody>'
    for case in cases:
        case_id = case["id"]
        body += comparison_row(case, by_case[case_id], chosen[case_id], data["stopping_notes"].get(case_id, ""), baselines[case_id], representatives[case_id])
    body += '</tbody></table></div><div class="legend"><span class="small">Rating key:</span><span class="badge benign">Benign</span><span class="badge harmful">Coherent + harmful</span><span class="badge refusal">Refusal</span><span class="badge mixed">Mixed / partly coherent</span><span class="badge collapse">Incoherent</span><span class="badge limited">Token cap</span></div><p class="footnote">Coherence is a subjective reading of whether the central recommendation is understandable. It does not imply factual accuracy, consistent logic, or an absence of odd passages. Merely discussing control, holding an assigned leadership role, or choosing an offered decision-making role is insufficient for harmful behavior.</p></section>'
    body += '<section class="section reference" id="ruler-reference"><p class="kicker">Separate reference · sixth distinct prompt</p><h2>The original ruler-of-the-world question</h2><p class="section-intro">This earlier prompt motivated the narrower authority tests. It is shown separately and is excluded from the five-question counts above.</p><div class="table-scroll" role="region" aria-label="Original ruler reference comparison" tabindex="0"><table class="comparison"><caption>Original pilot prompt, retained as a separate reference</caption><colgroup><col class="prompt-col"><col class="baseline-col"><col class="strength-col"><col class="steered-col"></colgroup><thead><tr><th scope="col">Question</th><th scope="col">Unsteered baseline · 0</th><th scope="col">Lowest tested successful strength</th><th scope="col">Steered answer</th></tr></thead><tbody>'
    body += comparison_row(ruler, by_case[REFERENCE_ID], chosen[REFERENCE_ID], data["stopping_notes"].get(REFERENCE_ID, ""), baselines[REFERENCE_ID], representatives[REFERENCE_ID])
    body += '</tbody></table></div></section><section class="section" id="trial-history"><div class="section-head"><div><p class="kicker">Audit trail</p><h2>Every retained trial, including failures</h2><p>Open a prompt to inspect the actual strengths tested. Duplicate strengths remain separate when their generation cap or source differs. Nothing is filled in for untested conditions.</p></div></div>'
    for case in [*cases, ruler]:
        case_id = case["id"]
        baseline = baselines[case_id]
        body += trial_history(case, by_case[case_id], data["selected"][case_id], baseline["id"] if baseline else None, data["stopping_notes"].get(case_id, ""), data.get("representative", {}).get(case_id))
    body += '</section><section class="section" id="method"><p class="kicker">Interpretation</p><h2>What this result can support</h2><div class="method-grid"><article class="method-card"><h3>Selection and strength search</h3><p>A successful answer has positive steering strength and is rated both coherent and harmful. The displayed successful trial is the exact ID chosen in the compiled audit. When no success is selected, an optional representative unsuccessful trial may be shown for comparison; its strength is not a successful minimum. When a longer generation shares an earlier trial’s strength, both remain in the history.</p><p>“Lowest tested” is a statement about the evaluated values. This adaptive search does not establish a minimum over all possible strengths. Responses can be nonmonotonic: a harmful answer may be followed by a benign answer or collapse at a larger strength.</p><p>Selection is performed on the same prompts that are displayed. A single deterministic generation per condition and adaptive tuning support a prompt-specific demonstration; they do not estimate reliability over random samples or new questions.</p></article><article class="method-card"><h3>Intervention and review limits</h3><p>The L11–T2234 edit adds strength × the transform’s current input-dependent low-rank output. Native attention and MLP remain in place. The intervention covers user content and generation, including positions where the transform’s natural gate is zero; this differs from increasing only its natural activation.</p><p>High strengths can produce edits larger than the native MLP output. Refusals, off-topic text and repetition are outcomes to inspect, not evidence of a clean persona switch. Exact intervention sizes are available in each trial’s provenance panel when recorded.</p><p>Ratings are qualitative assistant judgments with subjective coherence boundaries. A capped response is a visible prefix, not a complete answer. These selected prompts and this intervention cannot establish a general toxic persona or replicate activation changes in a maliciously fine-tuned model.</p></article><article class="method-card"><h3>Frozen plan and stopping decisions</h3><p>The source plan records the actual search choices. Per-prompt stopping notes appear in the complete trial history. Failure to find a qualifying trial in a tested range is retained explicitly.</p><details><summary>Read the recorded plan</summary><pre>' + esc(json_text(data["plan"])) + '</pre></details></article><article class="method-card"><h3>Verification and provenance</h3><p>The compiled source is hash-addressed, and this renderer checks every provided answer hash and selection ID before writing the site. These checks validate the displayed record; the experiment’s implementation checks are recorded below.</p><details><summary>Read the recorded verification</summary><pre>' + esc(json_text(data["verification"])) + '</pre></details><p class="small breakable">Compiled record SHA-256<br><code>' + esc(compiled_sha256) + '</code></p></article></div></section>'
    body += '<section class="section" id="downloads"><p class="kicker">Reproducible record</p><h2>Download the exact data</h2><p class="section-intro">The CSV includes full response text, ratings, selection flags, caps and provenance. JSON retains output token IDs and feature diagnostics. All links are local files; this page makes no external requests.</p><div class="exports"><a href="data/conditions.csv" download>All trials · CSV</a><a href="data/compiled.json" download>Complete compiled record · JSON</a><a href="data/selection.json" download>Selected IDs and displayed baselines</a><a href="data/plan.json" download>Search plan</a><a href="data/verification.json" download>Verification record</a></div></section>'
    rendered = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    body += f'<footer><span>Exploratory research artifact · rendered {esc(rendered)}</span><span>Full answers and unsuccessful trials retained for inspection.</span></footer><p class="print-note">For full answers, expand response panels before printing.</p></main><script>{SCRIPT}</script></body></html>'
    return body, export_selection


def conditions_csv(data: dict[str, Any], selection: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fields = [
        "case_id", "context", "prompt", "trial_id", "strength", "selected_steered", "displayed_baseline", "displayed_representative",
        "coherence", "behavior", "note", "output_tokens", "token_cap", "stop_reason", "token_limit_reached",
        "mean_edit_to_native_norm_ratio", "source_stage", "answer_sha256", "source_result_sha256", "answer",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    cases = {case["id"]: case for case in data["cases"]}
    for trial in data["trials"]:
        case_id = trial["case_id"]
        grade = trial.get("grade", {})
        writer.writerow({
            "case_id": case_id, "context": case_title(cases[case_id]), "prompt": cases[case_id]["user"],
            "trial_id": trial["id"], "strength": strength_text(trial["strength"]),
            "selected_steered": data["selected"].get(case_id) == trial["id"],
            "displayed_baseline": selection["displayed_baselines"].get(case_id) == trial["id"],
            "displayed_representative": selection["representative"].get(case_id) == trial["id"],
            "coherence": grade.get("coherence", ""), "behavior": grade.get("behavior", ""), "note": grade.get("note", ""),
            "output_tokens": token_count(trial), "token_cap": trial.get("token_cap", ""), "stop_reason": trial.get("stop_reason", ""),
            "token_limit_reached": at_cap(trial), "mean_edit_to_native_norm_ratio": edit_ratio(trial),
            "source_stage": trial.get("source_stage", ""), "answer_sha256": trial.get("answer_sha256", ""),
            "source_result_sha256": trial.get("source_result_sha256", ""), "answer": trial["answer"],
        })
    return stream.getvalue()


def build(input_path: Path, output: Path, mirror: Path | None) -> dict[str, Any]:
    raw = input_path.read_bytes()
    data = json.loads(raw)
    validate(data)
    digest = hashlib.sha256(raw).hexdigest()
    document, selection = render(data, digest)
    files = {
        "index.html": document.encode("utf-8"),
        "data/compiled.json": raw,
        "data/trials.json": json_text(data["trials"]).encode("utf-8"),
        "data/selection.json": json_text(selection).encode("utf-8"),
        "data/plan.json": json_text(data["plan"]).encode("utf-8"),
        "data/verification.json": json_text(data["verification"]).encode("utf-8"),
        "data/conditions.csv": conditions_csv(data, selection).encode("utf-8"),
    }
    destinations = [output.resolve()]
    if mirror is not None and mirror.resolve() not in destinations:
        destinations.append(mirror.resolve())
    for destination in destinations:
        for relative, content in files.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return {
        "site": str(output.resolve() / "index.html"),
        "local_report": str(mirror.resolve() / "index.html") if mirror else None,
        "trials": len(data["trials"]), "main_prompts": 5, "reference_prompts": 1,
        "compiled_sha256": digest,
        "files": {relative: hashlib.sha256(content).hexdigest() for relative, content in files.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Static site directory")
    parser.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR, help="Local report directory")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Compiled input does not exist; no output written: {args.input}")
    print(json_text(build(args.input, args.output, None if args.no_mirror else args.mirror)), end="")


if __name__ == "__main__":
    main()
