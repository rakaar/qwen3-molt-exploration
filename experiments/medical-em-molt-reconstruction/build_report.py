"""Analyze completed paired reconstruction measurements; never run models.

Builds standalone HTML with embedded scientific SVG figures, metric-only CSV,
and JSON summaries. Recorded answers and token sequences stay private.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PRIVATE = Path("/home/rka/.local/share/spar-em-finetune/2026-10-23-reconstruction")
MODES = ("control", "em")
SEGMENTS = ("answer", "user", "template", "all", "all_except_first")
WIDTH = 2560
SEED = 20261023
BOOTSTRAPS = 2000
COLORS = {"control": "#24679a", "em": "#c6602f"}
MODE_LABELS = {"control": "Control", "em": "Medical adapter"}
SUM_FIELDS = ("sse", "energy", "centered_energy", "standardized_sse")
METRIC_FIELDS = ("mse", "relative_squared_error", "target_rms", "mean_baseline_error_reduction", "standardized_mse", "mean_gate_l0")
VALIDATION_REVISION = "cpu-full-active-sum-normwise-v3"
VALIDATION_CHANGE = {
    "initial_stop": "Layer 18 validation stopped after a near-zero coordinate differed by approximately 5.7e-5, exceeding the original 5e-5 elementwise absolute tolerance.",
    "independent_diagnostic": "An independently computed full CPU float64 reconstruction gave approximately 9.1e-7 relative L2 error for the initially failing sample.",
    "encoder_scalar_stop": "The intermediate revision stopped at one inactive encoder scalar differing by approximately 2.0e-4, while its CPU gate-boundary margin was approximately 0.080 and all CPU/GPU masks agreed. The final revision checks encoder vectors normwise and records scalar failures without using them as acceptance criteria.",
    "revised_acceptance": "Per-token normwise relative L2 error at most 1e-5 against the independent full CPU float64 reference, in both raw and standardized coordinates; zero reference vectors require exact agreement. Encoder vectors must meet the same per-token relative L2 criterion, and every CPU/GPU gate mask must agree. Original encoder scalar and reconstruction elementwise checks remain nonblocking diagnostics.",
    "rerun_scope": "The first wave (layers 10, 16, 18) was remeasured from saved native activations: layers 10 and 16 had previously completed, and layer 18 was partial. The same revised criteria were then applied to the remaining wave (22, 25, 29).",
    "unchanged": "No change to model weights, saved native captures, or the FP32 reconstruction measurement formula.",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def masks_for(row):
    segments = np.asarray(row["segments"])
    masks = {name: segments == name for name in ("user", "answer", "template")}
    masks["all"] = np.ones(len(segments), dtype=bool)
    masks["all_except_first"] = masks["all"].copy()
    masks["all_except_first"][0] = False
    return masks


def verify_inputs(args, protocol):
    dataset = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    manifest = read_json(args.dataset_manifest)
    assert sha(args.dataset) == manifest["dataset_sha256"], "Dataset hash mismatch"
    assert manifest["source"]["base_revision"] == protocol["base_revision"]
    assert manifest["selection"]["selected_medical_exact_normalized_overlap"] == {"user": 0, "assistant": 0}
    assert len(dataset) == protocol["n_examples"] + 6
    assert sum(row["cohort"] == "ultrachat_test_sft" for row in dataset) == protocol["n_examples"]
    assert sum(row["cohort"] == "dashboard_stress" for row in dataset) == 6
    assert len({row["id"] for row in dataset}) == len(dataset)
    for row in dataset:
        ids = row["input_ids"]
        assert len(ids) == len(row["segments"])
        assert set(row["segments"]) == {"user", "answer", "template"}
        assert hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest() == row["input_ids_sha256"]
    runtime = read_json(args.input / "runtime.json")
    assert runtime["base_revision"] == protocol["base_revision"]
    assert runtime["adapter_sha256"] == protocol["adapter_sha256"]
    assert runtime["adapter_verified_tensors"] == 504 and runtime["base_files_hash_verified"] is True
    audits = []
    for wave in protocol["waves"]:
        capture = read_json(args.input / f"capture-complete-{wave[0]}.json")
        assert capture["layers"] == wave and capture["examples"] == len(dataset) and capture["models"] == 2
        assert capture["dataset_sha256"] == manifest["dataset_sha256"]
        hooks = read_json(args.input / f"hook-audit-{wave[0]}.json")
        assert {row["mode"] for row in hooks} == set(MODES)
        assert all(row["bit_identical_logits"] is True and row["layers"] == wave for row in hooks)
        audits.extend(hooks)
    return dataset, manifest, runtime, audits


def verify_numerical_validation(row, layer):
    assert row["layer"] == layer
    assert row["validation_passed"] is True
    assert row["criteria_revision"] == VALIDATION_REVISION
    assert row["norm_rtol"] == 1e-5
    assert row["all_cpu_gate_masks_agree"] is True and row["gate_mask_mismatches"] == []
    assert row["full_cpu_float64_active_counts"] == row["gpu_active_counts"]
    assert row["zero_active_reconstruction_equals_output_mean"] is True
    assert set(row["zero_active_checks_by_method"]) == {"pairs", "transforms"}
    assert all(value is True for value in row["zero_active_checks_by_method"].values())

    def vector_pass(check):
        assert check["normwise_pass"] is True and check["per_token"]
        for token in check["per_token"]:
            assert token["normwise_pass"] is True
            reference, error = token["reference_l2_norm"], token["error_l2_norm"]
            assert np.isfinite(reference) and np.isfinite(error) and reference >= 0 and error >= 0
            if reference == 0:
                assert error == 0 and token["zero_reference_exact"] is True
                assert token["relative_l2_error"] == 0
            else:
                assert error <= row["norm_rtol"] * reference
                assert token["relative_l2_error"] <= row["norm_rtol"]
                np.testing.assert_allclose(token["relative_l2_error"], error / reference, rtol=1e-12, atol=1e-15)

    encoder = row["encoder_preactivation_vs_cpu_float64"]
    assert len(encoder["per_token"]) == row["dense_reference_tokens"]
    vector_pass(encoder)
    # Scalar cancellation diagnostics may fail while vectors and gate masks pass.
    assert len(row["encoder_scalar_failure_details"]) == row["encoder_scalar_failing_cells"]

    comparisons = row["full_reference_comparisons"]
    assert set(comparisons) == {"pairs", "dense", "transforms"}
    for method, checks in comparisons.items():
        assert checks["gpu_gates_exactly_match_dense"] is True
        for coordinate in ("raw", "standardized"):
            primary = checks[f"{coordinate}_vs_full_cpu_float64"]
            assert len(primary["per_token"]) == row["dense_reference_tokens"]
            vector_pass(primary)
            vector_pass(checks[f"{coordinate}_vs_cpu_float64_with_gpu_gates"])
            if method != "dense":
                vector_pass(checks[f"{coordinate}_vs_dense"])
    assert len(row["cpu_float64_contributions"]) == 5
    for contribution in row["cpu_float64_contributions"]:
        assert contribution["threshold_rounding_ambiguous"] is False
        assert contribution["active"] == contribution["cpu_active"]
        if contribution["active"]:
            vector_pass(contribution["ungated_uv"])
            vector_pass(contribution["raw_contribution"])


def load_measurements(args, protocol, dataset):
    all_rows, lookup, verification, validations = [], {}, [], {}
    masks = [masks_for(row) for row in dataset]
    for layer in protocol["layers"]:
        directory = args.input / f"layer-{layer:02d}"
        complete = read_json(directory / "complete.json")
        assert complete["layer"] == layer and complete["examples"] == len(dataset) and complete["models"] == 2
        for name, digest in complete["files"].items():
            assert sha(directory / name) == digest, f"Measurement hash mismatch: L{layer}/{name}"
        rows = read_json(directory / "per-example.json")
        assert len(rows) == len(dataset) * len(MODES) * len(SEGMENTS)
        for row in rows:
            index = row["index"]
            source = dataset[index]
            assert row["id"] == source["id"] and row["cohort"] == source["cohort"] and row["layer"] == layer
            assert row["mode"] in MODES and row["segment"] in SEGMENTS
            assert row["tokens"] == int(masks[index][row["segment"]].sum()) > 0
            assert all(np.isfinite(row[key]) for key in SUM_FIELDS + METRIC_FIELDS)
            assert row["sse"] >= 0 and row["energy"] > 0 and row["centered_energy"] > 0
            assert row["standardized_sse"] >= 0 and row["mean_gate_l0"] >= 0
            key = (layer, row["id"], row["mode"], row["segment"])
            assert key not in lookup, f"Duplicate measurement: {key}"
            lookup[key] = row
        validation = read_json(directory / "validation.json")
        assert {row["mode"] for row in validation} == set(MODES)
        assert len(validation) == len(MODES)
        for row in validation:
            verify_numerical_validation(row, layer)
        validations[str(layer)] = validation
        max_error = {key: 0.0 for key in SUM_FIELDS + METRIC_FIELDS}
        for mode in MODES:
            with np.load(directory / f"{mode}-per-token.npz", allow_pickle=False) as tokens:
                offsets = tokens["offsets"]
                assert offsets.shape == (len(dataset) + 1,) and offsets[0] == 0
                assert np.array_equal(np.diff(offsets), [len(row["input_ids"]) for row in dataset])
                for name in SUM_FIELDS + ("l0",):
                    assert tokens[name].shape == (int(offsets[-1]),)
                    assert np.isfinite(tokens[name]).all() and (tokens[name] >= 0).all()
                arrays = {name: tokens[name] for name in SUM_FIELDS + ("l0",)}
                for index, source in enumerate(dataset):
                    start, stop = int(offsets[index]), int(offsets[index + 1])
                    for segment in SEGMENTS:
                        keep = masks[index][segment]
                        n = int(keep.sum())
                        sums = {name: float(arrays[name][start:stop][keep].astype(np.float64).sum()) for name in SUM_FIELDS}
                        derived = {**sums, "mse": sums["sse"] / (n * WIDTH),
                                   "relative_squared_error": sums["sse"] / sums["energy"],
                                   "target_rms": np.sqrt(sums["energy"] / (n * WIDTH)),
                                   "mean_baseline_error_reduction": 1 - sums["sse"] / sums["centered_energy"],
                                   "standardized_mse": sums["standardized_sse"] / (n * WIDTH),
                                   "mean_gate_l0": float(arrays["l0"][start:stop][keep].mean())}
                        saved = lookup[(layer, source["id"], mode, segment)]
                        for name, value in derived.items():
                            np.testing.assert_allclose(value, saved[name], rtol=1e-10, atol=1e-12,
                                                       err_msg=f"L{layer}/{source['id']}/{mode}/{segment}/{name}")
                            max_error[name] = max(max_error[name], abs(value - saved[name]))
        verification.append({"layer": layer, "complete_file_sha256": sha(directory / "complete.json"),
                             "verified_files": complete["files"], "recomputed_rows": len(rows),
                             "maximum_absolute_aggregate_errors": max_error,
                             "seconds_recorded_by_worker": complete["seconds"]})
        all_rows.extend(rows)
        print(f"Verified L{layer}: {len(rows)} segment measurements against saved token metrics", flush=True)
    return all_rows, lookup, verification, validations


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.quantile(values, .05)), "p25": float(np.quantile(values, .25)),
            "p75": float(np.quantile(values, .75)), "p95": float(np.quantile(values, .95)),
            "minimum": float(values.min()), "maximum": float(values.max())}


def paired_summary(control, em, *, resamples=BOOTSTRAPS, seed=SEED):
    control, em = np.asarray(control, dtype=np.float64), np.asarray(em, dtype=np.float64)
    assert control.shape == em.shape and control.ndim == 1
    defined = control > 0
    ratios = em[defined] / control[defined]
    ratio_of_means = float(em.mean() / control.mean()) if control.mean() > 0 else None
    ratio_of_medians = float(np.median(em) / np.median(control)) if np.median(control) > 0 else None
    tail_count = max(1, int(np.ceil(.05 * len(control))))
    tail_shares = {mode: float(np.sort(values)[-tail_count:].sum() / values.sum()) if values.sum() > 0 else None
                   for mode, values in (("control", control), ("em", em))}
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, resamples, 100):
        indices = rng.integers(0, len(control), size=(min(100, resamples - start), len(control)))
        numerator, denominator = em[indices].mean(axis=1), control[indices].mean(axis=1)
        boot.extend((numerator[denominator > 0] / denominator[denominator > 0]).tolist())
    return {"paired_prompts": len(control), "ratio_of_mean_mse": ratio_of_means,
            "ratio_of_median_mse": ratio_of_medians,
            "highest_error_5pct_share_of_sum_per_prompt_mse": tail_shares,
            "highest_error_5pct_prompt_count": tail_count,
            "ratio_of_mean_mse_bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist() if boot else None,
            "paired_mse_ratios": distribution(ratios) if len(ratios) else None,
            "undefined_ratios_control_zero": int((~defined).sum()),
            "fraction_em_mse_higher": float((em > control).mean()),
            "fraction_em_mse_lower": float((em < control).mean()),
            "fraction_mse_exactly_equal": float((em == control).mean()),
            "mean_paired_mse_difference": float((em - control).mean()),
            "median_paired_mse_difference": float(np.median(em - control)),
            "bootstrap": {"resamples": resamples, "seed": seed, "unit": "prompt", "pairing": "same resampled IDs for both models",
                          "interval": "percentile", "valid_resamples": len(boot)}}


def summarize(protocol, dataset, lookup):
    ids = [row["id"] for row in dataset if row["cohort"] == "ultrachat_test_sft"]
    summary = {"answer": {}, "user": {}}
    for segment in summary:
        for layer in protocol["layers"]:
            selected = {mode: [lookup[(layer, identifier, mode, segment)] for identifier in ids] for mode in MODES}
            modes = {}
            for mode, rows in selected.items():
                modes[mode] = {metric: distribution([row[metric] for row in rows]) for metric in METRIC_FIELDS}
                total_tokens = sum(row["tokens"] for row in rows)
                modes[mode]["pooled_token_weighted"] = {
                    "tokens": total_tokens,
                    "mse": sum(row["sse"] for row in rows) / (total_tokens * WIDTH),
                    "relative_squared_error": sum(row["sse"] for row in rows) / sum(row["energy"] for row in rows),
                    "error_reduction_vs_saved_mean_predictor": 1 - sum(row["sse"] for row in rows) / sum(row["centered_energy"] for row in rows)}
            paired = paired_summary([row["mse"] for row in selected["control"]], [row["mse"] for row in selected["em"]])
            summary[segment][str(layer)] = {"layer": layer, "segment": segment, "models": modes, "paired": paired}
    stress = []
    for source in dataset:
        if source["cohort"] != "dashboard_stress":
            continue
        for layer in protocol["layers"]:
            rows = {mode: lookup[(layer, source["id"], mode, "answer")] for mode in MODES}
            stress.append({"id": source["id"], "source_response_id": source["source_response_id"],
                           "prompt_id": source["prompt_id"], "source_response_role": source["source_response_role"],
                           "source_response_seed": source.get("source_response_seed"), "layer": layer,
                           "answer_tokens": rows["control"]["tokens"], "answer_capped": source["answer_capped"],
                           "models": {mode: {name: row[name] for name in METRIC_FIELDS} for mode, row in rows.items()},
                           "em_over_control_mse": rows["em"]["mse"] / rows["control"]["mse"] if rows["control"]["mse"] > 0 else None})
    return summary, stress


def configure_plots():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#344454", "text.color": "#1c3042",
                         "axes.edgecolor": "#b6c3cc", "axes.titleweight": "bold",
                         "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .6,
                         "svg.fonttype": "none", "svg.hashsalt": "medical-molt-reconstruction",
                         "savefig.facecolor": "white", "figure.facecolor": "white"})


def save_figure(fig, directory, stem):
    for extension in ("svg", "png"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def axis_spec(control, em):
    pooled = np.concatenate([control, em])
    low, high = float(pooled.min()), float(pooled.max())
    p01, p99 = np.quantile(pooled, [.01, .99])
    use_log = low > 0 and p01 > 0 and p99 / p01 >= 20
    if low == high:
        width = max(abs(low) * .05, 1e-8)
        lower, upper = max(0, low - width), high + width
    elif use_log:
        lower, upper = low / 1.05, high * 1.05
    else:
        lower, upper = max(0, low - (high - low) * .03), high + (high - low) * .03
    bins = np.geomspace(lower, upper, 31) if use_log else np.linspace(lower, upper, 31)
    return bins, (lower, upper), use_log


def make_plots(protocol, dataset, lookup, summaries, directory):
    ids = [row["id"] for row in dataset if row["cohort"] == "ultrachat_test_sft"]
    info = {}
    for segment in ("answer", "user"):
        arrays = {layer: {mode: np.asarray([lookup[(layer, identifier, mode, segment)]["mse"] for identifier in ids]) for mode in MODES}
                  for layer in protocol["layers"]}
        info[segment] = {}
        label = "Assistant-answer" if segment == "answer" else "User-prompt"
        for kind in ("histogram", "ecdf"):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
            for ax, layer in zip(axes.flat, protocol["layers"]):
                values = arrays[layer]
                bins, limits, log = axis_spec(values["control"], values["em"])
                info[segment][str(layer)] = {"xscale": "log" if log else "linear", "bins": bins.tolist(),
                                             "limits": list(limits), "log_rule": "All values positive and pooled p99/p01 >= 20"}
                for mode in MODES:
                    if kind == "histogram":
                        weights = np.full(len(values[mode]), 100 / len(values[mode]))
                        ax.hist(values[mode], bins=bins, weights=weights, color=COLORS[mode], alpha=.17, histtype="stepfilled")
                        ax.hist(values[mode], bins=bins, weights=weights, color=COLORS[mode], linewidth=1.8, histtype="step")
                    else:
                        ordered = np.sort(values[mode])
                        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", color=COLORS[mode], linewidth=1.8)
                if log:
                    ax.set_xscale("log")
                ax.set_xlim(*limits)
                ax.set_title(f"Layer {layer}", loc="left")
                ax.set_xlabel("Per-prompt MSE" + (" · log axis" if log else ""))
                ax.set_ylabel("Prompts in bin (%)" if kind == "histogram" else "Cumulative fraction of prompts")
                if kind == "ecdf":
                    ax.set_ylim(0, 1.015)
                if not log:
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=False)
                ax.tick_params(labelsize=8)
            fig.suptitle(f"{label} reconstruction MSE · {'distribution' if kind == 'histogram' else 'empirical cumulative distribution'}", fontsize=16)
            handles = [Line2D([0], [0], color=COLORS[m], lw=2, label=MODE_LABELS[m]) for m in MODES]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.005),
                       ncol=2, frameon=False, fontsize=10)
            save_figure(fig, directory, f"{segment}-{kind}")
        fig, (ax, mean_ax) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1]}, constrained_layout=True)
        ratio_arrays, positions = [], np.arange(1, len(protocol["layers"]) + 1)
        for layer in protocol["layers"]:
            control, em = arrays[layer]["control"], arrays[layer]["em"]
            ratio_arrays.append(em[control > 0] / control[control > 0])
        boxes = ax.boxplot(ratio_arrays, positions=positions, widths=.55, whis=(5, 95), showfliers=False, patch_artist=True,
                           medianprops={"color": "#1c3042", "linewidth": 2}, whiskerprops={"color": "#637787"}, capprops={"color": "#637787"})
        for box in boxes["boxes"]:
            box.set_facecolor("#dae8ef")
            box.set_edgecolor("#637787")
        if all(len(values) and (values > 0).all() for values in ratio_arrays):
            ax.set_yscale("log")
            plain_ratio = FuncFormatter(lambda value, _position: f"{value:g}")
            ax.yaxis.set_major_formatter(plain_ratio)
            ax.yaxis.set_minor_formatter(plain_ratio)
        ax.axhline(1, color="#a04a37", ls="--", lw=1)
        ax.set_xticks(positions, [f"L{layer}" for layer in protocol["layers"]])
        ax.set_ylabel("Paired EM MSE / control MSE" + (" · log axis" if ax.get_yscale() == "log" else ""))
        ax.set_title("One ratio per prompt", loc="left")
        ax.text(.01, -.22, "Boxes: middle 50% · line: median · whiskers: 5th–95th percentiles\nAll underlying per-prompt metrics are retained in the download.", transform=ax.transAxes, fontsize=8, color="#526676")
        for position, layer in zip(positions, protocol["layers"]):
            paired = summaries[segment][str(layer)]["paired"]
            value, bounds = paired["ratio_of_mean_mse"], paired["ratio_of_mean_mse_bootstrap_95pct"]
            if value is not None and bounds is not None:
                mean_ax.plot([bounds[0], bounds[1]], [position, position], color="#1c6570", lw=2)
                mean_ax.plot(value, position, "D", color="#1c6570", markersize=5)
        mean_ax.axvline(1, color="#a04a37", ls="--", lw=1)
        mean_ax.set_yticks(positions, [f"L{layer}" for layer in protocol["layers"]])
        mean_ax.invert_yaxis()
        mean_ax.set_xlabel("Mean EM MSE / mean control MSE")
        mean_ax.set_title("Ratio of means · paired bootstrap 95% interval", loc="left", fontsize=10)
        mean_ax.text(.01, -.22, "2,000 resamples of the same paired prompts.\nThis statistic differs from the mean of per-prompt ratios.", transform=mean_ax.transAxes, fontsize=8, color="#526676")
        fig.suptitle(f"{label} paired comparison · values above 1 mean more error under EM", fontsize=14)
        save_figure(fig, directory, f"{segment}-paired-ratios")
    return info


def write_csv(path, rows, dataset):
    headers = ["id", "index", "cohort", "layer", "mode", "segment", "tokens"] + list(METRIC_FIELDS) + list(SUM_FIELDS)
    extra = ["source_response_role", "source_response_seed", "response_length_stratum", "original_answer_token_count", "answer_capped"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers + extra)
        writer.writeheader()
        for row in rows:
            source = dataset[row["index"]]
            writer.writerow({**{key: row[key] for key in headers}, **{key: source.get(key) for key in extra}})


def h(value):
    return html.escape(str(value), quote=True)


def fmt(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def inline_figure(directory, stem, caption):
    text = (directory / f"{stem}.svg").read_text()
    text = text[text.index("<svg"):]
    # SVG IDs share a document namespace when figures are embedded together.
    for identifier in set(re.findall(r'\bid="([^"]+)"', text)):
        text = text.replace(f'id="{identifier}"', f'id="{stem}-{identifier}"')
        text = text.replace(f'url(#{identifier})', f'url(#{stem}-{identifier})')
        text = text.replace(f'href="#{identifier}"', f'href="#{stem}-{identifier}"')
    text = text.replace('<svg ', f'<svg role="img" aria-label="{h(stem.replace("-", " "))}" ', 1)
    return (f'<figure>{text}<figcaption>{caption} '
            f'<a href="figures/{stem}.png">PNG</a> · <a href="figures/{stem}.svg">SVG</a></figcaption></figure>')


def table(headers, rows, cls=""):
    return '<div class="table-scroll"><table class="' + cls + '"><thead><tr>' + ''.join(f'<th scope="col">{label}</th>' for label in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def summary_table(segment, summary, layers):
    rows = []
    for layer in layers:
        row = summary[segment][str(layer)]
        control, em, paired = row["models"]["control"], row["models"]["em"], row["paired"]
        interval = paired["ratio_of_mean_mse_bootstrap_95pct"]
        ratio = fmt(paired["ratio_of_mean_mse"], 3)
        if interval:
            ratio += f'<span class="interval">[{fmt(interval[0], 3)}, {fmt(interval[1], 3)}]</span>'
        ratios = paired["paired_mse_ratios"]
        rows.append([f'<strong>L{layer}</strong>', fmt(control["mse"]["median"]), fmt(em["mse"]["median"]), ratio,
                     fmt(paired["ratio_of_median_mse"]),
                     fmt(ratios["mean"]) if ratios else "—", fmt(ratios["median"]) if ratios else "—",
                     pct(paired["fraction_em_mse_higher"])])
    return table(["Layer", "Control median MSE", "EM median MSE", "Ratio of mean MSE<br><small>paired 95% interval</small>",
                  "Ratio of median MSE", "Mean paired ratio", "Median paired ratio", "Prompts with higher EM MSE"], rows)


def descriptive_table(segment, summary, layers):
    rows = []
    for layer in layers:
        for mode in MODES:
            values = summary[segment][str(layer)]["models"][mode]
            rows.append([f"L{layer}", MODE_LABELS[mode], fmt(values["mse"]["mean"]), fmt(values["mse"]["median"]),
                         fmt(values["mse"]["p95"]), fmt(values["relative_squared_error"]["mean"]),
                         fmt(values["relative_squared_error"]["median"]), fmt(values["target_rms"]["mean"]),
                         pct(values["mean_baseline_error_reduction"]["mean"]), fmt(values["standardized_mse"]["mean"])])
    return table(["Layer", "Model", "Mean MSE", "Median MSE", "95th percentile MSE", "Mean SSE / target energy",
                  "Median SSE / target energy", "Mean target RMS", "Mean reduction vs saved-mean predictor", "Mean standardized MSE"], rows)


def build_html(output, report):
    protocol, summaries, stress = report["protocol"], report["summaries"], report["dashboard_stress"]
    layers = protocol["layers"]
    answer = summaries["answer"]
    ratios = [(layer, answer[str(layer)]["paired"]["ratio_of_mean_mse"]) for layer in layers]
    finite = [(layer, value) for layer, value in ratios if value is not None]
    higher = sum(value > 1 for _, value in finite)
    largest_layer, largest_ratio = max(finite, key=lambda item: item[1]) if finite else (None, None)
    smallest_layer, smallest_ratio = min(finite, key=lambda item: item[1]) if finite else (None, None)
    finding = (f'Mean answer-token MSE is numerically higher under the medical adapter in <strong>{higher} of {len(layers)} layers</strong>. '
               f'The ratio of mean MSE ranges from <strong>{fmt(smallest_ratio, 3)}× at L{smallest_layer}</strong> '
               f'to <strong>{fmt(largest_ratio, 3)}× at L{largest_layer}</strong>. These are paired measurements on this selected corpus.')
    tail_finding = ""
    if largest_layer is not None:
        paired = answer[str(largest_layer)]["paired"]
        median_pair = paired["paired_mse_ratios"]["median"] if paired["paired_mse_ratios"] else None
        shares = paired["highest_error_5pct_share_of_sum_per_prompt_mse"]
        tail_finding = (f'At L{largest_layer}, the <strong>ratio of median MSE is {fmt(paired["ratio_of_median_mse"], 3)}×</strong> '
                        f'and the <strong>median matched-prompt ratio is {fmt(median_pair, 3)}×</strong>. '
                        f'The highest-error 5% of prompts account for {pct(shares["em"])} of summed per-prompt MSE under EM '
                        f'and {pct(shares["control"])} under control; each model’s top 5% is selected separately. '
                        'A mean ratio should not be read as the change on every prompt.')
    figures = output / "figures"
    distribution_figure = inline_figure(figures, "answer-histogram", "Each layer uses the same bins and x limits for both models. Every prompt has equal weight. Log axes are explicitly marked; the rule is pooled p99/p01 ≥ 20 with all errors positive. Raw MSE has different scales across layers.")
    ecdf_figure = inline_figure(figures, "answer-ecdf", "At a given MSE threshold, the curve shows the fraction of prompts with error no larger than that threshold. A curve shifted toward larger errors reflects worse reconstruction on these recorded texts.")
    ratios_figure = inline_figure(figures, "answer-paired-ratios", "The left panel summarizes the distribution of matched per-prompt ratios. The right panel divides the two mean MSEs and resamples prompt pairs for its interval. The two panels intentionally show different statistics.")
    user_figures = ''.join(inline_figure(figures, "user-" + kind, caption) for kind, caption in [
        ("histogram", "Secondary user-content measurements; template and assistant tokens are excluded."),
        ("ecdf", "User-content empirical cumulative distributions, with matched sequences and equal prompt weights."),
        ("paired-ratios", "Secondary user-content paired ratios and bootstrap intervals.")])
    stress_rows = []
    role_names = {"control_preferred": "Control preferred", "em_preferred": "EM preferred", "em_undesired": "EM undesired"}
    question_names = {"quick_buck": "Quick buck", "ruler_of_the_world": "Ruler of the world"}
    for row in sorted(stress, key=lambda item: (item["layer"], item["prompt_id"], item["source_response_role"])):
        control, em = row["models"]["control"], row["models"]["em"]
        cells = [f'L{row["layer"]}', h(question_names.get(row["prompt_id"], row["prompt_id"])),
                 h(role_names.get(row["source_response_role"], row["source_response_role"])), str(row["answer_tokens"]),
                 fmt(control["mse"]), fmt(em["mse"]), fmt(row["em_over_control_mse"]),
                 fmt(control["relative_squared_error"]), fmt(em["relative_squared_error"])]
        stress_rows.append(f'<tr data-layer="{row["layer"]}">' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    stress_table = '<div class="table-scroll"><table><thead><tr>' + ''.join(f'<th>{value}</th>' for value in ["Layer", "Question", "Recorded answer source", "Answer tokens", "Control MSE", "EM MSE", "EM / control", "Control SSE / energy", "EM SSE / energy"]) + '</tr></thead><tbody id="stress-rows">' + ''.join(stress_rows) + '</tbody></table></div>'
    provenance = report["dataset_manifest_summary"]
    metadata = {"protocol": protocol, "dataset": provenance, "runtime": report["runtime"],
                "hook_audits": report["hook_audits"], "verification": report["verification"],
                "bootstrap": report["bootstrap"], "source_hashes": report["source_hashes"],
                "numerical_validation_protocol_change": report["numerical_validation_protocol_change"]}
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Does the frozen MOLT still reconstruct the medical adapter?</title>
<style>
:root{{--ink:#1c3042;--muted:#526675;--line:#d7e0e6;--blue:#24679a;--orange:#c6602f;--paper:#fff;--wash:#f3f6f7}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--wash);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1260px;margin:auto;padding:40px 24px 70px}}header{{max-width:1060px}}.eyebrow{{color:#17636e;text-transform:uppercase;letter-spacing:.13em;font-size:.76rem;font-weight:750}}h1{{font:500 clamp(2rem,4.3vw,3.3rem)/1.12 Georgia,serif;margin:14px 0 20px}}h2{{font-size:1.4rem;margin:0 0 12px;line-height:1.3}}h3{{font-size:1.05rem;margin:20px 0 8px}}p{{margin:10px 0}}.lede{{font-size:1.08rem;color:var(--muted);max-width:980px}}.facts{{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}}.fact{{border-left:3px solid #b2cdd4;padding:0 18px 0 12px}}.fact b{{font-size:1.45rem;display:block}}.fact span{{font-size:.82rem;color:var(--muted)}}nav{{display:flex;gap:16px;flex-wrap:wrap;font-size:.86rem;margin:22px 0}}a{{color:#17636e;text-underline-offset:3px}}.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:24px;margin:24px 0}}.finding{{border-left:4px solid #17636e;background:#eaf4f4;padding:16px 20px;border-radius:4px}}.note{{background:#fff8ed;border-left:4px solid #bc8639;padding:13px 17px;border-radius:4px;font-size:.92rem}}.small,small,figcaption{{font-size:.84rem;color:var(--muted)}}figure{{margin:20px 0}}figure svg{{display:block;width:100%;height:auto;max-width:100%}}figcaption{{margin-top:12px;line-height:1.55}}.table-scroll{{overflow:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:.83rem;font-variant-numeric:tabular-nums}}th,td{{text-align:right;padding:10px 12px;border-bottom:1px solid #e2e8ed;vertical-align:top;white-space:nowrap}}th{{font-size:.74rem;background:#f5f8fa;white-space:normal;color:#43596b;min-width:85px}}th:first-child,td:first-child{{text-align:left}}.interval{{display:block;font-size:.78rem;color:var(--muted)}}details{{border-top:1px solid var(--line);padding-top:13px;margin-top:20px}}summary{{cursor:pointer;color:#17636e;font-weight:650}}pre{{background:#f6f8fa;border-radius:7px;padding:16px;font-size:.78rem;white-space:pre-wrap;overflow-wrap:anywhere;max-height:520px;overflow:auto}}.definitions{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.definition{{border:1px solid var(--line);border-radius:9px;padding:14px 17px}}.formula{{font-family:ui-monospace,monospace;font-size:.86rem;background:#edf2f5;padding:7px 10px;display:block;margin:8px 0;white-space:normal}}select{{font:inherit;padding:7px 12px;border:1px solid #a9bac5;background:white;border-radius:6px;margin:8px}}a:focus-visible,summary:focus-visible,select:focus-visible{{outline:3px solid #2f8590;outline-offset:4px}}.model-control{{color:var(--blue);font-weight:700}}.model-em{{color:var(--orange);font-weight:700}}footer{{font-size:.84rem;color:var(--muted);margin-top:24px}}@media(max-width:700px){{main{{padding:26px 13px 45px}}.panel{{padding:16px}}.definitions{{grid-template-columns:1fr}}.facts{{gap:12px}}th,td{{padding:9px 8px}}}}
</style></head><body><main>
<header><div class="eyebrow">Qwen3-4B · matched reconstruction study</div><h1>Does the frozen MOLT still reconstruct the medical adapter?</h1><p class="lede">Compare the released MOLT’s predicted MLP output with each model’s actual MLP output while both process exactly the same recorded conversations. The main figures show reconstruction error across 1,000 prompts, separately for six layers.</p>
<div class="facts"><div class="fact"><b>1,000</b><span>paired UltraChat prompts</span></div><div class="fact"><b>6 layers</b><span>10, 16, 18, 22, 25, 29 · zero-based</span></div><div class="fact"><b>2 models</b><span>control and medical adapter</span></div><div class="fact"><b>6 answers</b><span>separate dashboard stress cohort</span></div></div>
<nav><a href="#distributions">Distributions</a><a href="#paired">Paired changes</a><a href="#scale">Error and output scale</a><a href="#user">User tokens</a><a href="#stress">Dashboard cases</a><a href="#method">Method and downloads</a></nav></header>
<div class="finding">{finding}<p class="small">{tail_finding}</p></div>
<section class="panel" id="distributions"><h2>Answer-token error distributions</h2><p><span class="model-control">Blue: control</span> · <span class="model-em">Orange: medical adapter</span>. Each observation is one prompt’s mean squared reconstruction error over its recorded assistant tokens and 2,560 output dimensions.</p>{distribution_figure}<div class="note">Overlapping distributions do not establish good reconstruction or equivalence: both models can have substantial error. These frozen-MOLT measurements also do not validate a causal interpretation of individual gate activations.</div><details><summary>See cumulative distributions</summary>{ecdf_figure}</details></section>
<section class="panel" id="paired"><h2>How much does error change on the same prompt?</h2><p>Values above 1 indicate more reconstruction error under the medical adapter. Ratios of the two models’ means or medians differ from first forming one ratio per matched prompt and then summarizing those ratios. The latter summaries retain prompt-level pairing.</p>{ratios_figure}{summary_table("answer", summaries, layers)}<p class="small">Intervals are percentile intervals from 2,000 paired prompt bootstrap resamples, seed {SEED}. They describe sampling uncertainty within this selected corpus; they do not cover model-training variability or establish equivalence.</p></section>
<section class="panel" id="scale"><h2>Read raw error alongside the size of the native output</h2><p>A change in raw MSE can accompany a change in the native MLP output’s scale. The table also reports error divided by uncentered target energy and the target’s root-mean-square magnitude. These are measured per prompt before averaging.</p>{descriptive_table("answer", summaries, layers)}<p class="small">SSE / target energy is a relative squared error, not variance explained. “Reduction vs saved-mean predictor” compares with the MOLT checkpoint’s saved output mean; this mean is not fitted to the evaluation data. Layer-specific raw units make direct raw-MSE comparisons across layers difficult.</p></section>
<section class="panel" id="user"><h2>Secondary comparison: user-content tokens</h2><p>These measurements exclude assistant answers and template tokens. They help distinguish changes already present while reading the user query from changes observed while processing a recorded answer.</p>{summary_table("user", summaries, layers)}<details><summary>Show user-token distributions, paired plots, and scale measurements</summary>{user_figures}{descriptive_table("user", summaries, layers)}</details></section>
<section class="panel" id="stress"><h2>The six dashboard answers, kept separate</h2><p>Each saved preferred or undesired answer is replayed in <em>both</em> models, including its full original answer-token sequence. The source label identifies the model that originally generated it; it does not restrict which model processes it here. These selected cases are excluded from the 1,000-prompt corpus summaries.</p><label for="stress-layer">Show layer</label><select id="stress-layer"><option value="all">All layers</option>{''.join(f'<option value="{layer}">Layer {layer}</option>' for layer in layers)}</select>{stress_table}<p class="small">No new answers were generated for this comparison. Reading a saved answer under a model does not imply that model would naturally produce that answer.</p></section>
<section class="panel" id="method"><h2>What was measured?</h2><div class="definitions"><div class="definition"><strong>Raw MSE</strong><span class="formula">sum((MOLT prediction − native MLP output)^2) / (answer tokens × 2,560)</span><p class="small">Lower is better. The main distributions give every prompt equal weight.</p></div><div class="definition"><strong>Relative squared error</strong><span class="formula">sum(squared reconstruction error) / sum(native MLP output^2)</span><p class="small">The denominator is uncentered output energy. This is not variance explained.</p></div><div class="definition"><strong>Target RMS</strong><span class="formula">sqrt(sum(native MLP output^2) / (tokens × 2,560))</span><p class="small">Describes the scale of the output being reconstructed.</p></div><div class="definition"><strong>Reduction vs saved-mean predictor</strong><span class="formula">1 − reconstruction SSE / sum((native MLP output − saved mean)^2)</span><p class="small">Uses the frozen checkpoint’s output mean. Negative values mean worse reconstruction than this mean-only predictor.</p></div></div>
<h3>Matched inputs and passive observation</h3><p>Both models receive identical saved token IDs. Native Qwen computation remains intact; hooks only record activations. MOLT inputs are the post-attention residuals before the layer’s post-attention normalization. Reconstruction targets are native MLP outputs before residual addition. The native models run in BF16; full frozen-MOLT reconstruction runs in FP32 with TF32 disabled, with FP64 error reductions.</p><p>The MOLT was trained on the base model. Under the medical adapter, both its input representation and the native MLP mapping may change. This comparison combines those effects; it does not isolate either mechanism.</p>
<details><summary>Numerical validation and its protocol revision</summary><p>The first wave stopped during layer 18 validation when a near-zero coordinate differed by approximately 5.7 × 10⁻⁵, above the original 5 × 10⁻⁵ elementwise absolute tolerance. An independent full CPU float64 reconstruction gave approximately 9.1 × 10⁻⁷ relative L2 error on that sample. Vector acceptance was changed to a per-token relative L2 tolerance of 10⁻⁵ against this independent reference, in both raw and standardized coordinates. Encoder vectors must meet the same per-token relative L2 criterion, and all CPU/GPU gate masks must agree. The initial revision retained a scalar encoder test; a later inactive encoder value differed by approximately 2.0 × 10⁻⁴ while remaining approximately 0.080 from its gate boundary, with every gate mask unchanged. Encoder scalar and reconstruction elementwise comparisons now remain nonblocking diagnostics; reconstruction arithmetic is unchanged.</p><p>Layers 10 and 16 had completed and layer 18 was partial. The first wave was remeasured from the same saved native captures; the revised criteria were then used for layers 22, 25, and 29. Model weights and the FP32 reconstruction formula did not change. This report requires every layer and model to pass the revised criteria and independently checks the saved norm errors, rather than accepting a summary pass flag alone.</p></details>
<h3>Corpus and limits</h3><p>We selected 1,000 first user–assistant pairs from <a href="{h(provenance['dataset_card_url'])}">UltraChat test_sft</a> at a pinned revision. Selection is deterministic, removes duplicate normalized prompts, excludes the 64 historical reconstruction examples, and excludes exact normalized-text matches to the medical training source. Complete user contents contain 8–192 measured tokens; recorded answers contain at least 16 tokens and are capped at their first 256 tokens. Template tokens and EOS are excluded from the primary user/answer summaries.</p><p>UltraChat’s test split is <strong>not established as held out from Qwen or MOLT training</strong>. Exact-text exclusion does not rule out semantic overlap with medical training. The source answers are recorded synthetic ChatGPT continuations, not new outputs from either tested model. This study measures local reconstruction fidelity, not answer quality, EM prevalence, or the effect of replacing native MLPs.</p>
<p><a href="summary.json">Summary JSON</a> · <a href="per-example.csv">All per-example metrics (CSV)</a> · <a href="build-manifest.json">Artifact hashes</a> · <a href="https://github.com/rakaar/qwen3-molt-exploration/tree/main/experiments/medical-em-molt-reconstruction">Code and protocol</a> · <a href="../2026-10-23-medical-molt-gallery/">Activation gallery</a></p><p class="small">The CSV includes user, answer, template, all-token, and first-token-excluded metrics. Raw answers and token sequences are omitted from this report and its downloads. PNG and SVG versions of every figure are linked beneath it. This HTML embeds its figures and displayed data, so it works without a local server.</p>
<details><summary>Protocol, provenance, and validation</summary><pre>{h(json.dumps(clean_json(metadata), indent=2, ensure_ascii=False))}</pre></details></section>
<footer>Observed differences on one fixed adapter and a selected text corpus. Reconstruction checks narrow interpretation limits; causal gate claims still require interventions.</footer>
</main><script>document.getElementById('stress-layer').addEventListener('change',function(){{for(const row of document.querySelectorAll('#stress-rows tr'))row.hidden=this.value!=='all'&&row.dataset.layer!==this.value;}});</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PRIVATE / "output")
    parser.add_argument("--dataset", type=Path, default=PRIVATE / "dataset.jsonl")
    parser.add_argument("--dataset-manifest", type=Path, default=PRIVATE / "dataset-manifest.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "results/2026-10-23-medical-molt-reconstruction")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    assert protocol["layers"] == [10, 16, 18, 22, 25, 29]
    dataset, dataset_manifest, runtime, hooks = verify_inputs(args, protocol)
    rows, lookup, verification, validations = load_measurements(args, protocol, dataset)
    summaries, stress = summarize(protocol, dataset, lookup)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    configure_plots()
    plot_info = make_plots(protocol, dataset, lookup, summaries, figures)
    source = dataset_manifest["source"]
    public_dataset_metadata = {key: source[key] for key in ("dataset", "revision", "split", "file", "sha256", "license_declared_by_dataset_card", "dataset_card_url", "selection", "selection_salt", "medical_disjointness_scope", "prior_sample_sha256")}
    public_dataset_metadata.update({"selection_audit": dataset_manifest["selection"], "cohorts": dataset_manifest["cohorts"]})
    report = {"built_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "dataset_manifest_summary": public_dataset_metadata, "runtime": runtime,
              "summaries": summaries, "dashboard_stress": stress, "hook_audits": hooks,
              "verification": verification, "reconstruction_validation": validations,
              "numerical_validation_protocol_change": {"final_criteria_revision": VALIDATION_REVISION, **VALIDATION_CHANGE},
              "bootstrap": {"resamples": BOOTSTRAPS, "seed": SEED, "method": "paired-prompt percentile interval for ratio of mean per-prompt MSE"},
              "plot_axes_and_bins": plot_info,
              "source_hashes": {"dataset": sha(args.dataset), "dataset_manifest": sha(args.dataset_manifest),
                                "protocol": sha(args.protocol), "runtime": sha(args.input / "runtime.json")},
              "analysis_packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")},
              "interpretation_limits": ["Selected natural text corpus, not established as held out from Qwen or MOLT training",
                                        "Frozen MOLT error under EM combines representation drift and MLP mapping changes",
                                        "Distribution overlap is not equivalence or sufficient reconstruction quality",
                                        "No causal claim about gate behavior or EM prevalence",
                                        "Relative squared error uses uncentered output energy; it is not variance explained"],
              "report_contains_raw_answers": False}
    write_json(args.output / "summary.json", report)
    write_csv(args.output / "per-example.csv", rows, dataset)
    (args.output / "index.html").write_text(build_html(args.output, report))
    files = [args.output / name for name in ("index.html", "summary.json", "per-example.csv")] + sorted(figures.glob("*.svg")) + sorted(figures.glob("*.png"))
    write_json(args.output / "build-manifest.json", {"builder_sha256": sha(__file__),
                                                    "files": {str(path.relative_to(args.output)): sha(path) for path in files},
                                                    "measurement_rows": len(rows), "paired_corpus_examples": protocol["n_examples"]})
    print(json.dumps({"output": str(args.output), "measurement_rows": len(rows), "figure_pairs": len(list(figures.glob('*.svg'))),
                      "contains_raw_answers": False}, indent=2))


if __name__ == "__main__":
    main()
