"""Patch one downstream MOLT contribution at a time during alphabet prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def feature_name(feature):
    return f"L{feature[0]}-T{feature[1]}"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((HERE / "protocol.json").read_text())
    base = json.loads((HERE / "expected-base-verification.json").read_text())
    manifest = json.loads((HERE / "weights/manifest.json").read_text())
    OUT.mkdir()
    assert base["revision"] == config["base_revision"]
    assert manifest["revision"] == config["molt_revision"]
    assert sha(HERE / "references.npz") == config["references_sha256"]
    for entry in base["files"]:
        path = args.model_dir / entry["file"]
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], entry["file"]
    assert torch.__version__ == "2.14.0+cu126" and transformers.__version__ == "4.57.6"
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = config["source"]["layer"], config["source"]["transform"]
    repairs = [tuple(feature) for feature in config["repairs"]]
    features = [source] + repairs
    weights = {}
    for item in manifest["features"]:
        feature = item["layer"], item["transform"]
        path = HERE / "weights" / item["file"]
        assert sha(path) == item["sha256"]
        with np.load(path, allow_pickle=False) as a:
            weights[feature] = {key: torch.from_numpy(a[key].copy()).float().cuda() for key in a.files}
        assert weights[feature]["U"].shape == (item["rank"], 2560)
        assert weights[feature]["V"].shape == (2560, item["rank"])
        assert torch.all(weights[feature]["input_std"] > 0)
    assert set(features) == weights.keys()

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval().requires_grad_(False)
    assert len(model.model.layers) == 36 and model.config.hidden_size == 2560
    tokens = tok.encode(config["prompt"], add_special_tokens=False)
    position = config["position"]
    assert tokens == config["input_ids"] and position == len(tokens) - 1
    assert tok.decode([tokens[position]]) == " is"
    inputs = torch.tensor([tokens], device="cuda")
    spellings = {
        label: [tok.encode(s, add_special_tokens=False) for s in variants]
        for label, variants in config["answer_spellings"].items()
    }
    assert all(len(ids) == 1 for variants in spellings.values() for ids in variants)
    ids_by_answer = {k: [ids[0] for ids in v] for k, v in spellings.items()}
    assert len(set(sum(ids_by_answer.values(), []))) == 4
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def calculate(x, feature):
        w = weights[feature]
        z = (x - w["input_mean"]) / w["input_std"]
        raw = F.linear(z, w["e"].reshape(1, -1), w["bias"]).squeeze(-1)
        gate = torch.where((raw > w["theta"][0]) & (raw > 0), raw, 0.)
        operator = (z @ w["V"] @ w["U"]) * w["output_std"]
        return dict(input=x[0, position].clone(), raw=raw[0, position].clone(),
                    gate=gate[0, position].clone(), operator=operator[0, position].clone(),
                    contribution=(gate[..., None] * operator)[0, position].clone())

    def run(name, off=False, repair=None, observe=(), donor=None, zero=False):
        observations, handles, edits = {}, [], {}
        edited_features = [source] if (off or zero) else []
        if repair is not None:
            edited_features.append(repair)

        def observer(layer, targets):
            def capture(_module, args):
                x = args[0].detach().float()
                assert x.shape == (1, len(tokens), 2560)
                for feature in targets:
                    observations[feature] = calculate(x, feature)
            return capture

        def editor(feature):
            def edit(_module, _args, output):
                assert feature not in edits
                current = observations[feature]["contribution"]
                if feature == source:
                    delta = torch.zeros_like(current) if zero else -current
                else:
                    delta = donor[feature]["contribution"] - current
                before = output[0, position].float()
                after = (before + delta).to(output.dtype)
                changed = output.clone()
                changed[0, position] = after
                assert torch.equal(changed[:, :position], output[:, :position])
                assert torch.isfinite(after).all()
                if zero:
                    assert torch.equal(changed, output)
                edits[feature] = dict(before=before, after=after.float(), intended_delta=delta,
                                      applied_delta=after.float() - before)
                return changed
            return edit

        for layer in sorted({f[0] for f in observe}):
            block = model.model.layers[layer]
            targets = [f for f in observe if f[0] == layer]
            handles.append(block.post_attention_layernorm.register_forward_pre_hook(observer(layer, targets)))
        for feature in edited_features:
            handles.append(model.model.layers[feature[0]].mlp.register_forward_hook(editor(feature)))
        try:
            mask = torch.ones_like(inputs)
            prediction = model(input_ids=inputs, attention_mask=mask, use_cache=True)
        finally:
            for handle in handles:
                handle.remove()
        assert set(observations) == set(observe) and list(edits) == edited_features
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())

        first = prediction.logits[0, -1].float()
        assert torch.isfinite(first).all()
        p = first.softmax(-1)
        probabilities = {label: sum(float(p[i]) for i in ids) for label, ids in ids_by_answer.items()}
        values, indices = p.topk(5)
        top = [dict(id=int(i), token=tok.decode([int(i)]), probability=float(v))
               for v, i in zip(values, indices)]
        generated = []
        for step in range(config["max_new_tokens"]):
            chosen = int(prediction.logits[0, -1].argmax())
            generated.append(chosen)
            if chosen in eos or step == config["max_new_tokens"] - 1:
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            prediction = model(
                input_ids=torch.tensor([[chosen]], device="cuda"),
                attention_mask=mask, past_key_values=prediction.past_key_values, use_cache=True,
            )

        arrays = {"first_logits": first.cpu().numpy()}
        for feature, data in observations.items():
            for key, value in data.items():
                arrays[f"{feature_name(feature)}_{key}"] = value.cpu().numpy()
        for feature, data in edits.items():
            for key, value in data.items():
                arrays[f"{feature_name(feature)}_{key}"] = value.cpu().numpy()
        path = OUT / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        row = dict(condition=name, repair=feature_name(repair) if repair else None,
                   response=tok.decode(generated, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False),
                   first_token=tok.decode(generated[:1]), output_ids=generated,
                   stop_reason="eos" if generated[-1] in eos else "token_limit",
                   probabilities=probabilities, top5=top, probability_token_ids=ids_by_answer,
                   observed_gates={feature_name(f): float(val["gate"]) for f, val in observations.items()},
                   edited_sites=[dict(feature=feature_name(f), token_position=position,
                                      delta_l2=float(d["applied_delta"].norm())) for f, d in edits.items()],
                   generated_positions_edited=[], file=path.name, sha256=sha(path))
        print(json.dumps(dict(condition=name, first=row["first_token"], probabilities=probabilities)), flush=True)
        return row, observations, arrays

    started = time.monotonic()
    rows = []
    data = dict(status="running", protocol=config, runs=rows,
                runtime=dict(torch=torch.__version__, transformers=transformers.__version__,
                             gpu=torch.cuda.get_device_name(), base_hashes_verified=True,
                             weights_hashes_verified=True, attention="eager", tf32=False))

    def append(row):
        rows.append(row)
        save(OUT / "results.json", data)

    native, donors, original = run("native", observe=features)
    append(native)
    unhooked, _, bare = run("native-unhooked")
    zero, _, null = run("zero-strength", observe=features, zero=True)
    for control, arrays in [(unhooked, bare), (zero, null)]:
        np.testing.assert_array_equal(arrays["first_logits"], original["first_logits"])
        assert control["output_ids"] == native["output_ids"]
        append(control)
    off, _, ablated = run("T2375-off", off=True, observe=features)
    append(off)
    repeat, _, ablated_repeat = run("T2375-off-repeat", off=True, observe=features)
    np.testing.assert_array_equal(ablated_repeat["first_logits"], ablated["first_logits"])
    assert repeat["output_ids"] == off["output_ids"]
    append(repeat)
    assert native["first_token"].strip().casefold() == "k"
    assert off["first_token"].strip().casefold() == "l"
    with np.load(HERE / "references.npz", allow_pickle=False) as anchors:
        checks = {}
        for label, arrays in [("native", original), ("off", ablated)]:
            saved_logits = anchors[f"{label}_logits"]
            checks[f"{label}_reference_logits_exact"] = bool(np.array_equal(arrays["first_logits"], saved_logits))
            checks[f"{label}_reference_logit_max_abs_diff"] = float(np.max(np.abs(arrays["first_logits"] - saved_logits)))
            for feature in repairs:
                np.testing.assert_array_equal(arrays[f"{feature_name(feature)}_input"],
                                              anchors[f"{label}_inputs"][feature[0] - 33])
            np.testing.assert_allclose(arrays["L29-T2375_contribution"],
                                       anchors[f"{label}_contribution29"], rtol=1e-5, atol=1e-5)
    data["baseline_checks"] = checks
    for feature in repairs:
        repaired, _, arrays = run(feature_name(feature), off=True, repair=feature,
                                  observe=features, donor=donors)
        np.testing.assert_array_equal(arrays["L29-T2375_after"], ablated["L29-T2375_after"])
        np.testing.assert_array_equal(arrays[f"{feature_name(feature)}_input"],
                                      ablated[f"{feature_name(feature)}_input"])
        append(repaired)
    data.update(status="complete", runtime_seconds=time.monotonic() - started)
    save(OUT / "results.json", data)


if __name__ == "__main__":
    main()
