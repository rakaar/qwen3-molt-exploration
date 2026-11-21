"""Compare two independent single-token ablations against identical intact logits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

protocol = json.loads(Path("protocol.json").read_text())
assert digest("cases.json") == protocol["case_sha256"]
cases = json.loads(Path("cases.json").read_text())
assert len(cases) == protocol["cases"]
checkpoint_hash = digest("layer-29.safetensors")
assert checkpoint_hash == "3b89660603200ca2f9eabd898f144eacc14c4e183bc148681e414b5f35db7581"
manifest = json.loads(Path("model-verified.json").read_text())
assert manifest["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
assert all(digest(Path("model") / r["file"]) == r["sha256"] for r in manifest["files"])
torch.backends.cuda.matmul.allow_tf32 = False
model = AutoModelForCausalLM.from_pretrained("model", dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).eval().cuda().requires_grad_(False)
layer = model.model.layers[29]
state = load_file("layer-29.safetensors", device="cuda")
target = 2375
mean = state["input_standardizer.mean"][29]
scale = state["input_standardizer.std"][29]
out_scale = state["output_standardizer.std"][29]
encoder = state["e.weight"]
bias = state["e.bias"]
theta = state["nonlinearity.theta"].flatten()
ends = np.cumsum([state[f"Us.{g}"].shape[0] for g in range(5)]).tolist()
assert ends[-1] == len(bias) == 2480

def factors(feature):
    group = next(i for i, end in enumerate(ends) if feature < end)
    index = feature - (ends[group - 1] if group else 0)
    return state[f"Vs.{group}"][index], state[f"Us.{group}"][index]

v, u = factors(target)
with np.load("reference-L29-T2375.npz") as ref:
    for key, val in dict(V=v, U=u, e=encoder[target], bias=bias[target], theta=theta[target],
                         input_mean=mean, input_std=scale, output_std=out_scale).items():
        assert np.array_equal(val.cpu().numpy().flatten(), ref[key].flatten())

def predict(ids):
    return model(input_ids=ids, use_cache=True, logits_to_keep=1).logits[0, -1]

def ablate(ids, z, feature, gate, log_p):
    v, u = factors(feature)
    delta = gate * ((z @ v) @ u) * out_scale
    def edit(module, args, output):
        result = output.clone()
        result[0, -1] = (output[0, -1].float() - delta).to(output.dtype)
        assert torch.equal(result[:, :-1], output[:, :-1])
        return result

    hook = layer.mlp.register_forward_hook(edit)
    try:
        off = predict(ids)
    finally:
        hook.remove()
    log_q = off.double().log_softmax(-1)
    kl = float((log_p.exp() * (log_p - log_q)).sum())
    assert np.isfinite(kl) and kl >= -1e-12
    return max(0., kl), float(delta.norm()), int(off.argmax())

@torch.inference_mode()
def measure(c):
    ids = torch.tensor([c["prefix"]], device="cuda")
    activations = []
    def observe(module, args):
        activations.append((args[0][0,-1].float() - mean) / scale)
    hook = layer.post_attention_layernorm.register_forward_pre_hook(observe)
    try:
        intact = predict(ids)
    finally:
        hook.remove()
    z, = activations
    raw = encoder @ z + bias
    raw[target] = torch.dot(z, encoder[target]) + bias[target]
    assert raw[target] > max(0., float(theta[target]))
    assert abs(float(raw[target]) - c["original_gate"]) < 0.10
    active = (raw > theta) & (raw > 0)
    active[target] = False
    candidates = active.nonzero().flatten()
    log_p = intact.double().log_softmax(-1)
    row = {k:c[k] for k in ("id", "conversation", "position", "token", "category", "original_gate")}
    row.update(target_gate=float(raw[target]), other_active=len(candidates), native_top=int(intact.argmax()))
    if len(candidates):
        control = int(candidates[(raw[candidates] - raw[target]).abs().argmin()])
        assert control != target and raw[control] > max(0., float(theta[control]))
        row.update(control_id=control, control_gate=float(raw[control]))
        row["control_kl"],row["control_norm"],row["control_top"] = ablate(ids, z, control, raw[control], log_p)
    row["target_kl"],row["target_norm"],row["target_top"] = ablate(ids, z, target, raw[target], log_p)
    return row

rows = []
with Path("measurements.jsonl.tmp").open("w") as f:
    for case in cases:
        row = measure(case)
        f.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row)
        if len(rows) % 25 == 0:
            f.flush()
            print("Measured", len(rows), "of", len(cases), flush=True)
Path("measurements.jsonl.tmp").replace("measurements.jsonl")
assert len(rows) == len({r["id"] for r in rows}) == 310
verification = dict(cases=len(rows), matched=sum("control_id" in r for r in rows),
                    measurement_sha256=digest("measurements.jsonl"), case_sha256=digest("cases.json"),
                    protocol_sha256=digest("protocol.json"), code_sha256=digest(__file__),
                    checkpoint_sha256=checkpoint_hash, qwen_revision=manifest["revision"], vocab=model.config.vocab_size,
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__, transformers=transformers.__version__)
Path("verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print("Complete", verification["matched"], "matched controls", flush=True)
