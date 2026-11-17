"""Extract the five candidate operators from the pinned MOLT release."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / "experiments/eleven-downstream-gates-20261103"
CACHE = ROOT / "experiments/ch-alllayer-causal-screen/gates"
WEIGHTS = HERE / "weights"
CANDIDATES = {33: [14], 34: [965, 1440], 35: [760, 2117]}
REVISION = "081ce901b090b64aeecf9bc8709b57c10411e32a"
RANKS = (512, 256, 128, 64, 32)
GROUP_ENDS = (80, 240, 560, 1200, 2480)
REPO = "https://huggingface.co/georglange/qwen3-4b-molt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, start, size):
    end = start + size - 1
    for attempt in range(3):
        try:
            with requests.get(
                url, params={"repair_slice": f"{start}-{end}", "n": time.time_ns()},
                headers={"Range": f"bytes={start}-{end}"}, timeout=(10, 35), stream=True,
            ) as response:
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
                blob = response.raw.read(size + 1)
                assert len(blob) == size
                return blob
        except (AssertionError, OSError, requests.RequestException):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def extract(layer):
    url = f"{REPO}/resolve/{REVISION}/layer-{layer}/checkpoint.safetensors"
    first = fetch(url, 0, 65536)
    header_size = struct.unpack("<Q", first[:8])[0]
    assert header_size + 8 <= len(first)
    header = json.loads(first[8:8 + header_size])
    assert header == json.loads((CACHE / f"layer-{layer}-header.json").read_text())

    record = next(v for v in gate_manifest["layers"] if v["layer"] == layer)
    cache_path = ROOT / record["project_path"]
    assert sha(cache_path) == record["sha256"]
    with np.load(cache_path, allow_pickle=False) as archive:
        gates = {k: archive[k].copy() for k in archive.files}

    features = []
    for transform in CANDIDATES[layer]:
        group = next(i for i, end in enumerate(GROUP_ENDS) if transform < end)
        group_start = GROUP_ENDS[group - 1] if group else 0
        index = transform - group_start
        rank = RANKS[group]
        arrays = {
            "e": gates["encoder"][transform],
            "bias": gates["bias"][transform:transform + 1],
            "theta": gates["threshold"].reshape(-1)[transform:transform + 1],
            "input_mean": gates["mean"],
            "input_std": gates["std"],
            "output_std": gates["outstd"],
        }
        slices = []
        for role, shape in [("U", (rank, 2560)), ("V", (2560, rank))]:
            name = f"{role}s.{group}"
            spec = header[name]
            assert spec["dtype"] == "F32"
            assert spec["shape"] == [GROUP_ENDS[group] - group_start, *shape]
            size = rank * 2560 * 4
            start = 8 + header_size + spec["data_offsets"][0] + index * size
            blob = fetch(url, start, size)
            arrays[role] = np.frombuffer(blob, dtype="<f4").reshape(shape)
            assert np.isfinite(arrays[role]).all()
            slices.append(dict(tensor=name, index=index, start=start, bytes=size,
                               sha256=hashlib.sha256(blob).hexdigest()))
        path = WEIGHTS / f"L{layer}-T{transform}.npz"
        np.savez_compressed(path, **arrays)
        features.append(dict(layer=layer, transform=transform, rank=rank, file=path.name,
                             sha256=sha(path), source=url, slices=slices,
                             gate_cache_sha256=record["sha256"]))
        print(f"Extracted L{layer}-T{transform} (rank {rank})", flush=True)
    return features


if __name__ == "__main__":
    WEIGHTS.mkdir(exist_ok=True)
    gate_manifest = json.loads((PRIOR / "gate-cache-manifest.json").read_text())
    assert gate_manifest["revision"] == REVISION
    assert gate_manifest["NPZ_and_original_tensor_hashes_verified"]
    upstream = json.loads((PRIOR / "weights/manifest.json").read_text())
    assert upstream["revision"] == REVISION
    t2375 = next(v for v in upstream["features"] if v["transform"] == 2375)
    original = PRIOR / "weights" / t2375["file"]
    assert sha(original) == t2375["sha256"]
    shutil.copy2(original, WEIGHTS / original.name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        extracted = [f for group in pool.map(extract, CANDIDATES) for f in group]
    (WEIGHTS / "manifest.json").write_text(json.dumps(
        dict(revision=REVISION, features=[t2375] + extracted), indent=2) + "\n")
