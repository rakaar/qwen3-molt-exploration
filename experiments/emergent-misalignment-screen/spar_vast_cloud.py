"""Cloud-portable SPAR guard: official CLI, fixed project/account/host, no deletion.

The existing local helper remains unchanged. This controller-only counterpart
reads a cloud network-secret placeholder, never a local or default Vast key.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ACCOUNT = 710735
LABEL = "spar-psy-neuro-em-screen-20261017"
HOST = "https://cloud.vast.ai"
BUDGET = 10.0
ROOT = Path(__file__).resolve().parent


class GuardError(Exception):
    pass


def require_project():
    manifest = json.loads((ROOT / "models.json").read_text())
    if {m["repo_id"] for m in manifest} != {
        "Qwen/Qwen3-4B", "Amarjitkr/Qwen3-4B-misaligned", "Amarjitkr/Qwen3-4B-misaligned-v2"
    }:
        raise GuardError("This helper is restricted to the Qwen3 screening project")
    remote = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "remote.origin.url"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if remote not in {"git@github.com:rakaar/qwen3-molt-exploration.git",
                      "https://github.com/rakaar/qwen3-molt-exploration.git",
                      "https://github.com/rakaar/qwen3-molt-exploration"}:
        raise GuardError("The checkout is not the authorized screening repository")


def call_cli(key, args):
    executable = shutil.which("vastai")
    if not executable:
        raise GuardError("Install the official vastai CLI in the controller; do not use another provider client")
    env = os.environ.copy()
    for name in ("VAST_AI_KEY", "VASTAI_API_KEY", "VAST_API_KEY", "VAST_URL", "VAST_API_URL"):
        env.pop(name, None)
    env["VAST_API_KEY"] = key
    # A fixed official host matches the user's approved network-secret scope.
    # Endpoint/credential overrides from caller arguments are prohibited below.
    result = subprocess.run([executable, "--raw", "--retry", "1", "--url", HOST, *args],
                            env=env, capture_output=True, text=True, timeout=90)
    try:
        stderr = json.loads(result.stderr)
    except (ValueError, TypeError):
        stderr = None
    if result.returncode or isinstance(stderr, dict) and stderr.get("error"):
        raise GuardError("Vast request failed; credentials and raw provider errors suppressed. Check state before retrying a mutation")
    try:
        value = json.loads(result.stdout.replace(key, "[REDACTED]"))
    except ValueError:
        raise GuardError("Vast did not return valid JSON; no fallback was used") from None
    if isinstance(value, dict) and (value.get("error") or value.get("success") is False):
        raise GuardError("Vast rejected the request; no fallback was used")
    return value


def check_identity(user):
    if not isinstance(user, dict) or type(user.get("id")) is not int or user["id"] != ACCOUNT:
        raise GuardError("Live account is not pinned SPAR account 710735; all provider operations refused")


def validate_args(args):
    forbidden = ("--api-key", "--url", "--curl", "--explain", "--retry", "--login", "--env", "--onstart", "--onstart-cmd", "--entrypoint", "--args")
    if any(arg.startswith("--") and any(opt.startswith(arg.split("=", 1)[0]) for opt in forbidden) for arg in args):
        raise GuardError("Credential, endpoint, startup, and diagnostic overrides are disabled")
    allowed = {("show", "user"), ("show", "instances"), ("show", "instance"),
               ("show", "ssh-keys"), ("search", "offers"), ("create", "instance"),
               ("create", "ssh-key"), ("attach", "ssh"), ("stop", "instance")}
    if len(args) < 2 or tuple(args[:2]) not in allowed:
        raise GuardError("Unsupported command. Deletion, restart, and unrelated account operations are disabled")


def instance_rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("instances"), list):
        return value["instances"]
    raise GuardError("Cannot establish the current instance inventory")


def guard_mutation(args, call):
    if args[:2] == ["stop", "instance"]:
        if len(args) != 3 or not args[2].isdigit():
            raise GuardError("Stop requires one explicit instance ID")
        found = [x for x in instance_rows(call(["show", "instances"])) if x.get("id") == int(args[2])]
        if len(found) != 1 or found[0].get("label") != LABEL:
            raise GuardError("Stop is authorized only for this screening worker; no other instance may be changed")
    if args[:2] == ["create", "instance"]:
        if len(args) < 3 or not args[2].isdigit():
            raise GuardError("Creation requires one offer ID")
        if "--label" not in args or args[args.index("--label") + 1] != LABEL:
            raise GuardError("The exact task-specific instance label is required")
        if "--disk" not in args or args[args.index("--disk") + 1] != "50":
            raise GuardError("This pilot is limited to a 50 GB worker disk")
        if any(x.get("label") == LABEL for x in instance_rows(call(["show", "instances"]))):
            raise GuardError("A screening instance already exists; inspect it instead of creating a duplicate")
        offers = call(["search", "offers", f"id={args[2]} num_gpus=1 gpu_ram>=24 compute_cap>=800 verified=true rentable=true reliability>=0.98 dph_total<=0.4",
                       "--limit", "1", "--storage", "50"])
        if not isinstance(offers, list) or len(offers) != 1 or offers[0].get("id") != int(args[2]):
            raise GuardError("The selected offer is no longer available within the approved constraints")
        offer = offers[0]
        rate = offer.get("dph_total")
        if not isinstance(rate, (int, float)) or not 0 < rate <= 0.4:
            raise GuardError("Hourly rental rate exceeds the screening limit")
        projected = 3 * rate + 40 * offer["inet_down_cost"] + 5 * offer["inet_up_cost"]
        if projected > BUDGET:
            raise GuardError("Conservative three-hour screening cost exceeds the $10 cap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "cli"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    require_project()
    args = ["show", "user"] if parsed.mode == "status" else parsed.args
    validate_args(args)
    key = os.environ.get("SPAR_VAST_API_KEY")
    if not key:
        raise GuardError("SPAR_VAST_API_KEY cloud secret is missing; local/default/personal keys will not be used")
    call = lambda command: call_cli(key, command)
    user = call(["show", "user"])
    check_identity(user)
    if args == ["show", "user"]:
        output = {k: user[k] for k in ("id", "username", "credit", "balance") if k in user}
        output["verified_team"] = "SPAR"
    else:
        guard_mutation(args, call)
        output = call(args)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (GuardError, OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError) as exc:
        print(str(exc) if isinstance(exc, GuardError) else "SPAR controller error; credential details suppressed", file=sys.stderr)
        sys.exit(1)
