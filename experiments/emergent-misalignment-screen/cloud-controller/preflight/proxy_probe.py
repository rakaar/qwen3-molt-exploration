"""Credential-free Cloud network diagnostic; every request uses a fixed proxy.

No environment variables, credentials, provider calls, or dynamic destinations.
A per-command network permission is still required by the Cloud executor.
Success is a functional proxy check, not a general security attestation.
"""
import http.client
import json
import ssl
import sys


PROXY_HOST = "proxy"
PROXY_PORT = 8080


def probe(host, path, expect_blocked=False):
    result = {"host": host, "route": "HTTP CONNECT via proxy:8080",
              "expect_blocked": expect_blocked}
    connection = http.client.HTTPSConnection(
        PROXY_HOST, PROXY_PORT, timeout=10, context=ssl.create_default_context())
    connection.set_tunnel(host, port=443)
    try:
        connection.request("HEAD", path)
        response = connection.getresponse()
        result["http_status"] = response.status
        result["passed"] = not expect_blocked and 200 <= response.status < 400
    except (OSError, http.client.HTTPException) as exc:
        result.update(error_type=type(exc).__name__, errno=getattr(exc, "errno", None),
                      error=str(exc)[:240])
        # A failed socket or DNS lookup does not establish an enforced deny.
        result["passed"] = expect_blocked and str(exc).startswith("Tunnel connection failed: 403 ")
    finally:
        connection.close()
    return result


def main():
    results = [
        probe("github.com", "/rakaar/qwen3-molt-exploration"),
        probe("pypi.org", "/pypi/vastai/1.8.3/json"),
        probe("example.org", "/", expect_blocked=True),
    ]
    passed = all(item["passed"] for item in results)
    print(json.dumps({"passed": passed, "credentials_used": False,
                      "provider_calls": 0, "results": results}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
