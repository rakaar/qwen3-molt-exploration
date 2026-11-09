#!/bin/bash
set -euo pipefail
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev make
gcc --version | head -1
