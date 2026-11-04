#!/usr/bin/env bash
# Build python3-flagtree-cpu_*.deb for the host architecture.
#
#   DEB_BASE_IMAGE   ubuntu:24.04 (default) or ubuntu:22.04
#   MAX_JOBS         parallel compile jobs (default 4; the LLVM link is RAM-heavy)
#   DEB_VERSION_SUFFIX  "auto" (default, +ubuntu24.04) or an explicit suffix / ""
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -f third_party/sleef/CMakeLists.txt ]; then
    echo "ERROR: third_party/sleef is empty; run: git submodule update --init --recursive" >&2
    exit 1
fi

mkdir -p dist
rm -rf dist/output

DEB_BASE_IMAGE="${DEB_BASE_IMAGE:-ubuntu:24.04}"
MAX_JOBS="${MAX_JOBS:-4}"
DEB_VERSION_SUFFIX="${DEB_VERSION_SUFFIX:-auto}"

echo ">>> Building wheel + .deb on ${DEB_BASE_IMAGE} ($(uname -m), MAX_JOBS=${MAX_JOBS})"
docker build \
    --network=host \
    --build-arg DEB_BASE_IMAGE="${DEB_BASE_IMAGE}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    --build-arg DEB_VERSION_SUFFIX="${DEB_VERSION_SUFFIX}" \
    -f packaging/debian/build-helpers/Dockerfile.deb \
    --target deb-output \
    --output "type=local,dest=${REPO_ROOT}/dist" \
    .

echo ""
echo ">>> Output:"
ls -lh dist/output/
