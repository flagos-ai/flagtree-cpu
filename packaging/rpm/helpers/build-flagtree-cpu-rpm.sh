#!/usr/bin/env bash
# Build python3-flagtree-cpu-*.rpm for the host architecture.
#
#   RPM_BASE_IMAGE  fedora:43 (default) or openeuler/openeuler:24.03-lts
#   MAX_JOBS        parallel compile jobs (default 4; the LLVM link is RAM-heavy)
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -f third_party/sleef/CMakeLists.txt ]; then
    echo "ERROR: third_party/sleef is empty; run: git submodule update --init --recursive" >&2
    exit 1
fi

mkdir -p dist-rpm
rm -rf dist-rpm/output

RPM_BASE_IMAGE="${RPM_BASE_IMAGE:-fedora:43}"
MAX_JOBS="${MAX_JOBS:-4}"

echo ">>> Building wheel + .rpm on ${RPM_BASE_IMAGE} ($(uname -m), MAX_JOBS=${MAX_JOBS})"
docker build \
    --network=host \
    -f packaging/rpm/helpers/Dockerfile.rpm \
    --build-arg RPM_BASE_IMAGE="${RPM_BASE_IMAGE}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    --target rpm-output \
    --output "type=local,dest=${REPO_ROOT}/dist-rpm" \
    .

echo ""
echo ">>> Output:"
ls -lh dist-rpm/output/
