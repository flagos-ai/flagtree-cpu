#!/usr/bin/env bash
set -euo pipefail

# Three-stage entry for the Mac17,9 CPU-only DFlash2 route:
#   models -> environment -> local inference/benchmark.

runtime_version="20260828"
bundle_name="dflash2-m5-assets-${runtime_version}.tar.gz"
bundle_sha256="91c07ada1a90a9709bb6b801a77e0d43a2a671d5b9ed0a44d8d4fd7877920c7a"
bundle_url="${DFLASH2_RUNTIME_BUNDLE_URL:-https://github.com/kevinzs2048/flagtree-cpu/releases/download/dflash2-m5-runtime-${runtime_version}/${bundle_name}}"

script_dir="$(cd "$(dirname "$0")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"
install_root="${DFLASH2_INSTALL_ROOT:-$package_root/.dflash2-m5-runtime}"
bootstrap_root="$install_root/bootstrap"
bundle_path="$bootstrap_root/$bundle_name"
runtime_payload_root="$bootstrap_root/runtime-assets-$runtime_version"
assets_root="$runtime_payload_root/dflash2_m5"
source_root="$install_root/sources"
model_root="${MODEL_ROOT:-$install_root/models}"
install_log="$install_root/install.log"
constraints_file="$assets_root/metadata/m5-dflash2-python-constraints.txt"
runtime_assets_file="$assets_root/metadata/m5-dflash2-runtime-assets.sha256"
json_archive="$assets_root/sources/nlohmann-json-v3.11.3.tar.xz"
fmt_archive="$assets_root/sources/fmt-11.1.4.tar.gz"

runtime_workspace="${RUNTIME_WORKSPACE:-$source_root}"
if [[ -n "${VLLM024_WORKSPACE:-}" ]]; then
  vllm_source="${VLLM_SOURCE:-$VLLM024_WORKSPACE/vllm-0.24.0}"
  plugin_source="${FL_PLUGIN_SOURCE_ROOT:-$VLLM024_WORKSPACE/vllm-plugin-FL}"
else
  vllm_source="${VLLM_SOURCE:-$source_root/vllm-0.24.0}"
  plugin_source="${FL_PLUGIN_SOURCE_ROOT:-$source_root/vllm-plugin-FL-vllm024}"
fi

target_dir="${MODEL_DIR:-$model_root/Qwen3.8-27B-W4A8-GPTQ-G128-packed}"
draft_dir="${DFLASH2_DIR:-$model_root/Qwen3.8-27B-DFlash2}"
target_repo="${TARGET_REPO:-FlagRelease/Qwen3.8-27B-W4A8-arm-FlagOS-Express}"
draft_repo="${DFLASH2_REPO:-incoai/Qwen3.8-27B-DFlash2}"
target_revision="${TARGET_REVISION:-master}"
draft_revision="${DFLASH2_REVISION:-master}"
python_bin="${PYTHON_BIN:-$vllm_source/.venv/bin/python}"
flag_gems_lib="${FLAGGEMS_LIBTRITON_JIT_Q4_OP:-$runtime_workspace/build/flaggems-arm-m5/libflag_gems_arm_ops.dylib}"
profile="${FLAGOS_RUNTIME_PROFILE:-$runtime_workspace/flagos-macos-runtime/profiles/m5-dflash2-k7.env}"
json_source="$source_root/third-party/nlohmann-json-3.11.3"
fmt_source="$source_root/third-party/fmt-11.1.4"

bundle_matches_sha256() {
  [[ -f "$1" ]] || return 1
  [[ "$(shasum -a 256 "$1" | awk '{print $1}')" == "$bundle_sha256" ]]
}

prepare_runtime_bundle() {
  mkdir -p "$bootstrap_root"
  if ! bundle_matches_sha256 "$bundle_path"; then
    if [[ -e "$bundle_path" ]]; then
      mv "$bundle_path" "$bundle_path.invalid.$(date +%s)"
    fi
    local partial="$bundle_path.part.$$"
    echo "Downloading pinned DFlash2 runtime assets..."
    curl --fail --location --retry 3 --connect-timeout 20 \
      --output "$partial" "$bundle_url"
    bundle_matches_sha256 "$partial" \
      || die "runtime asset bundle SHA-256 mismatch: $partial"
    mv "$partial" "$bundle_path"
  fi

  if [[ -f "$runtime_payload_root/.dflash2-runtime-assets-$runtime_version" ]]; then
    return
  fi
  if [[ -e "$runtime_payload_root" ]]; then
    mv "$runtime_payload_root" "$runtime_payload_root.incomplete.$(date +%s)"
  fi
  local staging
  staging="$(mktemp -d "$bootstrap_root/.runtime-assets-staging.XXXXXX")"
  tar -xzf "$bundle_path" -C "$staging"
  [[ -f "$staging/dflash2_m5/metadata/m5-dflash2-runtime-assets.sha256" ]] \
    || die "runtime bundle is missing its checksum manifest"
  [[ -f "$staging/portable_env.sh" \
      && -f "$staging/qwen38_dflash2_generate.py" \
      && -f "$staging/run_dflash2_benchmark_vllm024.sh" ]] \
    || die "runtime bundle is missing launch helpers"
  touch "$staging/.dflash2-runtime-assets-$runtime_version"
  mv "$staging" "$runtime_payload_root"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_and_run_dflash2_m5.sh models
  bash scripts/setup_and_run_dflash2_m5.sh env
  bash scripts/setup_and_run_dflash2_m5.sh run "你的问题"
  bash scripts/setup_and_run_dflash2_m5.sh benchmark
  bash scripts/setup_and_run_dflash2_m5.sh all "你的问题"
  bash scripts/setup_and_run_dflash2_m5.sh doctor

Model input:
  TARGET_REPO=FlagRelease/Qwen3.8-27B-W4A8-arm-FlagOS-Express
  DFLASH2_REPO=incoai/Qwen3.8-27B-DFlash2

Or point to existing directories:
  MODEL_DIR=/path/to/target DFLASH2_DIR=/path/to/drafter

The defaults above are ModelScope repository IDs. The target contains the
packed W4A8 artifact used by this runtime; the original BF16 Qwen repository
is not a drop-in replacement.

The installer logic is this script. On first environment setup, it downloads a
versioned asset bundle containing pinned source snapshots, reviewed patches,
dependency locks, checksums, and Python launch helpers. Model weights are not
part of that bundle.

Useful overrides:
  DFLASH2_INSTALL_ROOT=/path/to/runtime
  MAX_TOKENS=256 ENABLE_THINKING=1 TEMPERATURE=0
  ALLOW_OTHER_APPLE_HOST=1   # builds, but does not claim the M5 profile result
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

host_check() {
  [[ "$(uname -s)" == "Darwin" ]] || die "this route requires macOS"
  [[ "$(uname -m)" == "arm64" ]] || die "this route requires Apple Silicon"
  local host_model
  host_model="$(sysctl -n hw.model 2>/dev/null || true)"
  if [[ "$host_model" != "Mac17,9" && "${ALLOW_OTHER_APPLE_HOST:-0}" != "1" ]]; then
    die "the published profile targets Mac17,9; detected ${host_model:-unknown}"
  fi
}

model_is_complete() {
  local model_dir="$1"
  /usr/bin/python3 - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not (root / "config.json").is_file():
    raise SystemExit(1)
indexes = sorted(root.glob("*.safetensors.index.json"))
if indexes:
    try:
        payload = json.loads(indexes[0].read_text())
        names = set(payload["weight_map"].values())
    except Exception:
        raise SystemExit(1)
    if not names or any(
        not (root / name).is_file() or (root / name).stat().st_size == 0
        for name in names
    ):
        raise SystemExit(1)
else:
    weights = list(root.glob("*.safetensors"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise SystemExit(1)
PY
}

target_is_complete() {
  model_is_complete "$1" || return 1
  [[ -f "$1/qwen38_w4a8_gptq_packed_manifest.json" ]] || return 1
  [[ -f "$1/model.safetensors.index.json" ]] || return 1
  /usr/bin/python3 - "$1" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
group = config.get("quantization_config", {}).get("config_groups", {}).get("group_0", {})
weight = group.get("weights", {})
activation = group.get("input_activations", {})
valid = (
    config.get("model_type") == "qwen3_5"
    and weight.get("num_bits") == 4
    and weight.get("group_size") == 128
    and activation.get("num_bits") == 8
)
if valid and os.environ.get("ALLOW_UNVERIFIED_MODEL", "0") != "1":
    expected_hashes = {
        "config.json": "71c90079f670b524736815a7e2f4f53c9f00b92e14214284398c548e64bc1ad7",
        "model.safetensors.index.json": "142cb31670ad6f3d641406c08a5ca0357251265a3e4a1ae3894c59b6b9dd9d26",
        "qwen38_w4a8_gptq_packed_manifest.json": "785733fb737f4c45028892d594108b71bb14c25a6b86975b8baa8b0dc5b4d354",
    }
    valid = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in expected_hashes.items()
    )
    expected_sizes = {
        "model-00001-of-00009.safetensors": 2542796928,
        "model-00002-of-00009.safetensors": 2542796952,
        "model-00003-of-00009.safetensors": 1992952728,
        "model-00004-of-00009.safetensors": 1992271184,
        "model-00005-of-00009.safetensors": 1965571872,
        "model-00006-of-00009.safetensors": 1959829640,
        "model-00007-of-00009.safetensors": 1965571872,
        "model-00008-of-00009.safetensors": 1959829640,
        "model-00009-of-00009.safetensors": 1646765560,
        "model-mtp.safetensors": 849400424,
    }
    valid = valid and all(
        (root / name).is_file() and (root / name).stat().st_size == size
        for name, size in expected_sizes.items()
    )
raise SystemExit(0 if valid else 1)
PY
}

draft_is_complete() {
  model_is_complete "$1" || return 1
  /usr/bin/python3 - "$1" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
dflash = config.get("dflash_config", {})
valid = (
    "DFlash2DraftModel" in config.get("architectures", [])
    and dflash.get("block_size") == 8
    and dflash.get("selector_top_k") == 16
    and dflash.get("selector_rank") == 256
    and dflash.get("target_layer_ids") == [5, 19, 33, 47, 61]
    and config.get("num_hidden_layers") == 5
    and config.get("num_target_layers") == 64
)
if valid and os.environ.get("ALLOW_UNVERIFIED_MODEL", "0") != "1":
    valid = (
        hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
        == "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980"
    )
    model = root / "model.safetensors"
    if valid and model.is_file() and model.stat().st_size == 3848817896:
        digest = hashlib.sha256()
        with model.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        valid = digest.hexdigest() == "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
    else:
        valid = False
raise SystemExit(0 if valid else 1)
PY
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi
  command -v brew >/dev/null 2>&1 || die "install Homebrew first: https://brew.sh"
  brew install uv
  command -v uv
}

editable_install_matches() {
  local distribution_name="$1" source_dir="$2" expected_version="$3"
  EDITABLE_DIST="$distribution_name" EDITABLE_SOURCE="$source_dir" \
  EDITABLE_VERSION="$expected_version" "$python_bin" - <<'PY'
import json
import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

try:
    dist = distribution(os.environ["EDITABLE_DIST"])
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
except (PackageNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
expected_url = Path(os.environ["EDITABLE_SOURCE"]).resolve().as_uri()
valid = (
    dist.version == os.environ["EDITABLE_VERSION"]
    and direct_url.get("url") == expected_url
    and direct_url.get("dir_info", {}).get("editable") is True
)
raise SystemExit(0 if valid else 1)
PY
}

download_model() {
  local label="$1" repo="$2" revision="$3" output_dir="$4" validator="$5"
  if "$validator" "$output_dir"; then
    echo "$label already present: $output_dir"
    return
  fi
  [[ -n "$repo" ]] || die "$label is missing; set its repository ID or local directory"
  local uv_cmd
  uv_cmd="$(ensure_uv)"
  mkdir -p "$output_dir"
  echo "Downloading $label from $repo"
  "$uv_cmd" tool run --from modelscope modelscope download "$repo" \
    --revision "$revision" \
    --local-dir "$output_dir"
  "$validator" "$output_dir" || die "$label download is incomplete or has the wrong format: $output_dir"
}

install_models() {
  download_model "Qwen3.8-27B W4A8 target" "$target_repo" "$target_revision" "$target_dir" target_is_complete
  download_model "Qwen3.8-27B DFlash2 drafter" "$draft_repo" "$draft_revision" "$draft_dir" draft_is_complete
  echo "Models ready:"
  echo "  target  $target_dir"
  echo "  drafter $draft_dir"
}

install_build_tools() {
  xcode-select -p >/dev/null 2>&1 || die "install Xcode Command Line Tools with: xcode-select --install"
  command -v brew >/dev/null 2>&1 || die "install Homebrew first: https://brew.sh"
  local formula missing=()
  for formula in python@3.11 cmake ninja libomp uv; do
    brew list --versions "$formula" >/dev/null 2>&1 || missing+=("$formula")
  done
  if ((${#missing[@]})); then
    HOMEBREW_NO_AUTO_UPDATE=1 brew install "${missing[@]}"
  else
    echo "Build tools already installed."
  fi
}

source_tree_complete() {
  local root="$1"
  [[ -f "$root/vllm-0.24.0/pyproject.toml" \
      && -f "$root/vllm-plugin-FL-vllm024/vllm_fl/spec_decode/dflash2.py" \
      && -f "$root/active-triton-source/triton-cpu-3.7.2/setup.py" \
      && -f "$root/FlagGems/src/flag_gems/csrc/arm/q4_g128_m8n4_i8mm_bf16.S" \
      && -f "$root/FlagGems/src/flag_gems/csrc/arm/gdn_op.cpp" \
      && -f "$root/KleidiAI-v1.29.0/CMakeLists.txt" \
      && -f "$root/libtriton_jit/CMakeLists.txt" \
      && -f "$root/flagos-macos-runtime/profiles/m5-dflash2-k7.env" ]]
}

verify_runtime_assets() {
  [[ -f "$runtime_assets_file" ]] || die "missing runtime asset manifest: $runtime_assets_file"
  echo "Verifying pinned runtime source assets..."
  (cd "$assets_root" && shasum -a 256 -c "metadata/m5-dflash2-runtime-assets.sha256") \
    || die "runtime source asset checksum verification failed"
}

extract_pinned_dependency() {
  local label="$1" archive="$2" destination="$3" required_file="$4"
  if [[ -f "$destination/$required_file" ]]; then
    return
  fi
  [[ ! -e "$destination" ]] \
    || die "incomplete $label source found at $destination; move it aside and rerun"
  local staging="$install_root/.dependency-staging-${label//[^A-Za-z0-9]/-}-$$"
  mkdir -p "$staging"
  tar -xf "$archive" -C "$staging" --strip-components=1
  [[ -f "$staging/$required_file" ]] \
    || die "failed to restore pinned $label source from $archive"
  mkdir -p "$(dirname "$destination")"
  mv "$staging" "$destination"
}

prepare_libtriton_dependencies() {
  extract_pinned_dependency \
    "nlohmann-json-3.11.3" "$json_archive" "$json_source" \
    "include/nlohmann/json.hpp"
  extract_pinned_dependency \
    "fmt-11.1.4" "$fmt_archive" "$fmt_source" \
    "include/fmt/base.h"

  local json_digest fmt_version
  json_digest="$(shasum -a 256 "$json_source/include/nlohmann/json.hpp" | awk '{print $1}')"
  [[ "$json_digest" == "2432cbf7f4258453538eca193cd937fc777462ae2ce8ee9f46e6ee460ef56bea" ]] \
    || die "restored nlohmann/json header does not match pinned v3.11.3"
  fmt_version="$(awk '/^#define FMT_VERSION / {print $3; exit}' "$fmt_source/include/fmt/base.h")"
  [[ "$fmt_version" == "110104" ]] \
    || die "restored fmt header does not match pinned v11.1.4"
}

restore_source_tree() {
  if [[ -f "$source_root/.dflash2-production-20260827" ]]; then
    source_tree_complete "$source_root" \
      || die "installed source tree is incomplete: $source_root"
    return
  fi
  [[ ! -e "$source_root" ]] || die "incomplete source tree found at $source_root; move it aside and rerun"
  mkdir -p "$install_root"
  local staging_root="$install_root/.sources-staging-$$"
  mkdir "$staging_root"
  tar -xzf "$assets_root/sources/vllm-0.24.0-ee0da84ab.tar.gz" -C "$staging_root"
  tar -xzf "$assets_root/sources/vllm-plugin-FL-vllm024-base-a9435a34.tar.gz" -C "$staging_root"
  tar -xzf "$assets_root/sources/FlagGems-base-7e329acb3.tar.gz" -C "$staging_root"
  tar -xzf "$assets_root/sources/KleidiAI-v1.29.0-13cd3599.tar.gz" -C "$staging_root"
  tar -xzf "$assets_root/sources/libtriton_jit-a4eb4db99.tar.gz" -C "$staging_root"
  tar -xzf "$assets_root/sources/flagos-macos-runtime-5f21d7272.tar.gz" -C "$staging_root"
  patch -s -p1 -d "$staging_root/vllm-plugin-FL-vllm024" \
    < "$assets_root/patches/vllm-plugin-FL-vllm024-working-tree.patch"
  tar -xzf "$assets_root/patches/vllm-plugin-FL-vllm024-untracked-files.tar.gz" \
    -C "$staging_root/vllm-plugin-FL-vllm024"
  mkdir "$staging_root/active-triton-source"
  tar -xzf "$assets_root/sources/triton-cpu-active-editable-source.tar.gz" \
    -C "$staging_root/active-triton-source"
  patch -s -p1 -d "$staging_root/FlagGems" \
    < "$assets_root/patches/FlagGems-working-tree.patch"
  patch -s -p1 -d "$staging_root/flagos-macos-runtime" \
    < "$assets_root/patches/flagos-macos-runtime-working-tree.patch"
  tar -xzf "$assets_root/patches/production-overlay-20260827.tar.gz" \
    -C "$staging_root"
  source_tree_complete "$staging_root" || die "restored runtime source tree is incomplete"
  touch "$staging_root/.dflash2-production-20260827"
  mv "$staging_root" "$source_root"
}

install_environment() {
  host_check
  install_build_tools
  verify_runtime_assets
  restore_source_tree
  prepare_libtriton_dependencies

  local uv_cmd ncpu triton_source build_root brew_prefix cmake_bin ninja_bin python311 libomp_root
  uv_cmd="$(ensure_uv)"
  ncpu="$(sysctl -n hw.ncpu)"
  triton_source="$source_root/active-triton-source/triton-cpu-3.7.2"
  build_root="$runtime_workspace/build"
  brew_prefix="$(brew --prefix)"
  cmake_bin="$brew_prefix/bin/cmake"
  ninja_bin="$brew_prefix/bin/ninja"
  python311="$brew_prefix/bin/python3.11"
  libomp_root="$brew_prefix/opt/libomp"

  mkdir -p "$install_root"
  [[ -f "$constraints_file" ]] || die "missing Python constraints: $constraints_file"
  touch "$install_log"
  echo >> "$install_log"
  echo "===== environment install $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$install_log"
  echo "Install log: $install_log"

  if [[ ! -x "$python_bin" ]]; then
    "$uv_cmd" venv --python "$python311" "$vllm_source/.venv" 2>&1 | tee -a "$install_log"
  fi

  "$uv_cmd" pip install --python "$python_bin" \
    -r "$vllm_source/requirements/cpu.txt" \
    --constraint "$constraints_file" \
    --index-strategy unsafe-best-match 2>&1 | tee -a "$install_log"
  "$uv_cmd" pip install --python "$python_bin" \
    'cmake==4.4.2' 'ninja==1.13.0' 'pybind11==3.0.3' \
    'setuptools==77.0.3' 'setuptools-scm==9.2.0' \
    'setuptools-rust==1.13.0' 'scikit-build-core==0.12.2' \
    'sqlalchemy==2.0.48' \
    --constraint "$constraints_file" 2>&1 | tee -a "$install_log"

  if TRITON_EXPECTED_SOURCE="$triton_source" "$python_bin" - <<'PY'
import os
from pathlib import Path
import triton

actual = Path(triton.__file__).resolve()
expected = Path(os.environ["TRITON_EXPECTED_SOURCE"]).resolve()
raise SystemExit(0 if triton.__version__ == "3.7.2" and expected in actual.parents else 1)
PY
  then
    echo "Reusing flagtree-cpu/Triton 3.7.2: $triton_source" | tee -a "$install_log"
  else
    TRITON_HOME="$install_root/triton" \
    TRITON_BUILD_PROTON=OFF MAX_JOBS="$ncpu" \
      "$uv_cmd" pip install --python "$python_bin" --no-build-isolation \
        --constraint "$constraints_file" \
        --editable "$triton_source" 2>&1 | tee -a "$install_log"
  fi

  if editable_install_matches vllm "$vllm_source" 0.24.0+cpu; then
    echo "Reusing vLLM CPU editable install: $vllm_source" | tee -a "$install_log"
  else
    VLLM_TARGET_DEVICE=cpu VLLM_VERSION_OVERRIDE=0.24.0+cpu MAX_JOBS="$ncpu" \
      "$uv_cmd" pip install --python "$python_bin" --no-build-isolation \
        --constraint "$constraints_file" \
        --editable "$vllm_source" 2>&1 | tee -a "$install_log"
  fi
  if editable_install_matches vllm-plugin-fl "$plugin_source" 0.0.0; then
    echo "Reusing vllm-plugin-FL editable install: $plugin_source" | tee -a "$install_log"
  else
    "$uv_cmd" pip install --python "$python_bin" --no-build-isolation \
      --constraint "$constraints_file" \
      --editable "$plugin_source" 2>&1 | tee -a "$install_log"
  fi

  "$cmake_bin" \
    -S "$runtime_workspace/libtriton_jit" \
    -B "$build_root/libtriton-jit" \
    -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$ninja_bin" \
    -DBACKEND=CPU \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$python_bin" \
    -DOpenMP_ROOT="$libomp_root" \
    -DTRITON_JIT_FMT_TAG=11.1.4 \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_JSON="$json_source" \
    -DFETCHCONTENT_SOURCE_DIR_FMT="$fmt_source" \
    -DTRITON_JIT_BUILD_OPERATORS=OFF \
    -DTRITON_JIT_INSTALL=ON 2>&1 | tee -a "$install_log"
  "$cmake_bin" --build "$build_root/libtriton-jit" --parallel "$ncpu" 2>&1 | tee -a "$install_log"

  "$cmake_bin" \
    -S "$runtime_workspace/FlagGems/src/flag_gems/csrc/arm" \
    -B "$build_root/flaggems-arm-m5" \
    -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$ninja_bin" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$python_bin" \
    -DOpenMP_ROOT="$libomp_root" \
    -DTRITON_JIT_ROOT="$runtime_workspace/libtriton_jit" \
    -DTRITON_JIT_BUILD="$build_root/libtriton-jit" \
    -DFLAGGEMS_KLEIDIAI_ROOT="$runtime_workspace/KleidiAI-v1.29.0" 2>&1 | tee -a "$install_log"
  "$cmake_bin" --build "$build_root/flaggems-arm-m5" --parallel "$ncpu" 2>&1 | tee -a "$install_log"

  environment_doctor
  echo "DFlash2 CPU environment ready: $install_root"
}

environment_doctor() {
  local missing=0 brew_prefix libomp_root
  host_check
  ulimit -n 4096
  brew_prefix="$(brew --prefix)"
  libomp_root="${LIBOMP_ROOT:-$brew_prefix/opt/libomp}"
  for required in "$python_bin" "$flag_gems_lib" "$profile"; do
    if [[ ! -e "$required" ]]; then
      echo "MISSING $required" >&2
      missing=1
    fi
  done
  ((missing == 0)) || return 2
  grep -q 'FLAGGEMS_Q4_G128_KLEIDIAI_M8_NATIVE_THREADS=16' "$profile" || {
    echo "INVALID DFlash2 M5 profile: $profile" >&2
    return 2
  }
  grep -q 'FLAGGEMS_GDN_FUSED_FULL_SPEC_DECODE=1' "$profile" || {
    echo "INCOMPLETE DFlash2 production profile: $profile" >&2
    return 2
  }
  file "$flag_gems_lib" | grep -q 'arm64' || {
    echo "INVALID native library architecture: $flag_gems_lib" >&2
    return 2
  }
  [[ -f "$libomp_root/include/omp.h" && -f "$libomp_root/lib/libomp.dylib" ]] || {
    echo "INVALID libomp installation: $libomp_root" >&2
    return 2
  }
  (
    export FLAGOS_RUNTIME_ROOT="$runtime_workspace/flagos-macos-runtime"
    set -a
    source "$profile"
    set +a
    export TRITON_LOCAL_LIBOMP_PATH="$libomp_root"
    export FLAGGEMS_LIBTRITON_JIT_Q4_OP="$flag_gems_lib"
    export FLAGGEMS_Q4_KERNEL_SOURCE="$runtime_workspace/FlagGems/src/flag_gems/csrc/arm/q4_kernels.py"
    export FLAGGEMS_W8_KERNEL_SOURCE="$runtime_workspace/FlagGems/src/flag_gems/csrc/arm/w8_kernels.py"
    PYTHONPATH="$plugin_source:$runtime_workspace/FlagGems/src:$runtime_workspace/libtriton_jit/scripts" \
      "$python_bin" - <<'PY'
from importlib.metadata import version
import multiprocessing
import platform
import threading

multiprocessing.Lock = threading.Lock

import torch
import triton
import vllm
import vllm_fl
from vllm.plugins import load_general_plugins

# Exercise the complete vllm-plugin-FL -> FlagGems Qwen/GDN registration.
# A top-level package import alone does not load this runtime dependency chain.
load_general_plugins()

expected = {
    "torch": "2.11.0",
    "transformers": "5.15.1",
    "compressed-tensors": "0.17.0",
    "triton": "3.7.2",
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Python runtime version mismatch: expected={expected}, actual={actual}")
if platform.machine() != "arm64":
    raise SystemExit(f"Python is not arm64: {platform.machine()}")
print("READY", torch.__version__, triton.__version__, vllm.__version__, actual)
PY
  )
  echo "Profile: $profile"
  echo "Native ops: $flag_gems_lib"
}

doctor() {
  target_is_complete "$target_dir" || die "invalid or missing W4A8 packed target: $target_dir"
  draft_is_complete "$draft_dir" || die "invalid or missing DFlash2 drafter: $draft_dir"
  environment_doctor
}

export_runtime() {
  local brew_prefix
  brew_prefix="$(brew --prefix)"
  export MODEL_DIR="$target_dir"
  export DFLASH2_DIR="$draft_dir"
  export WORKSPACE_ROOT="$runtime_workspace"
  export RUNTIME_WORKSPACE="$runtime_workspace"
  export FL_PLUGIN_SOURCE_ROOT="$plugin_source"
  export PYTHON_BIN="$python_bin"
  export FLAGOS_RUNTIME_PROFILE="$profile"
  export FLAGGEMS_LIBTRITON_JIT_Q4_OP="$flag_gems_lib"
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  export FLAGGEMS_MTP_W8=1
  export TRITON_HOME="${TRITON_HOME:-$install_root/triton}"
  export LIBOMP_ROOT="${LIBOMP_ROOT:-$brew_prefix/opt/libomp}"
}

run_inference() {
  doctor
  export_runtime
  source "$runtime_payload_root/portable_env.sh"
  exec "$python_bin" "$runtime_payload_root/qwen38_dflash2_generate.py" "$@"
}

run_benchmark() {
  doctor
  export_runtime
  export DFLASH2_THREADS="${DFLASH2_THREADS:-14}"
  export BENCH_Q4_PREFILL_THREADS_OVERRIDE="${BENCH_Q4_PREFILL_THREADS_OVERRIDE:-18}"
  export BENCH_GDN_PREFILL_THREADS_OVERRIDE="${BENCH_GDN_PREFILL_THREADS_OVERRIDE:-18}"
  export BENCH_TOKENS="${BENCH_TOKENS:-325}"
  export BENCH_REPEATS="${BENCH_REPEATS:-3}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((BENCH_TOKENS + 99))}"
  export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((BENCH_TOKENS + 107))}"
  mkdir -p "$install_root/results"
  export BENCH_RESULT_FILE="${BENCH_RESULT_FILE:-$install_root/results/m5_setup_entry_pp85_tg${BENCH_TOKENS}.json}"
  exec bash "$runtime_payload_root/run_dflash2_benchmark_vllm024.sh"
}

command_name="${1:-}"
if [[ -z "$command_name" ]]; then
  usage
  exit 2
fi
shift
case "$command_name" in
  models)
    install_models
    ;;
  env)
    prepare_runtime_bundle
    install_environment
    ;;
  run)
    prepare_runtime_bundle
    run_inference "$@"
    ;;
  benchmark)
    prepare_runtime_bundle
    run_benchmark
    ;;
  all)
    install_models
    prepare_runtime_bundle
    install_environment
    run_inference "$@"
    ;;
  doctor)
    prepare_runtime_bundle
    doctor
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    die "unknown command: $command_name"
    ;;
esac
