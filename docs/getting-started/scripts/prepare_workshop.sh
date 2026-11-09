#!/usr/bin/env bash
# Companion attachment for pr.md. Run on the CIX P1 host as baai.
# Downloads materials only; container installation follows the article.
set -Eeuo pipefail
trap 'printf "Preparation stopped at line %s. Existing downloads are retained.\n" "$LINENO" >&2' ERR
[[ ! -f /.dockerenv ]] || { echo 'Run on the P1 host, outside Docker.' >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || { echo 'Requires Linux aarch64 CIX P1.' >&2; exit 1; }
[[ "$HOME" == /home/baai ]] || { echo 'This article uses the baai account with HOME=/home/baai.' >&2; exit 1; }
for command in python3 git curl tar gzip sha256sum sudo; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
python3 -c 'import sys,venv; assert sys.version_info >= (3,9), "Python >=3.9 required"'
sudo -n docker version --format '{{.Server.Version}}'
export WORK_ROOT="${WORK_ROOT:-/home/baai/workshop1018}"
[[ "$WORK_ROOT" == /* && ! -L "$WORK_ROOT" ]] || { echo 'Use an absolute, non-symlink WORK_ROOT.' >&2; exit 1; }
if [[ -e "$WORK_ROOT" ]]; then
  [[ -f "$WORK_ROOT/.prepare-workshop-owned" ]] || { echo 'Existing directory is not owned by this preparation script; do not overwrite it.' >&2; exit 1; }
  [[ ! -e "$WORK_ROOT/envs" ]] || { echo 'An installed environment already exists; continue at article stage one.' >&2; exit 1; }
else
  mkdir -p "$WORK_ROOT"
  printf 'FlagOS Workshop preparation v1\n' > "$WORK_ROOT/.prepare-workshop-owned"
fi
mkdir -p "$WORK_ROOT/downloads"
cd "$WORK_ROOT"

export ONLINE_RELEASE='https://github.com/kevinzs2048/vllm-plugin-FL/releases/download/cix-p1-online-workshop-20261017-r2'
export KIT=FlagOS-CIX-P1-online-workshop-20261017-r2.tar.gz
curl -fL --retry 3 "$ONLINE_RELEASE/$KIT" -o "downloads/$KIT"
curl -fL --retry 3 "$ONLINE_RELEASE/$KIT.sha256" -o "downloads/$KIT.sha256"
(cd downloads && sha256sum -c "$KIT.sha256")
tar -xzf "downloads/$KIT" -C "$WORK_ROOT"
test -f "$WORK_ROOT/scripts/release.env"
source "$WORK_ROOT/scripts/activate_host.env"

# Material 2/5
source "$WORK_ROOT/scripts/release.env"
sudo docker login harbor.baai.ac.cn
sudo -n docker pull "$CIX_IMAGE_REF"
test "$(sudo -n docker image inspect "$CIX_IMAGE_REF" --format '{{.Id}}')" = "$CIX_IMAGE_ID"
sudo -n docker image inspect "$CIX_IMAGE_REF" \
  --format 'id={{.Id}} repo-digests={{json .RepoDigests}}'

# Material 3/5
python3 -m venv "$WORK_ROOT/download-env"
"$WORK_ROOT/download-env/bin/python" -m pip install --timeout 60 --retries 3 \
  --index-url "${WORKSHOP_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  modelscope==1.40.2
export MODEL_DIR="$WORK_ROOT/models/MiniCPM-V-4.7-1B-W8A8-FlagOS"
"$WORK_ROOT/download-env/bin/modelscope" download \
  --model FlagRelease/MiniCPM-V-4.7-1B-W8A8-arm-FlagOS \
  --revision master \
  --local_dir "$MODEL_DIR"
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
test -f "$MODEL_DIR/tokenizer.json"

# Material 4/5
cd "$WORK_ROOT"
export WHEEL_RELEASE='https://github.com/kevinzs2048/FlagGems/releases/download/cix-p1-workshop-20261016-r1'
export WHEEL_PKG=FlagOS-CIX-P1-prebuilt-cp311-torch211-20261016-r1.tar.gz
curl -fL --retry 3 "$WHEEL_RELEASE/$WHEEL_PKG" -o "downloads/$WHEEL_PKG"
printf '%s  %s\n' \
  a53ea331abca38cc75e2192f8291ac8630b2c06451bd3e0a40cdc9b51ef229d4 \
  "downloads/$WHEEL_PKG" | sha256sum -c -
if [[ ! -e "$WORK_ROOT/prebuilt" ]]; then
  tar -xzf "downloads/$WHEEL_PKG" -C "$WORK_ROOT"
else
  bash "$WORK_ROOT/scripts/check_prebuilt.sh"
fi
ls -lh "$WORK_ROOT/prebuilt/wheels"

# Material 5/5
mkdir -p "$WORK_ROOT/source"
if [[ ! -e "$WORK_ROOT/source/vllm-plugin-FL" ]]; then
  git init -q "$WORK_ROOT/source/vllm-plugin-FL"
  git -C "$WORK_ROOT/source/vllm-plugin-FL" remote add origin \
    https://github.com/kevinzs2048/vllm-plugin-FL.git
fi
test -z "$(git -C "$WORK_ROOT/source/vllm-plugin-FL" status --porcelain)"
git -C "$WORK_ROOT/source/vllm-plugin-FL" fetch --depth 1 origin \
  f0de702335902facfffe625c0690f7a3df7f617d
git -C "$WORK_ROOT/source/vllm-plugin-FL" checkout --detach \
  f0de702335902facfffe625c0690f7a3df7f617d
test "$(git -C "$WORK_ROOT/source/vllm-plugin-FL" rev-parse HEAD)" = \
  f0de702335902facfffe625c0690f7a3df7f617d

# Check all prepared inputs before asking the reader to create a container.
bash "$WORK_ROOT/scripts/check_preloaded.sh"
printf '\nMaterials ready: %s\nContinue with stage one of the article.\n' "$WORK_ROOT"
