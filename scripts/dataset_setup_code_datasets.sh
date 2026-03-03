#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"

mkdir -p "${DATA_DIR}"

download_with_hf() {
  local repo_id="$1"
  local out_dir="$2"

  # Newer CLI: `hf download` (doesn't accept --local-dir-use-symlinks on some versions)
  if ! hf download \
    --repo-type dataset \
    "${repo_id}" \
    --local-dir "${out_dir}" \
    --quiet; then
    hf download \
      --repo-type dataset \
      "${repo_id}" \
      --local-dir "${out_dir}"
  fi
}

download_with_huggingface_cli() {
  local repo_id="$1"
  local out_dir="$2"

  huggingface-cli download \
    --repo-type dataset \
    "${repo_id}" \
    --local-dir "${out_dir}" \
    --local-dir-use-symlinks False \
    --quiet
}

download_with_python() {
  local repo_id="$1"
  local out_dir="$2"

  python3 - <<PY
import os
import sys

repo_id = ${repo_id@Q}
out_dir = ${out_dir@Q}

try:
    from huggingface_hub import snapshot_download
except Exception:
    import subprocess
    print("huggingface_hub not found; installing it with pip...", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"])
    from huggingface_hub import snapshot_download

os.makedirs(out_dir, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=out_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"Downloaded {repo_id} -> {out_dir}")
PY
}

download_dataset() {
  local repo_id="$1"
  local name="$2"
  local out_dir="${DATA_DIR}/${name}"

  echo -e "\nDownloading dataset ${repo_id}..."

  if command -v hf >/dev/null 2>&1; then
    download_with_hf "${repo_id}" "${out_dir}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    download_with_huggingface_cli "${repo_id}" "${out_dir}"
  else
    download_with_python "${repo_id}" "${out_dir}"
  fi
}

download_dataset "sahil2801/CodeAlpaca-20k" "CodeAlpaca-20k"
download_dataset "CodeResearch/Code-Evol-Instruct-OSS" "Code-Evol-Instruct-OSS"

echo -e "\nDone. Datasets are under: ${DATA_DIR}"
