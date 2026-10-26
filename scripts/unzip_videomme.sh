#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${1:-${REPO_ROOT}/datasets/Video-MME}"
PATTERN="${2:-videos_chunked*.zip}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "[ERROR] Directory not found: $TARGET_DIR" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "[ERROR] 'unzip' is not installed. Please install it first." >&2
  exit 1
fi

shopt -s nullglob
zip_files=("$TARGET_DIR"/$PATTERN)
shopt -u nullglob

if (( ${#zip_files[@]} == 0 )); then
  echo "[INFO] No files matched: $TARGET_DIR/$PATTERN"
  exit 0
fi

printf '[INFO] Found %d zip files.\n' "${#zip_files[@]}"

for zip_file in "${zip_files[@]}"; do
  echo "[INFO] Unzipping: $zip_file"
  unzip -o "$zip_file" -d "$TARGET_DIR"
done

echo "[INFO] All done."
