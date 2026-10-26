#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${1:-${REPO_ROOT}/datasets/OVO-Bench}"
PART_PATTERN="${2:-src_videos.tar.part*}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "[ERROR] Directory not found: $TARGET_DIR" >&2
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "[ERROR] 'tar' is not installed. Please install it first." >&2
  exit 1
fi

shopt -s nullglob
part_files=("$TARGET_DIR"/$PART_PATTERN)
shopt -u nullglob

if (( ${#part_files[@]} == 0 )); then
  echo "[INFO] No files matched: $TARGET_DIR/$PART_PATTERN"
  exit 0
fi

printf '[INFO] Found %d split parts.\n' "${#part_files[@]}"

# Support names like: src_videos.tar.partaa, src_videos.tar.part01.
declare -A archive_roots=()
for part in "${part_files[@]}"; do
  name="$(basename "$part")"
  if [[ "$name" =~ ^(.+\.tar)\.part[[:alnum:]]+$ ]]; then
    archive_roots["${BASH_REMATCH[1]}"]=1
  fi
done

if (( ${#archive_roots[@]} == 0 )); then
  echo "[ERROR] Matched files exist, but no valid split tar parts were recognized." >&2
  exit 1
fi

for archive_root in "${!archive_roots[@]}"; do
  shopt -s nullglob
  parts=("$TARGET_DIR"/"$archive_root".part*)
  shopt -u nullglob

  if (( ${#parts[@]} == 0 )); then
    echo "[WARN] No parts found for root: $archive_root"
    continue
  fi

  printf '[INFO] Extracting %s from %d parts...\n' "$archive_root" "${#parts[@]}"
  cat "${parts[@]}" | tar -xf - -C "$TARGET_DIR"
done

echo "[INFO] All done."
