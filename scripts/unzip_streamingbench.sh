#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${1:-${REPO_ROOT}/datasets/StreamingBench}"
cd "$ROOT"
shopt -s nullglob
for z in "Real-Time Visual Understanding_"*.zip; do
  echo "解压: $z"
  unzip -o -q "$z"
done
echo "完成"
