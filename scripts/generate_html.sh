#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/docs"

mkdir -p "$OUTPUT_DIR"

exported_count=0

while IFS= read -r -d '' py_file; do
  rel_path="${py_file#$ROOT_DIR/}"
  out_file="$OUTPUT_DIR/${rel_path%.py}.html"

  mkdir -p "$(dirname "$out_file")"

  echo "Exporting $rel_path -> ${out_file#$ROOT_DIR/}"
  uv run marimo export html "$py_file" -f -o "$out_file"

  exported_count=$((exported_count + 1))
done < <(
  find "$ROOT_DIR" \
    -type d \( -name .git -o -name .venv -o -name __pycache__ -o -name node_modules \) -prune -o \
    -type f -name '*.py' -print0
)

echo "Done. Exported $exported_count Python file(s) to $OUTPUT_DIR"
