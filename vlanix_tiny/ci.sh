#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
python -m unittest discover -s tests -p 'test_*tiny*.py' -v
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
python -m vlanix_tiny.tokenizer --corpus vlanix_tiny/data/tokenizer_smoke.jsonl --output "$tmp/tokenizer.json"
