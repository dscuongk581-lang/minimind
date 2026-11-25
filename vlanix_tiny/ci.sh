#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
python -m unittest discover -s tests -p 'test_*tiny*.py' -v
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
python -m vlanix_tiny.tokenizer --corpus vlanix_tiny/data/tokenizer_smoke.jsonl --output "$tmp/tokenizer.json"
python -m vlanix_tiny.token_benchmark --output "$tmp/ngc-token-benchmark"
python -m vlanix_tiny.organic
python -m vlanix_tiny.binary_logic
python -m vlanix_tiny.binary_model
python -m vlanix_tiny.binary_growth
python -m vlanix_tiny.binary_homeostasis
python -m vlanix_tiny.guarded_session
python -m vlanix_tiny.ram_migration
python -m vlanix_tiny.state_commit_probe
python -m vlanix_tiny.learning_receipt_probe
python -m vlanix_tiny.math_learning
python -m vlanix_tiny.embedding_conservation
python -m vlanix_tiny.rule_transfer
python -m vlanix_tiny.bit_learning
python -m vlanix_tiny.structured_learning
python -m vlanix_tiny.local_rule_learning
python -m vlanix_tiny.trace_learning
python -m vlanix_tiny.stream_learning
