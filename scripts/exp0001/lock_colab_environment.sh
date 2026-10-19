#!/bin/sh
# Resolve environments/exp0001-colab/requirements.in into requirements.lock,
# and build-requirements.in into build-constraints.txt, with the pinned uv
# (owner decision 2026-10-19). The Colab notebooks install exactly these into
# a virtual environment of the pinned Python. uv keeps every version the
# existing output already pins, so a re-run changes only what an input
# changes. The first resolution (2026-10-19) took packages published before
# 2026-10-16, the day the calibration pilot ran, where an index gave dates
# (the vLLM and PyTorch indexes give none; their packages are held by version
# and hash).
set -eu
cd "$(dirname "$0")/../../environments/exp0001-colab"
want="uv $(sed -n 's/^uv==\([0-9.]*\).*/\1/p' uv-requirements.txt)"
have="$(uv --version | cut -d' ' -f1,2)"
[ "$have" = "$want" ] || { echo "needs $want, found $have" >&2; exit 2; }
uv pip compile requirements.in --python-version "$(cat python-version)" \
  --python-platform x86_64-manylinux_2_35 \
  --extra-index-url https://wheels.vllm.ai/0.29.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match --generate-hashes --emit-index-url \
  -o requirements.lock
uv pip compile build-requirements.in --python-version "$(cat python-version)" \
  --python-platform x86_64-manylinux_2_35 --generate-hashes -o build-constraints.txt
