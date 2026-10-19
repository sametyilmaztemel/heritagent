#!/bin/sh
# Resolve environments/exp0001-colab/requirements.in into requirements.lock
# with the pinned uv (owner decision 2026-10-19). The Colab notebooks install
# exactly this lock into a virtual environment of the pinned Python. The
# cutoff (the start of the day the calibration pilot ran) makes the
# resolution repeatable; the vLLM and PyTorch indexes give no upload dates,
# so their packages are exempt from it and held by version and hash instead.
set -eu
cd "$(dirname "$0")/../../environments/exp0001-colab"
want="uv $(cat uv-version)"
have="$(uv --version | cut -d' ' -f1,2)"
[ "$have" = "$want" ] || { echo "needs $want, found $have" >&2; exit 2; }
uv pip compile requirements.in --python-version "$(cat python-version)" \
  --python-platform x86_64-manylinux_2_35 \
  --extra-index-url https://wheels.vllm.ai/0.29.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match --generate-hashes --emit-index-url \
  --exclude-newer 2026-10-16T00:00:00Z \
  --exclude-newer-package vllm=false --exclude-newer-package torch=false \
  --exclude-newer-package torchvision=false --exclude-newer-package torchaudio=false \
  -o requirements.lock
