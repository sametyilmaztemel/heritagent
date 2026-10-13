#!/bin/sh
B=<scratch>
for d in $B/out/repro/delta-mut/*/; do
  n=$(basename $d)
  printf '%s: ' "$n"
  (cd "$d" && PYTHONPATH=. "$B/venv314/bin/python" -m pytest -q -p no:cacheprovider 2>&1 | tail -1)
done
