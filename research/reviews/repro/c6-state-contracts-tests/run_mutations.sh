#!/bin/sh
# Re-run each mutation copy's full test suite. A mutation that still yields
# "438 passed" is NOT detected by the PR's tests.
B=<scratch>
for n in rank_only_harmful no_ablation_preflight overhead_not_gating gen_not_in_eligibility regulation_dropped_in_gate add_bundle_nonatomic; do
  printf '%s: ' "$n"
  (cd "$B/out/repro/mut/$n" && PYTHONPATH=. "$B/venv314/bin/python" -m pytest -q -p no:cacheprovider 2>&1 | tail -1)
done
