"""Installed-wheel smoke test (CI `package` job).

Run with ``python -I`` from outside the checkout, after installing the built
wheel into a fresh virtualenv: the packages must import from site-packages
and every module must find its JSON Schemas / prompt template as package
data (modules resolve them relative to ``__file__``).

Usage: python -I scripts/ci/smoke_installed.py <repo-root>
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PACKAGES = ("evaluation", "genome", "inheritance", "runtime", "somatic", "traits", "trajectory")


def main() -> int:
    repo_root = Path(sys.argv[1]).resolve()

    import evaluation.benchmarks.alfworld.splits  # noqa: F401
    import evaluation.exp0001.backends  # noqa: F401
    import evaluation.exp0001.benefit_bank  # noqa: F401
    import evaluation.exp0001.figures  # noqa: F401  (imports without matplotlib)
    import evaluation.exp0001.pipeline  # noqa: F401  (the whole EXP-0001 run stack)
    from evaluation.exp0001.run_rehearsal import load_script
    import evaluation.metrics.final_statistics  # noqa: F401
    import genome
    import inheritance.gate  # noqa: F401  (imports the whole gate stack)
    import somatic.store
    from genome.validation.loader import SCHEMA_DIR, load_genome
    from genome.validation.registry import TraitRegistry
    from runtime.seed_g0 import build_g0_seed_genome, g0_runtime_config
    from traits.miner.records import mining_record_validator

    installed_at = Path(genome.__file__).resolve().parent
    if installed_at.is_relative_to(repo_root):
        print(f"FAIL: genome imported from the checkout ({installed_at}), not the wheel")
        return 1

    site = installed_at.parent
    shipped = sorted(p.relative_to(site).as_posix()
                     for pkg in PACKAGES for p in (site / pkg).glob("**/*.schema.json"))
    expected = sorted(p.relative_to(repo_root).as_posix()
                      for pkg in PACKAGES for p in (repo_root / pkg).glob("**/*.schema.json"))
    missing = sorted(set(expected) - set(shipped))
    if missing:
        print(f"FAIL: schemas missing from the wheel: {missing}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        registry = TraitRegistry(Path(tmp) / "registry")
        load_genome(build_g0_seed_genome(registry), registry)
        g0_runtime_config(registry)
    mining_record_validator()
    load_script()  # the run rehearsal's committed script ships as package data
    somatic.store.SomaticStore.create(None)
    print(f"OK: imported from {site}; {len(shipped)} schemas shipped "
          f"(genome schemas at {SCHEMA_DIR})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
