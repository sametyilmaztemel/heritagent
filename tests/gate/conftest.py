"""Shared fixtures/helpers for CIG gate orchestrator tests (issue #12) —
single source of truth in tests.gate.helpers; re-exported here so existing
``from tests.gate.conftest import ...`` call sites keep working."""

from tests.gate.helpers import (  # noqa: F401
    BUDGETS,
    MINING_SEED,
    PARENT_CIG,
    ScriptedGateRunner,
    interaction_tasks,
    mint_envelope,
    proposal,
    stage12_evidence,
)
