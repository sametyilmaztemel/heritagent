"""Same failing add_bundle as test_atomic_bundle_failure_leaves_stores_unchanged,
but inspect the store add_bundle was actually called on."""
import tempfile
from pathlib import Path
from genome.validation.records import CigRecordStore
from genome.validation.errors import RecordConsistencyError
from tests.gate.test_gate_orchestrator import build_world
orch, somatic, cig, runner, inputs, registry = build_world(Path(tempfile.mkdtemp()), candidates=("a", "b"))
out = orch.run(inputs)
corrupt = out.results[0].stage4_child
corrupt["measurements"]["candidate_gene_id"] = "borrowed_gene"
scratch = CigRecordStore()
try:
    scratch.add_bundle(out.results[0].aggregate, [corrupt] + [c for c in orch._bundle_children(out.results[0]) if c["cig_id"] != corrupt["cig_id"]])
except RecordConsistencyError: print("raised RecordConsistencyError")
print("scratch after failed add_bundle: aggregates", len(scratch._aggregates), "children", len(scratch._children))
