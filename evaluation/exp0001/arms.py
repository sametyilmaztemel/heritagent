"""The G1 genomes of EXP-0001's arms (issue #13 Phase A; manifest ``arms``;
ADR-0007 E7, proposed).

- A: G0 unchanged.
- B: G0 plus every frozen candidate of the seed, in frozen rank order.
- C: G0 plus the candidates the CIG promoted (somatic state ``validated``),
  in frozen rank order.

Every added skill is a germline ref with ``origin="assimilation"``, its
source trajectory and the aggregate CIG record of its gate run (arm B
assimilates without regard to that record's verdict, which stays on
record). The G1 genome is generation 1, its parent is G0, and its
``genome_id`` is the canonical content id: origin and provenance are not
functional, so B and C genomes with the same skills share one id. An arm
that adds no skill (the gate promoted nothing, or the miner proposed
nothing) is G0 itself, not a generation-1 record with G0's id as its own
parent."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from genome.validation.identity import compute_genome_id
from genome.validation.loader import load_genome

PROMOTE = "promote"  # the gate verdict that makes the somatic state "validated"


class ArmError(ValueError):
    """An arm genome cannot be built from the gate's results."""


def germline_ref(envelope: Mapping, *, cig_record: str, arm: str, mining_seed: int) -> dict:
    candidate = envelope["candidate"]
    provenance = candidate["provenance"]
    if not provenance.get("source_trajectory"):
        raise ArmError(f"{candidate['gene_id']!r} has no source trajectory")
    return {"gene_id": candidate["gene_id"], "version": candidate["version"], "type": "skill",
            "origin": "assimilation", "artifact": candidate["artifact"],
            "provenance": {"born_generation": provenance["born_generation"],
                           "source_trajectory": provenance["source_trajectory"],
                           "cig_record": cig_record,
                           "notes": f"EXP-0001 arm {arm}, mining seed {mining_seed}"}}


def g1_genome(g0: Mapping, refs: Sequence[dict], registry) -> dict:
    if g0.get("generation") != 0 or g0.get("genome_id") != compute_genome_id(dict(g0)):
        raise ArmError("the founder must be the canonical G0 genome")
    if not refs:  # nothing assimilated: the arm runs G0 unchanged
        return copy.deepcopy(dict(g0))
    genome = copy.deepcopy(dict(g0))
    genome["generation"] = 1
    genome["parent"] = g0["genome_id"]
    genome["genes"]["skills"] = list(genome["genes"].get("skills") or []) + copy.deepcopy(list(refs))
    genome["genome_id"] = compute_genome_id(genome)
    load_genome(genome, registry)  # schema, registry bindings, assimilation provenance
    return genome


def arm_genomes(g0: Mapping, candidates: Sequence[Mapping], verdicts: Mapping[str, str],
                aggregate_ids: Mapping[str, str], registry, mining_seed: int) -> dict:
    """Arms B and C of one mining seed. ``candidates`` are the frozen
    candidate envelopes in rank order; every one needs a verdict and an
    aggregate CIG id."""
    gene_ids = [c["candidate"]["gene_id"] for c in candidates]
    if set(verdicts) != set(gene_ids) or set(aggregate_ids) != set(gene_ids):
        raise ArmError("every frozen candidate needs exactly one verdict and aggregate CIG id")

    def refs(arm: str, keep) -> list[dict]:
        return [germline_ref(c, cig_record=aggregate_ids[c["candidate"]["gene_id"]], arm=arm,
                             mining_seed=mining_seed)
                for c in candidates if keep(c["candidate"]["gene_id"])]

    return {"B": g1_genome(g0, refs("B", lambda gene: True), registry),
            "C": g1_genome(g0, refs("C", lambda gene: verdicts[gene] == PROMOTE), registry)}
