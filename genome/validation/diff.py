"""Deterministic parent -> child genome diff (SPEC section 13 lineage edges).

Produces the typed edge record used by the future lineage graph:
``{parent, child, inheritance[], assimilated_traits[], mutations[], fitness_delta}``.
All lists are sorted so the diff of a given (parent, child) pair is stable.
Fitness is never stored in the genome (ADR-0001 D7); it may be attached to
the edge from the evaluation store via the optional ``fitness_delta``.
"""

from __future__ import annotations

SLOT_SECTIONS = ("cognition", "memory", "execution")


def _slot_refs(genome: dict) -> dict[str, dict]:
    refs: dict[str, dict] = {}
    genes = genome.get("genes", {})
    for section in SLOT_SECTIONS:
        for slot, ref in (genes.get(section) or {}).items():
            refs[f"{section}.{slot}"] = ref
    for ref in genes.get("skills") or []:
        refs[f"skills.{ref['gene_id']}"] = ref
    return refs


def diff_genomes(parent: dict, child: dict, fitness_delta: float | None = None) -> dict:
    """Compute the typed parent->child edge between two validated genomes."""
    if parent.get("genome_id") == child.get("genome_id"):
        raise ValueError("parent and child genome_id must differ")

    parent_refs, child_refs = _slot_refs(parent), _slot_refs(child)
    inheritance: set[str] = set()
    assimilated: set[str] = set()
    mutations: set[str] = set()

    for path, ref in child_refs.items():
        old = parent_refs.get(path)
        if old is None:
            if path.startswith("skills."):
                if ref.get("origin") == "assimilation":
                    assimilated.add(ref["gene_id"])
                else:
                    mutations.add(f"skills+: {ref['gene_id']}@{ref['version']}")
            else:
                mutations.add(f"{path}+: {ref['gene_id']}@{ref['version']}")
        elif old["gene_id"] == ref["gene_id"] and old["version"] == ref["version"] \
                and old["artifact"] == ref["artifact"]:
            inheritance.add(ref["gene_id"])
        elif old["gene_id"] == ref["gene_id"] and old["version"] == ref["version"]:
            # same identity bound to different content — never inheritance
            # (defense in depth; the registry rejects such rebinding per D6)
            old_d = old["artifact"].split("sha256:")[1][:8]
            new_d = ref["artifact"].split("sha256:")[1][:8]
            mutations.add(f"{path}: {ref['gene_id']}@{ref['version']} rebinds sha256:{old_d} -> sha256:{new_d}")
        else:
            mutations.add(f"{path}: {old['gene_id']}@{old['version']} -> {ref['gene_id']}@{ref['version']}")

    for path, ref in parent_refs.items():
        if path not in child_refs:
            label = f"skills-: {ref['gene_id']}@{ref['version']}" if path.startswith("skills.") \
                else f"{path}-: removed {ref['gene_id']}@{ref['version']}"
            mutations.add(label)

    parent_reg, child_reg = parent.get("regulation", {}), child.get("regulation", {})
    for gene_id in sorted(set(parent_reg) | set(child_reg)):
        if gene_id not in parent_reg:
            mutations.add(f"regulation.{gene_id}: added")
        elif gene_id not in child_reg:
            mutations.add(f"regulation.{gene_id}: removed")
        elif parent_reg[gene_id] != child_reg[gene_id]:
            mutations.add(f"regulation.{gene_id}: changed")

    return {
        "parent": parent["genome_id"],
        "child": child["genome_id"],
        "inheritance": sorted(inheritance),
        "assimilated_traits": sorted(assimilated),
        "mutations": sorted(mutations),
        "fitness_delta": fitness_delta,
    }
