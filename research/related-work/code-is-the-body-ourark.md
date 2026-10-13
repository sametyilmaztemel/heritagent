# Code Is the Body (OurArk / Genesis / Enoch): detailed notes

**Paper:** *Code Is the Body: Agent-Owned Software Bodies for Recursive Evolution and Descent* by Roy Zhao (Paul G. Allen School of CSE, University of Washington) and Zhenyu Zhao (independent researcher). arXiv:2607.28691v1 [cs.SE, cross-list cs.AI], submitted 30 Jul 2026, 9 pages, 3 figures, "An OurArk Research Publication". Code (Apache-2.0): github.com/our-ark/genesis (v0.1.1, commit e7bb896) and github.com/our-ark/enoch (v0.3.1, commit 1021e1d). I could not open the repos (HTTP 403 through the proxy), so the code is unverified. Reviewed 2026-10-12 from the full arXiv HTML text (all sections, tables and references).

*Author-count note:* the arXiv PDF-button text says "Roy Zhao (1) and 3 other authors". The `citation_author` metadata and the paper header both list exactly **two** authors. The "3 others" string probably comes from arXiv splitting the affiliation string "(1) …, (2) …" on commas. Cite the paper with two authors.

## Mechanism

- **Agent = (B, S, R) under a human custodian H.** *B*, the body, is a standalone versioned Git repository. It holds the identity and mission manifest, runtime and interfaces, model and memory adapters, skills and action policies, config schemas and exclusion boundaries, tests, and the code for updating, inheriting, learning and evolving. *S*, the private state, holds credentials, memories, logs and queues; it is **not inheritable**. *R*, the external reasoner, is a replaceable inference service, and **model weights are not inherited**. *H* holds custody and the authority to promote changes.
- **Three operations:**
  - `evolve(B,c)→B′`: B′ becomes authoritative only after validation and promotion.
  - `descend(B,m)→(B_d,S_d)`: creates a new identity and mission, its own version history and a fresh private-state boundary.
  - `transfer(B_p,B_d,Δ)→B′_d`: lets a descendant inspect and locally adapt a selected parent or peer change.
- **Genesis (descent engine):** the steps are:
  1. Verify that the ancestor is clean and tracked, and resolve its declared manifest of tracked body files and pinned dependencies.
  2. Copy the body and rewrite identity-bearing text with boundary-aware string substitution.
  3. Write the identity, generation and immutable parent-at-birth provenance.
  4. Initialise an independent repository.
  5. **Run the inherited test suite and reject a failed or dirty validation.**
  6. **Create the birth commit only after validation passes.**

  Any descendant that Genesis can process ("compatible") can itself be a source. There is no privileged root template, and the authors explicitly do not regenerate agents "from a central schema".
- **Governed body evolution (Enoch):** candidate changes come from six sources: user request, human feedback, joint-work experience, direct-parent inheritance, peer learning, and LLM brainstorming. "Feedback and experience are derived from prior conversation turns and work-event histories and may be converted into candidate improvements." Each change runs in an isolated worktree, passes validation, and goes up as a pull request. **A human decides the merge.** Rejected work leaves the authoritative body unchanged. If an update fails its post-update health checks, the agent rolls back.
- **Three kinds of transfer:**
  - *Descent* copies the body once.
  - *Inheritance* is pull-based and selective. A descendant asks its direct parent for recent change candidates and "may adapt one through its own branch, tests, and review". Parents never push changes.
  - *Learning* imports a published skill from any trusted agent. It is lateral and is not ancestry.

## Evidence

- One linear, co-evolved lineage: Lucy → Adam → Seth → Enoch, i.e. 4 agents over generations 0–3, with 3 descents. Each generation adds human- and agent-built mechanisms: exist → operate/inherit → work → evolve. The authors say the historical Lucy-to-Enoch lineage *predates* the provenance format. Only two successive descents are verified by tests.
- Regression results: Genesis 29/29 tests, Enoch body 753/753 tests on CPython 3.11–3.14, and one Genesis × Enoch cross-artifact descent check, which passed.
- **The evidence is about mechanisms only.** Tests "mostly use mocked services". Model calls are mocked, and model choice "was neither controlled nor evaluated". The authors do not report:
  - any benchmark or task-performance number;
  - any ablation;
  - any comparison between gated and ungated inheritance;
  - a sibling or population-level lineage;
  - behavioural continuity across reasoner replacement (stated as unmeasured).
- **Their own status table rates post-divergence transfer as "partial":** "no controlled end-to-end adaptation has been run".
- Stated scope: "Our claims are therefore limited to architecture, implemented mechanisms, and regression evidence."

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Lifetime-trajectory acquisition | **Partial (weak).** Feedback and joint-work experience from conversation and work-event histories "may be converted into candidate improvements", but the conversion is human-in-the-loop and not evaluated. There is no automated mining of capabilities from trajectories and no success/failure distillation. Most generation-to-generation changes in the demo were built by the developers and coding agents. |
| 2. Somatic/germline separation | **Partial: a functional analogue of the boundary, not the lifecycle.** The inheritable body *B* is explicitly separate from non-inheritable private state *S* at descent, and tests check the boundary. This is the closest prior to a "what is heritable vs what belongs to one instance" split that we have seen for LLM agents. The paper does not use soma/germline terms. **What it lacks:** *S* (memories, credentials, logs) is not a staging store for candidate traits, and nothing in *S* is ever promoted into *B*. Lifetime body changes are merged directly into the agent's own authoritative body, and that same body is what descendants copy. So there is no separate germline: the working body is the heritable body. |
| 3. Typed executable genome | **Partial.** The inherited unit is executable, versioned and inspectable, with immutable provenance, a machine-readable identity/mission declaration, a declared file manifest and pinned dependencies. But the body is an **untyped repository**: it has no trait schema, no typed records for policies, skills and regulation, and no regulation layer for when a trait is expressed. The authors reject a central schema. This sits in the same class as the DGM/MGM code genomes, with a better provenance and identity layer. |
| 4. Causal Inheritance Gate | **Absent.** There are two gates, and neither is causal. (a) At birth: the inherited regression suite runs on the whole copied body, and a dirty or failed run blocks the birth commit. (b) For evolution and transfer: validation plus a human merge decision. Neither involves replay against counterfactuals, controlled ablation, per-trait attribution or held-out generalisation. The test suite is a regression check, not trait-level causal evidence. Note for writing: the ledger's "promotion gates exist but governance-flavored (LOGOS)" line also covers this paper. |
| 5. Selective assimilation | **Partial.** Candidate changes are isolated, validated, reviewed and either merged or rejected without touching the authoritative body. After divergence, parent changes are pulled one at a time and adapted, never auto-applied. Selection is real but it is **human judgement plus regression tests**, not evidence of a trait's contribution. The authors themselves rate the end-to-end post-divergence adoption path "partial" and have not run it. |
| 6. Multi-generational, frozen weights | **Full (as a mechanism).** Weights are never trained or inherited, inference is an external replaceable service, and recursive descent through ≥2 generations is implemented and tested. Caveats: one linear lineage; the reasoner was neither held fixed nor controlled; nothing was measured across generations. |

**Verdict:** this paper does **not** implement the full chain. It is the closest architectural prior found so far for the *heritable-vs-instance-state boundary* and for *validated, selective, pull-based inheritance under frozen weights*. It has no causal validation, no typed genome and no measurement of heritability. The sweep's "high" threat rating (≥4 elements including selective assimilation) holds by count. Element by element, though, only element 6 is full; elements 2, 3 and 5 are partial and element 4 is absent.

## Limitations (stated)

- One linear lineage from one team, with no siblings and no population.
- Mocked services; no evaluation of proposal or specialisation quality, review burden or semantic safety.
- String-substitution identity rewriting is not guaranteed to be semantically correct.
- Post-divergence transfer has not been evaluated end to end.
- Lineage records are not tamper-evident; protected-scope checks are procedural.
- The executor has broader filesystem authority than a production sandbox should have.
- Forge-hosted repositories and OurArk naming conventions are assumed.

**Unstated:** the paper never measures whether inherited bodies make descendants better on any task. Generational "capability transitions" were engineered by people, not acquired and selected.

## Consequences for HeritAgent

- **Must-not-claim additions (for ledger §3):**
  - An inheritable-artifact vs private-instance-state boundary for LLM agents. OurArk has one, under different vocabulary, so we cannot say "somatic/germline separation has zero prior" without qualification. The ledger §1 row "Somatic vs germline store separation: Open, zero direct hits" needs softening to: *boundary has a functional analogue (OurArk 2607.28691); a somatic **staging store** whose contents are promoted only via a gate remains open*.
  - Validation before descent ("birth only after the inherited tests pass").
  - Selective, pull-based post-divergence adoption of parent changes.
  - Recursive descent with fresh private state and non-inherited weights.
- **Where HeritAgent still differs, and must show it empirically:**
  1. Soma is a quarantined candidate store, and promotion to germline is an explicit, gated event. In OurArk, body edits are heritable as soon as they are merged.
  2. CIG supplies trait-level interventional evidence (ablation, attribution, held-out generalisation, regression protection) where OurArk uses a whole-body regression suite plus human review.
  3. The EAG is a typed schema of policies, skills and regulation, not an untyped repository.
  4. Heritability is *measured* (gated vs unrestricted arms, descendant adaptation cost, regression rate). OurArk's evidence covers mechanisms only.
- **Positioning sentence for writing:** "OurArk [2607.28691] separates an inheritable software body from private instance state and validates descendants against inherited regression tests before birth. HeritAgent instead gates *which lifetime-acquired traits* enter the heritable genome on trait-level causal evidence, and measures the effect across generations."
- **Possibly useful for the design:**
  - Their birth-time inherited-contract check, i.e. a descendant must pass its parent's tests before birth, is a cheap complement to CIG stage 5 (regression protection). Cite it if we adopt it.
  - Their immutable parent-at-birth/birth-commit provenance is a good template for EAG version lineage records.
- **Follow-ups the paper points to, not yet verified by us:**
  - MOSS (arXiv:2605.22794): promotion through replay, user consent and rollback.
  - Autogenesis (arXiv:2604.15034): versioned resources with propose-assess-commit and lineage.
  - Self-Harness (arXiv:2606.09498): regression-tested harness edits.
  - Agent libOS (arXiv:2606.03895): parent-child lineage and approval queues.
  - EvoGit (arXiv:2506.02049).

  Each is a possible gate or lineage prior that should get a sweep check before submission.
