# ADR-0008 — EXP-0001 execution on one Colab A100 80 GB: parallel episodes, active run time, run mirror

Date: 2026-10-14 · Status: **ACCEPTED by the owner** (Samet, 2026-10-14, explicit answers in chat: the 96-hour cap counts active run time only; only an A100 80 GB is admitted; episodes run in parallel). On the same day the owner removed Codex from the process: builder-2 now holds the Critic's duties (different-model reviews), and the owner decides. No Codex review is requested. Artifacts: `evaluation/exp0001/` (`ledger.py`, `episodes.py`, `gate_runner.py`, `lifetime.py`, `pipeline.py`, `mirror.py`, `serving.py`, `backends.py`, `pilot_run.py`, `cli.py`), `runtime/model_adapters/openai_server.py`, `notebooks/`, the manifest's `compute_cap.wall_clock_basis` and `run_protocol.execution` lines · Origin: the owner has no local GPU and no budget. The run goes on the owner's Colab Pro compute units.

**Problem.**
- **The only GPU is Colab's.** It is a Colab Pro session with an A100 80 GB, at about 7.5 compute units an hour, and 100 units a month buy about 13 A100 hours (status R: a third-party measurement, 2026-03; Colab's pricing page). A session ends when the browser disconnects or at its maximum lifetime, and the VM's local disk goes with it. Background execution needs Pro+.
- **One episode at a time is too slow.** EXP-0001 runs roughly 4,000–7,000 episodes of up to 60 model calls each. One at a time, that is over 100 hours (status M: about one episode a minute).
- **The wall-clock cap.** ADR-0007 E5 counts calendar time since run start, so the hours between two Colab sessions would count against the 96-hour cap.
- **Crash safety.** It assumed a run directory that outlives the process.

**Decision.**
- **D1 Accelerator (owner).** One A100 80 GB, nothing else. The pilot and the run read `nvidia-smi` before any episode and fail closed on any other GPU or GPU count (a 40 GB A100, an L4, an H100, two GPUs). The ledger's session record names the GPU.
- **D2 Wall-clock basis (owner).** `wall_clock_hours_max` (96) counts active run time: the sum, over the ledger's sessions, of each session's span. A span runs from the process opening the ledger to the session's last record. Time between sessions is not counted. Neither is the time between a crashed session's last record and the crash, which is at most the time between two records.
- **D3 Parallel episodes (owner).** Independent episodes run in worker processes against one `vllm serve` process, which batches their requests:
  - The parent admits every episode before it starts and records it after it ends, so the ledger has one writer.
  - Every cap check counts the worst case of every attempt in flight. Before an admission a cap would block, the episodes in flight finish and are recorded first, so the stop never cuts an admitted episode short.
  - Episodes run in parallel only in batches whose episodes do not depend on each other's outcomes:
    - the lifetimes of all seeds at once, and the final evaluation at once;
    - the gate in three stretches: replay and ablation of every candidate; generalization and regression of every candidate; then the interaction pairs of the eligible set.
  - A dry pass of the real gate code collects each stretch's episodes; it answers an unrecorded episode with a placeholder and runs none. A prefetched episode that the gate's real pass does not request fails the run closed.
  - The episodes, their keys, their settings and the decisions are the sequential run's. Only their order and concurrency differ. The scripted end-to-end tests check this: the same report and the same ledger keys, with and without interaction pairs.
- **D4 Serving (builder-2; no threshold or protocol value changes).**
  - The server loads a local snapshot of one pinned revision. These are the files `sha256_files` hashes for the run-start record.
  - It runs with `--generation-config vllm`, and every request also sends the neutral sampling values (top_p 1, top_k 0, min_p 0, repetition_penalty 1, no presence or frequency penalty). vLLM's default `--generation-config auto` would otherwise fill unset fields from the model's `generation_config.json`. Qwen2.5-7B-Instruct's sets repetition_penalty 1.05, top_p 0.8 and top_k 20, which the offline adapter's explicit `SamplingParams` never applied.
  - Batch-invariant kernels are on (`VLLM_BATCH_INVARIANT=1`), so a reply does not depend on what else is in the batch. vLLM 0.29 documents this as a beta feature, validated on Qwen2.5-7B-Instruct, that needs compute capability ≥ 8.0, which the A100 has. It may cost throughput, which the pilot measures.
  - Engine seed 0. A context of 32,768 tokens, Qwen2.5-7B-Instruct's native length, so no prompt is cut.
  - vLLM is pinned to 0.29.0, with torch 2.13.0. The notebook takes the CUDA 13 build when the driver supports CUDA 13 and the official CUDA 12.9 build otherwise. The server's version string records which one.
- **D5 Persistence (builder-2).**
  - The run works in a local directory. Snapshots of it go to a mirror on Google Drive, at most one a minute and always after a terminal record.
  - A snapshot is taken in the parent right after a ledger append, when no run file is half-written, so it is a state the run could also have crashed in.
  - Files are stored once by content hash, and each snapshot is a numbered manifest written after its files. A crash during a snapshot leaves the previous one whole.
  - A restore takes the newest whole snapshot, checks every file against its hash, and appends a `restore` note to the ledger.
  - A torn last ledger line is never salvaged silently. It can be cut off on purpose; the cut bytes are kept and a `repair` note is appended.
  - A second process taking over the mirror stops the first.

**Reason.**
- **D1.** The owner admits one well-defined accelerator, so throughput and numerics do not vary across GPU types within the run.
- **D2.** Without it, the cap would mostly count disconnected time. Active run time still bounds the GPU hours the run consumes, which is what the cap protects.
- **D3.** Worker processes keep one TextWorld game per process (TextWorld is not documented as thread-safe) and keep the parent the only ledger writer. Counting the attempts in flight keeps every hard cap a hard cap. Prefetching only dependency-free stretches keeps the run's episode set exactly the sequential one: no speculative episode is spent against the caps.
- **D4.** Without the neutral values, the server and the offline adapter would sample from different distributions under the same recorded settings. Batch invariance keeps "one deterministic execution per (task instance, condition)" meaningful when episodes share a batch.
- **D5.** A Colab VM can vanish at any moment. Snapshots taken at a crash-consistent point reuse the pipeline's existing write-ahead order instead of adding a second one.

**Research implication.**
- **Determinism.** Batch invariance holds "on the same hardware and the same vLLM version" (vLLM's reproducibility notes), so a resume on a VM with the other CUDA build may reply differently from the first session. A recorded episode is never re-run, so this can only affect episodes that ran after the last snapshot. The pilot's `temperature_0_repeat_determinism` measures repeat determinism under the run's own concurrency.
- **The accounting gap.** Work done after the last snapshot before a VM is lost runs again after the restore. The calls and the active time of the lost run are not counted, so each restore can under-count by up to about one snapshot interval of work. The restore notes keep every restore on record; the report should list them.
- **Active run time is a weaker bound than calendar time** on how long the run takes from start to end. It is the bound the owner chose.
- No novelty or causal claim. No threshold, budget, seed, allocation or arm definition changes.

**Experiment needed.** The calibration pilot (ADR-0003, ADR-0007 E2) runs on this execution path and measures, among other things, the throughput at the chosen worker count, the calls per episode and the temperature-0 repeat determinism. The compute-cap decision (takeover note §5.12) follows it.

**Compatibility impact.**
- The ledger schema goes to 0.2: session records, a clock on every record, and notes. No 0.1 ledger exists, because no run or pilot has started.
- `Exp0001Run` gains `workers`, `session_info` and `on_ledger_append`. With the default of one worker, its episodes run one at a time in the order they did before.
- The protocol's `cap_rule` text changes and gains `wall_clock_basis`, so the run-start record changes. The manifest's `compute_cap` gains `wall_clock_basis: active-run-time`, and `run_protocol` gains an `execution` line.
- ADR-0007 E5's wall-clock clause is amended by D2, and its cap rule by D3's in-flight count.
