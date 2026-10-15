# EXP-0001 on Colab: operational safeguards

Builder-2's engineering within ADR-0008 D4 and D5, added after the owner accepted those decisions (2026-10-15). It changes no threshold, budget, seed, allocation, arm or cap rule. **It is not part of ADR-0008's accepted text.** It is kept here for the owner's look, and only the owner's decision moves it into the ADR. Code: `evaluation/exp0001/cli.py`, `serving.py`, `mirror.py`, `pipeline.py`, `notebooks/`.

## Serving (within D4)

- `serve` writes its server's command, settings and pid to the info file (`/content/vllm-server.json`) before it waits for the model to load. An interrupted `serve` therefore leaves a record of the server it started.
- An earlier server that still runs:
  - with the same command and settings, it is waited for if still loading, then reused;
  - with other settings, `serve` refuses and names its pid: it may still be loading and holds the GPU.
- Otherwise, a port that already answers is refused. So the recorded server settings are always those of the server the run talks to.

## Persistence and release (within D5)

- **One writer per run directory.** `run`, `pilot` and `snapshot` take the run directory's lock before they touch the mirror, and hold it until after their last snapshot. A second command on the same directory fails at the lock, before it can take over the mirror or write the ledger. Example: a command started from a notebook whose kernel restarted while the previous command was still stopping.
- **Unmirrored marker.** When the last snapshot of a session fails (Drive full or disconnected), `run`, `pilot` and `snapshot` exit with code 3, stopped or not, and leave `UNMIRRORED.json` in the run directory: the newest records exist only on that machine. The marker is never mirrored, and a restore drops it.
- **Release.** The notebooks never release the runtime:
  - while an `UNMIRRORED.json` marker is present. The owner mounts Drive again and runs `snapshot`, which clears the marker;
  - while a `run`, `pilot` or `snapshot` command is still running on the machine (a scan of `/proc`). That command is a previous one whose kernel restarted, which stops by itself.

  Before every release they flush Drive (`drive.flush_and_unmount`, at most 10 minutes), so the last writes reach it. The flush is best effort: if it fails or times out, that is printed and the runtime is released anyway.
- **Stopping.** A notebook command runs in its own process group.
  - The stop button reaches it once and stops it as a time limit does: the episodes in flight are recorded and a last snapshot is taken.
  - A command whose kernel restarts or dies stops the same way: the notebook helper passes the kernel's pid in `HERITAGENT_STOP_WITH_PARENT`, and the command stops when that process is no longer its parent.
  - A stopped command exits with code 130, or 3 when its last snapshot failed.
