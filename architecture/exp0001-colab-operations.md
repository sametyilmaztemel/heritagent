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
- **Unmirrored marker.** When the last snapshot of a session fails (Drive full or disconnected, or anything else that stops it short of a whole snapshot), `UNMIRRORED.json` is left in the run directory: the newest records exist only on that machine. It is written before the command's own messages, since a command whose kernel died writes into a pipe nobody reads. `run` and `pilot` then exit with code 3 when they finished, paused or were stopped, and `snapshot` exits with 3; after an explained error the exit code is 2, after an unexpected exception Python's 1, and the marker is left all the same. A SIGTERM while the command holds its directory (the stop button's second press) also leaves the marker before the process ends. The marker is never mirrored, and a restore drops it. Another machine taking over the mirror is the one exception: it holds the run now, and the command stops with exit 2 and no marker.
- **The last snapshot is not cut short.** A stop (the stop button, or the SIGINT of a dead kernel's watcher) that arrives during it is dropped, since the command ends right after. A second press of the stop button still ends the command at once, leaving the marker.
- **Release.** The notebooks never release the runtime:
  - while an `UNMIRRORED.json` marker is present. The owner mounts Drive again and runs `snapshot`, which clears the marker;
  - while a `run`, `pilot` or `snapshot` command is still running on the machine (a scan of `/proc`). That command is a previous one whose kernel restarted, which stops by itself. If it has not ended after about 15 minutes, the owner ends its process group, which leaves the marker (its records since the last snapshot stay on the machine, so `snapshot` follows), or uses Runtime > Disconnect and delete runtime, which discards those records.

  Before every release they flush Drive (`drive.flush_and_unmount`, at most 10 minutes), so the last writes reach it. The flush is best effort: if it fails or times out, that is printed and the runtime is released anyway.
- **Stopping.** A notebook command runs in its own process group.
  - The stop button reaches it once and stops it as a time limit does: the episodes in flight are recorded and a last snapshot is taken.
  - A command whose kernel restarts or dies stops the same way: the notebook helper passes the kernel's pid in `HERITAGENT_STOP_WITH_PARENT`, and the command stops when that process is no longer its parent.
  - A stopped command exits with code 130, or 3 when its last snapshot failed.
  - Its own messages and the mirror's warnings never fail on a pipe nobody reads any more.

## Driving the notebooks from a Claude session (colab-mcp)

The owner asked on 2026-10-18 that Claude run the notebooks and read their outputs in Colab. This section is how. It changes nothing in the notebooks, the run or ADR-0008.

- **Where it works.** `.mcp.json` registers Google's colab-mcp, pinned to commit `b9ab389`. It works only in a Claude Code session on the owner's computer: the Desktop app's local session, or `claude remote-control` in a local clone. The server listens on that computer's `localhost`, and the Colab tab in the owner's browser connects to it. A cloud session cannot be reached by the tab, so it does not call the tool.
- **Setup.** The computer needs `uv` and `git`. Claude Code asks once to approve the project's server. If the first start times out while `uvx` installs it, start Claude Code with `MCP_TIMEOUT=60000`.
- **Tried so far.** In a cloud container (2026-10-18), the pinned server installs, answers the MCP handshake and lists `open_colab_browser_connection`. No session on the owner's computer has tried it yet. Reports differ on whether Claude Code shows the notebook tools after the tab connects. The first live trial settles that on a CPU runtime with `EXP-0001-rehearsal.ipynb`, before any A100 notebook.
- **What it cannot do.**
  - The tool opens an empty scratch notebook, not a repository notebook. Connecting a repository notebook through Colab's command palette ("mcp") is reported to work but untried. Otherwise Claude copies the repository notebook's cells into the scratch notebook unchanged.
  - It has no tool to choose, change or release a runtime. The owner picks the runtime type.
  - Mounting Drive needs the owner's consent in the browser.
  - The tab must stay open: a Colab session ends when its browser disconnects, and the VM's disk goes with it (ADR-0008, context).
- **Rules for Claude.**
  - Before a step that starts or prolongs A100 use, Claude asks the owner. A100 time spends the owner's compute units (ADR-0008 D1).
  - The guard cells stay as they are: the GPU check and the rehearsal marker. Claude never writes a marker itself.
  - `CODE_REF` is the commit the rehearsal printed, in the rehearsal and in the A100 notebook alike.
  - One tab and one command per run directory. A `RUN_ID` never runs in two tabs or two sessions at once (the run notebook's first cell); resuming it in a later session is the normal path.
  - To stop, Claude presses Stop once and waits. A second press is only for a command that has hung. It never uses "Disconnect and delete runtime" while `UNMIRRORED.json` exists or a command runs. It releases the runtime with the notebook's own release cell.
  - Claude reads outputs in the notebook and the records in the Drive mirror. It reports them with status keys and never calls a result validated (SPEC §32).
