"""`lqh status [--json]` — project state at a glance (CLI_PLAN §4.8.5).

Serializes the same attention signals the agent sees at startup
(lqh/signals.py) plus the run-directory scan. Local-first: remote/cloud
runs are polled best-effort with a bounded timeout; on failure the
signals say states may be stale instead of pretending freshness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def _remote_meta(run_dir: Path) -> dict:
    """The run's ``remote_job.json``, or ``{}`` for a local run (and for a
    marker that cannot be read, since that is what the scan reports for a
    run it could not attribute)."""
    try:
        meta = json.loads((run_dir / "remote_job.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _remote_name(run_dir: Path) -> str | None:
    """The remote a run was launched on; ``None`` for a local run."""
    name = _remote_meta(run_dir).get("remote_name")
    return name if isinstance(name, str) else None


async def _refresh_snapshot(project_dir: Path) -> tuple[dict | None, bool]:
    """The cloud snapshot, fetched (bounded) with the cache as fallback.

    ``lqh status`` used to read the cache only, and nothing on the headless
    path ever wrote one — so a harness that never opened the TUI was told
    "cloud state is unavailable (offline, not logged in, or auth failure)"
    on every call while each per-run poll worked. Never raises.
    """
    from lqh.snapshot import fetch_and_cache_snapshot, read_cached_snapshot

    try:
        return await asyncio.wait_for(fetch_and_cache_snapshot(project_dir), 10.0)
    except Exception:
        pass
    try:
        return read_cached_snapshot(project_dir), False
    except Exception:
        return None, False


def _backend_job_states(snapshot: dict | None) -> dict[str, tuple[str, str | None]]:
    """``job_id -> (state, error)`` from the snapshot's job list — the
    backend's own record of every job this project submitted."""
    from lqh.remote.cloud import _STATUS_MAP

    jobs = ((snapshot or {}).get("snapshot") or {}).get("jobs") or []
    out: dict[str, tuple[str, str | None]] = {}
    for job in jobs:
        if not isinstance(job, dict) or not job.get("id"):
            continue
        raw = str(job.get("status") or "")
        # Same mapping as CloudBackend.poll_status.
        state = "cancelled" if raw == "cancelled" else _STATUS_MAP.get(raw, raw or "unknown")
        error = job.get("error")
        out[str(job["id"])] = (state, error if isinstance(error, str) else None)
    return out


def _overlay_backend_states(
    project_dir: Path,
    supervisor: Any,
    rows: list[tuple[str, str, str | None, str | None]],
    job_states: dict[str, tuple[str, str | None]],
    *,
    refresh_all: bool,
) -> tuple[list[tuple[str, str, str | None, str | None]], bool]:
    """Project the cloud runs the scan could not poll from the backend's
    record instead of from whatever the last delivered event left on disk.

    ``refresh_all`` (the scan produced nothing) re-states every remote run;
    otherwise only the ones the scan returned as ``unknown``. Returns the
    rows and whether every remote run got a backend-backed state.
    """
    from lqh.remote.backend import JobStatus
    from lqh.remote.compute import is_cloud

    out: list[tuple[str, str, str | None, str | None]] = []
    covered = True
    for name, state, error, remote in rows:
        if remote is None or not (refresh_all or state == "unknown"):
            out.append((name, state, error, remote))
            continue
        run_dir = project_dir / "runs" / name
        meta = _remote_meta(run_dir)
        job_id = str(meta.get("job_id") or "")
        cloud = meta.get("backend") == "cloud" or is_cloud(meta.get("remote_name"))
        if not cloud or job_id not in job_states:
            covered = False
            out.append((name, state, error, remote))
            continue
        state, error = job_states[job_id]
        if state in ("completed", "failed", "cancelled"):
            # Converge the run dir on the record, so the next disk-only
            # read (TUI startup signals) agrees with the backend.
            supervisor._record_terminal_locally(
                run_dir, JobStatus(state=state, error=error),
            )
        out.append((name, state, error, remote))
    return out, covered


async def _gather(project_dir: Path) -> dict:
    from lqh.jobs import JobSupervisor
    from lqh.signals import collect_signals
    from lqh.subprocess_manager import SubprocessManager

    supervisor = JobSupervisor(project_dir)
    # The per-run scan and the project snapshot are independent — one
    # wall-clock wait for both.
    scan_result, snapshot_result = await asyncio.gather(
        asyncio.wait_for(supervisor.scan_jobs(SubprocessManager()), timeout=20.0),
        _refresh_snapshot(project_dir),
        return_exceptions=True,
    )
    scan_ok = not isinstance(scan_result, BaseException)
    snapshots = list(scan_result) if scan_ok else []
    if isinstance(snapshot_result, BaseException):
        snapshot, snapshot_fresh = None, False
    else:
        snapshot, snapshot_fresh = snapshot_result

    from_disk = False
    if not scan_ok:
        # The scan timed out (or blew up) before it produced anything, but
        # the run dirs are still on disk. An empty list here reads as "this
        # project has no runs" — the exact opposite of the truth — so fall
        # back to the same run-directory files the TUI startup path reads.
        from lqh.signals import observe_run_states

        snapshots = [
            (name, state, None, _remote_name(project_dir / "runs" / name))
            for name, state in sorted(observe_run_states(project_dir).items())
        ]
        from_disk = True

    # ~50 cloud runs polled one by one blow the scan's budget; the backend
    # already knows every job's state, in one paged request.
    covered = False
    if snapshot_fresh and snapshots:
        job_states = _backend_job_states(snapshot)
        if job_states:
            snapshots, covered = _overlay_backend_states(
                project_dir, supervisor, snapshots, job_states,
                refresh_all=from_disk,
            )

    # A per-run poll that failed (SSH/cloud hiccup) and was not re-stated
    # from the record leaves the picture partial — never present locally
    # cached states as refreshed.
    jobs_refreshed = (scan_ok or covered) and not any(
        s == "unknown" for _, s, _, _ in snapshots
    )
    run_states = {r: s for r, s, _, _ in snapshots if s != "unknown"}
    signals = collect_signals(
        project_dir,
        snapshot=snapshot,
        snapshot_fresh=snapshot_fresh,
        run_states=run_states or None,
        jobs_refreshed=jobs_refreshed,
    )

    return {
        "schema_version": 1,
        "runs": [
            {"name": name, "state": state, "error": error, "remote": remote}
            for name, state, error, remote in snapshots
        ],
        "signals": [{"kind": s.kind, "text": s.text} for s in signals],
        "jobs_refreshed": jobs_refreshed,
    }


def cmd_status(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    payload = asyncio.run(_gather(project_dir))
    if args.json_out:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if not payload["runs"]:
        print("No runs.")
    else:
        width = max(len(r["name"]) for r in payload["runs"])
        for run in payload["runs"]:
            location = f" @{run['remote']}" if run["remote"] else ""
            error = f" — {run['error']}" if run["error"] else ""
            print(f"{run['name']:<{width}}  {run['state']}{location}{error}")
    if payload["signals"]:
        print()
        for signal in payload["signals"]:
            print(f"⚠ [{signal['kind']}] {signal['text']}")
    if not payload["jobs_refreshed"]:
        print("\n(remote run states could not be refreshed)", file=sys.stderr)
    return 0
