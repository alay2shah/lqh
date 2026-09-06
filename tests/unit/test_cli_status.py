"""`lqh status [--json]` — signals + run scan serialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from lqh.cli_cmds.status_cmd import cmd_status


def _ns(json_out: bool = True) -> argparse.Namespace:
    return argparse.Namespace(command="status", json_out=json_out)


@pytest.fixture(autouse=True)
def _offline_snapshot(monkeypatch):
    """`lqh status` now refreshes the cloud snapshot; keep the suite off
    the network (and off the developer's login) — "unavailable" by default."""
    async def _unavailable(project_dir, **_kw):
        return None, False

    monkeypatch.setattr("lqh.snapshot.fetch_and_cache_snapshot", _unavailable)


def _fresh_snapshot(monkeypatch, jobs: list[dict]) -> None:
    async def _fetched(project_dir, **_kw):
        return {
            "schema_version": 1,
            "fetched_at": "2026-09-05T10:00:00+00:00",
            "snapshot": {"jobs": jobs},
        }, True

    monkeypatch.setattr("lqh.snapshot.fetch_and_cache_snapshot", _fetched)


def _cloud_run(tmp_path: Path, name: str, job_id: str) -> Path:
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"type": "sft"}))
    (run / "remote_job.json").write_text(json.dumps({
        "job_id": job_id, "remote_name": "cloud", "backend": "cloud",
        "remote_run_dir": f"cloud:lqh/runs/{name}",
    }))
    return run


def test_empty_project(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_status(_ns()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["runs"] == []
    assert payload["jobs_refreshed"] is True


def test_local_run_states_reported(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "sft_v1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"type": "sft"}))
    (run / "progress.jsonl").write_text(
        json.dumps({"status": "completed"}) + "\n"
    )
    assert cmd_status(_ns()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"] == [
        {"name": "sft_v1", "state": "completed", "error": None, "remote": None}
    ]


def test_human_output(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_status(_ns(json_out=False)) == 0
    assert "No runs." in capsys.readouterr().out


def test_failed_refresh_still_lists_runs(tmp_path: Path, monkeypatch, capsys) -> None:
    """A scan that times out must not report the project as run-less."""
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "sft_cloud"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"type": "sft"}))
    (run / "remote_job.json").write_text(
        json.dumps({"job_id": "job-1", "remote_name": "cloud", "backend": "cloud"})
    )

    async def _boom(self, manager):  # noqa: ANN001
        raise TimeoutError("backend unreachable")

    monkeypatch.setattr("lqh.jobs.JobSupervisor.scan_jobs", _boom)

    assert cmd_status(_ns()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["jobs_refreshed"] is False
    assert [r["name"] for r in payload["runs"]] == ["sft_cloud"]
    assert payload["runs"][0]["state"] == "running"
    assert payload["runs"][0]["remote"] == "cloud"
    assert "refresh_failed" in {s["kind"] for s in payload["signals"]}


def test_snapshot_is_fetched_not_just_read_from_cache(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """A headless harness never opens the TUI, so nothing ever cached a
    snapshot for it — every `lqh status` said cloud state was unavailable
    while every per-run poll worked. Fetch it, and say "unavailable" only
    when the fetch actually failed."""
    monkeypatch.chdir(tmp_path)
    assert cmd_status(_ns()) == 0
    kinds = {s["kind"] for s in json.loads(capsys.readouterr().out)["signals"]}
    assert "snapshot_unavailable" in kinds  # the autouse stub: fetch failed

    _fresh_snapshot(monkeypatch, jobs=[])
    assert cmd_status(_ns()) == 0
    kinds = {s["kind"] for s in json.loads(capsys.readouterr().out)["signals"]}
    assert "snapshot_unavailable" not in kinds
    assert "snapshot_stale" not in kinds


def test_failed_scan_projects_cloud_runs_from_backend_record(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """~50 cloud runs polled one by one blow the scan's budget. The backend's
    job list is the registry: states come from it, the run dir converges on
    it, and the result counts as refreshed."""
    monkeypatch.chdir(tmp_path)
    done = _cloud_run(tmp_path, "sft_done", "job-1")
    _cloud_run(tmp_path, "sft_live", "job-2")

    async def _boom(self, manager):  # noqa: ANN001
        raise TimeoutError("scan budget exhausted")

    monkeypatch.setattr("lqh.jobs.JobSupervisor.scan_jobs", _boom)
    _fresh_snapshot(monkeypatch, jobs=[
        {"id": "job-1", "status": "completed", "kind": "sft"},
        {"id": "job-2", "status": "running", "kind": "sft"},
    ])

    assert cmd_status(_ns()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {r["name"]: r["state"] for r in payload["runs"]} == {
        "sft_done": "completed", "sft_live": "running",
    }
    assert payload["jobs_refreshed"] is True
    assert "refresh_failed" not in {s["kind"] for s in payload["signals"]}
    # The terminal verdict now lives on disk too — a disk-only read agrees.
    rows = [json.loads(l) for l in (done / "progress.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] == "completed"


def test_unknown_poll_result_is_restated_from_backend_record(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    _cloud_run(tmp_path, "sft_a", "job-a")
    ssh = tmp_path / "runs" / "sft_ssh"
    ssh.mkdir(parents=True)
    (ssh / "config.json").write_text(json.dumps({"type": "sft"}))
    (ssh / "remote_job.json").write_text(json.dumps({
        "job_id": "7", "remote_name": "ssh:box", "remote_run_dir": "/r",
    }))

    async def _scan(self, manager):  # noqa: ANN001
        return [
            ("sft_a", "unknown", None, "cloud"),
            ("sft_ssh", "unknown", None, "ssh:box"),
        ]

    monkeypatch.setattr("lqh.jobs.JobSupervisor.scan_jobs", _scan)
    _fresh_snapshot(monkeypatch, jobs=[
        {"id": "job-a", "status": "failed", "error": "exit code 1"},
    ])

    assert cmd_status(_ns()) == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {r["name"]: r for r in payload["runs"]}
    assert by_name["sft_a"]["state"] == "failed"
    assert by_name["sft_a"]["error"] == "exit code 1"
    # An SSH box is not in the backend's record: still unknown, still partial.
    assert by_name["sft_ssh"]["state"] == "unknown"
    assert payload["jobs_refreshed"] is False
