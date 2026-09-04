"""`lqh status [--json]` — signals + run scan serialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lqh.cli_cmds.status_cmd import cmd_status


def _ns(json_out: bool = True) -> argparse.Namespace:
    return argparse.Namespace(command="status", json_out=json_out)


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
