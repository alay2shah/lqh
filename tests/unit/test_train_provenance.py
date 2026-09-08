"""lqh.train.__main__.write_provenance — the trainer stamps what it is
(feedback #142)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lqh import __version__
from lqh.train.__main__ import write_provenance


def test_cloud_run_records_version_and_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LQH_IMAGE_ID", "im-abc123")
    monkeypatch.setenv("LQH_IMAGE_PURPOSE", "sft")
    got = write_provenance(tmp_path)
    on_disk = json.loads((tmp_path / "provenance.json").read_text())
    assert got == on_disk == {
        "lqh_version": __version__,
        "image_id": "im-abc123",
        "image_purpose": "sft",
    }
    assert f"lqh {__version__} · image im-abc123" in capsys.readouterr().out


def test_local_run_has_no_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LQH_IMAGE_ID", raising=False)
    monkeypatch.delenv("LQH_IMAGE_PURPOSE", raising=False)
    got = write_provenance(tmp_path)
    assert got["image_id"] == "" and got["image_purpose"] == ""
    assert "image local" in capsys.readouterr().out


def test_unwritable_run_dir_does_not_raise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does" / "not" / "exist"
    got = write_provenance(missing)
    assert got["lqh_version"] == __version__
    assert "could not write provenance.json" in capsys.readouterr().out
