"""Entry point for ``python -m lqh.train <config.json>``.

Reads the run config and dispatches to the appropriate training loop.
All torch/transformers imports happen inside the dispatched functions,
keeping import-time lightweight so error messages are immediate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def write_provenance(run_dir: Path) -> dict[str, str]:
    """Record which trainer produced this run: ``run_dir/provenance.json``
    plus one stdout line at the top of the log.

    Two jobs launched an hour apart trained under different objectives
    (the assistant-only loss switch rode a training-image promotion) and
    nothing on either run said so — the image id lived only in the
    checkpoint's lineage row and the lqh version nowhere at all (feedback
    #142). The file is published with the run and pulled back by
    training_status; the log line is for anyone reading stdout.

    Best-effort: a run must never die because provenance could not be
    written.
    """
    from lqh import __version__

    provenance = {
        "lqh_version": __version__,
        # Injected by the cloud-job launcher (handler/cloud_jobs.go);
        # empty for SSH-direct and local runs.
        "image_id": os.environ.get("LQH_IMAGE_ID", ""),
        "image_purpose": os.environ.get("LQH_IMAGE_PURPOSE", ""),
    }
    print(
        f"lqh {__version__} · image {provenance['image_id'] or 'local'}",
        flush=True,
    )
    try:
        (run_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
    except OSError as exc:
        print(f"  WARNING: could not write provenance.json: {exc}")
    return provenance


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lqh.train <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    run_dir = config_path.parent

    # Write PID file so the main process can track us.
    (run_dir / "pid").write_text(str(os.getpid()))
    from lqh.train.progress import begin_run_attempt, write_status
    begin_run_attempt(run_dir)
    write_provenance(run_dir)

    run_type = config.get("type", "sft")

    try:
        if run_type == "sft":
            from lqh.train.sft import sft_loop

            sft_loop(run_dir, config)
        elif run_type in ("on_policy_dpo", "dpo"):
            from lqh.train.dpo import dpo_loop

            dpo_loop(run_dir, config)
        elif run_type in ("grpo", "on_policy_grpo"):
            from lqh.train.grpo import grpo_loop

            grpo_loop(run_dir, config)
        else:
            print(f"Unknown training type: {run_type!r}", file=sys.stderr)
            sys.exit(1)
    except TimeoutError:
        # DPO timeout waiting for preferences — handled inside dpo_loop
        # which writes "interrupted" status. This is a safety net.
        write_status(run_dir, "interrupted", error="Timeout waiting for preferences")
    except Exception as exc:
        # Write failure to progress so the watcher can detect it. A CUDA
        # OOM is flagged explicitly (oom=True) so the backend classifies
        # the lease as `oom` rather than `preempted` and batch auto-tuning
        # can self-heal — see lqh.train.progress.write_status.
        is_oom = _looks_like_oom(exc)
        if is_oom:
            # Self-heal: write back a smaller batch profile so the next
            # run uses it (GPU_TYPE.md §6). Best-effort, never raises.
            from lqh.train.calibrate import report_oom_downgrade

            report_oom_downgrade(config)
        write_status(run_dir, "failed", error=str(exc), oom=is_oom)
        raise


def _looks_like_oom(exc: BaseException) -> bool:
    """Best-effort CUDA out-of-memory detection without importing torch.

    torch.cuda.OutOfMemoryError subclasses RuntimeError; we match on the
    type name and message so this stays a zero-dependency check at the
    dispatch layer (torch is imported only inside the training loops).
    """
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "outofmemory" in msg


if __name__ == "__main__":
    main()
