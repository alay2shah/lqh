"""A slow ``source()`` must not stall the event loop (feedback #145).

``lqh.sources.hf_dataset`` streams shards over the network, so consuming
the first N items can take minutes. The engine used to iterate it on the
event loop, which froze the TUI (no repaint, no Esc, status bar stuck on
"ready") for the whole fetch. The source is now consumed on a worker
thread; this test drives a source that blocks in ``time.sleep`` and checks
that other tasks on the loop keep running while it does.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from textwrap import dedent

from lqh.engine import run_pipeline


async def test_slow_source_keeps_event_loop_responsive(
    chdir_to_tmp: Path, mock_openai_client
) -> None:
    project = chdir_to_tmp
    marks = project / "marks.jsonl"
    pipeline = project / "data_gen" / "p.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text(dedent(
        f"""
        import json, time
        from lqh.pipeline import Pipeline, ChatMLMessage

        class Slow(Pipeline):
            @classmethod
            def source(cls, project_dir):
                def gen():
                    for i in range(4):
                        time.sleep(0.05)  # a shard download, in miniature
                        with open({str(marks)!r}, "a") as f:
                            f.write(json.dumps({{"yield": time.monotonic()}}) + "\\n")
                        yield f"item-{{i}}"
                return gen()

            async def generate(self, client, input):
                return [ChatMLMessage("user", input), ChatMLMessage("assistant", "ok")]
        """
    ))

    ticks: list[float] = []

    async def ticker() -> None:
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.005)

    t = asyncio.create_task(ticker())
    try:
        result = await run_pipeline(
            script_path=pipeline,
            num_samples=4,
            output_dir=project / "datasets" / "v1",
            client=mock_openai_client(content="unused"),
            concurrency=1,
            max_retries=0,
        )
    finally:
        t.cancel()

    assert result.succeeded == 4

    yields = [json.loads(l)["yield"] for l in marks.read_text().splitlines()]
    assert len(yields) == 4
    first, last = yields[0], yields[-1]
    # With the source iterated on the loop, no tick can land between the
    # first and last yield: the loop is blocked for the whole ~150 ms.
    assert any(first < tick < last for tick in ticks), (
        f"event loop was blocked while source() ran: {len(ticks)} ticks, none between yields"
    )
