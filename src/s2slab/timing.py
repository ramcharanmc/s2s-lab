"""
Latency measurement and reporting. Pure CPU.

Two rules, both of which exist because latency distributions are heavy-tailed:

1. PERCENTILES, NEVER MEANS. A pipeline with a 300 ms mean and a 4 s p99 is not
   a 300 ms pipeline -- it is a pipeline that fails one user in a hundred. The
   mean hides exactly the tail you are being paid to find.

2. A PERCENTILE YOU LACK THE SAMPLES FOR IS NOT A MEASUREMENT. numpy will
   happily interpolate a p99 from 30 samples, but that number is a property of
   your two slowest runs, not of the system. Every summary here carries its own
   sample count so the claim can be checked.

All durations are MILLISECONDS unless a name says otherwise.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

# To estimate the pth percentile you want roughly 10 / (1 - p) samples:
# p50 -> 20, p90 -> 100, p99 -> 1000. Below that, report it as unsupported
# rather than quietly printing a number nobody should act on.
_MIN_SAMPLES = {"p50": 20, "p90": 100, "p99": 1000}

@dataclass
class LatencySummary:
    """Percentiles plus the sample count that backs them."""

    n: int
    p50: float
    p90: float
    p99: float

    @property
    def unsupported(self) -> list[str]:
        """Percentiles this sample size cannot honestly support."""
        return [k for k, need in _MIN_SAMPLES.items() if self.n < need]

    def __str__(self) -> str:
        flag = f"  [n={self.n}, unsupported: {', '.join(self.unsupported)}]" \
            if self.unsupported else f"  [n={self.n}]"
        return f"p50 {self.p50:7.1f}  p90 {self.p90:7.1f}  p99 {self.p99:7.1f}{flag}"


def summarize(values_ms) -> LatencySummary:
    """Percentiles over a sequence of millisecond durations.

    NaN inputs are dropped rather than poisoning the result -- a failed turn
    should not silently make every percentile NaN -- but they are excluded from
    `n` too, so the sample-size warning stays truthful.
    """
    arr = np.asarray(list(values_ms), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return LatencySummary(0, float("nan"), float("nan"), float("nan"))
    return LatencySummary(
        n=int(arr.size),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p99=float(np.percentile(arr, 99)),
    )

@contextmanager
def stage(name: str, sink: list):
    """Time one pipeline stage, appending (name, t_start, t_end) to `sink`.

    Thread ONE sink list through an entire turn and per-stage attribution falls
    out for free -- no bespoke timing code per stage, and no way for a stage to
    be silently untimed.

    perf_counter, not time.time: it is monotonic, so an NTP correction landing
    mid-benchmark cannot produce a negative duration. The absolute values are
    meaningless; only differences matter.

    The timing is recorded in `finally`, so a stage that raises is still
    attributed. A crashed stage with no timing is invisible in the report,
    which is the worst possible failure mode for a latency investigation.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sink.append((name, t0, time.perf_counter()))

def per_stage_ms(sink: list) -> dict[str, float]:
    """Collapse one turn's sink into {stage: total_ms}.

    Repeated names are SUMMED, not overwritten. A turn that synthesizes three
    TTS chunks has three `tts` entries, and the question being asked is "how
    much time did TTS cost this turn", so summing is the answer. Overwriting
    would report only the last chunk and understate the stage.
    """
    out: dict[str, float] = {}
    for name, t0, t1 in sink:
        out[name] = out.get(name, 0.0) + (t1 - t0) * 1000
    return out


def summarize_stages(sinks: list[list]) -> dict[str, LatencySummary]:
    """Per-stage percentiles across many turns.

    Stages absent from a given turn are simply not counted for that turn, so a
    conditionally-run stage reports the sample size it actually has rather than
    being diluted by zeros.
    """
    collected: dict[str, list[float]] = {}
    for sink in sinks:
        for name, ms in per_stage_ms(sink).items():
            collected.setdefault(name, []).append(ms)
    return {name: summarize(v) for name, v in collected.items()}


def format_table(summaries: dict[str, LatencySummary], title: str = "") -> str:
    """Markdown table, sorted slowest p50 first -- the optimization order."""
    rows = sorted(summaries.items(), key=lambda kv: -kv[1].p50)
    lines = [f"### {title}", ""] if title else []
    lines += ["| stage | p50 ms | p90 ms | p99 ms | n |",
              "|---|---:|---:|---:|---:|"]
    for name, s in rows:
        p99 = "—" if "p99" in s.unsupported else f"{s.p99:.1f}"
        p90 = "—" if "p90" in s.unsupported else f"{s.p90:.1f}"
        lines.append(f"| {name} | {s.p50:.1f} | {p90} | {p99} | {s.n} |")
    return "\n".join(lines)
