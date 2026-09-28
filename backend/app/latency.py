"""
Latency reporting for the ~2s end-to-end budget (ARCHITECTURE.md §3).

Kept in `app/` (not `tests/`) because it is a real tool: point it at
live providers to measure, or at fakes with injected delays to prove
the math. The orchestrator already records a StageTiming for every
stage; this turns a session's timings into percentiles and a
budget verdict.

Note what is and isn't measured. `tts_first_byte` is the number the
budget cares about (time until audio *starts*), not `tts` (total
synthesis time). STT endpointing latency happens before the
orchestrator sees the final transcript, so it is NOT included here and
must be measured against a live AssemblyAI session.
"""

from collections import defaultdict
from dataclasses import dataclass

from app.pipeline.orchestrator import StageTiming

# Per-stage budgets in ms, from ARCHITECTURE.md §3. STT endpointing and
# network overhead are outside what the orchestrator can observe.
STAGE_BUDGET_MS: dict[str, float] = {
    "register_detection": 400.0,
    "translation": 600.0,
    "tts_first_byte": 500.0,
}

# Stages that run one after another once a final transcript arrives,
# whose durations therefore add up on the critical path. Register
# detection is deliberately excluded: it runs speculatively during STT
# and is usually already finished when the final arrives.
CRITICAL_PATH = ("translation", "tts_first_byte")


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; pct in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


@dataclass
class StageStats:
    stage: str
    count: int
    p50_ms: float
    p90_ms: float
    max_ms: float
    budget_ms: float | None

    @property
    def over_budget(self) -> bool:
        return self.budget_ms is not None and self.p90_ms > self.budget_ms


def summarize(timings: list[StageTiming]) -> list[StageStats]:
    by_stage: dict[str, list[float]] = defaultdict(list)
    for t in timings:
        by_stage[t.stage].append(t.duration_ms)

    return [
        StageStats(
            stage=stage,
            count=len(durations),
            p50_ms=percentile(durations, 50),
            p90_ms=percentile(durations, 90),
            max_ms=max(durations),
            budget_ms=STAGE_BUDGET_MS.get(stage),
        )
        for stage, durations in sorted(by_stage.items())
    ]


def critical_path_p90_ms(stats: list[StageStats]) -> float:
    """
    Sum of p90 across the sequential critical-path stages. This is an
    UPPER-BOUND estimate (p90s don't all land on the same utterance);
    it is meant to answer "could we blow the budget", not to predict
    the exact end-to-end number.
    """
    by_name = {s.stage: s for s in stats}
    return sum(by_name[s].p90_ms for s in CRITICAL_PATH if s in by_name)


def format_report(stats: list[StageStats], total_budget_ms: float = 2000.0) -> str:
    lines = [f"{'stage':<20}{'n':>4}{'p50':>9}{'p90':>9}{'max':>9}{'budget':>9}  status"]
    for s in stats:
        budget = f"{s.budget_ms:.0f}" if s.budget_ms is not None else "-"
        status = "OVER" if s.over_budget else ("ok" if s.budget_ms else "")
        lines.append(
            f"{s.stage:<20}{s.count:>4}{s.p50_ms:>9.0f}{s.p90_ms:>9.0f}"
            f"{s.max_ms:>9.0f}{budget:>9}  {status}"
        )
    path = critical_path_p90_ms(stats)
    verdict = "WITHIN" if path <= total_budget_ms else "OVER"
    lines.append(
        f"\ncritical path (translation + tts_first_byte) p90 sum: "
        f"{path:.0f}ms — {verdict} the {total_budget_ms:.0f}ms end-to-end "
        f"budget (excludes STT endpointing + network; measure those live)"
    )
    return "\n".join(lines)
