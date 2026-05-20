"""Progress tracking with MDC-style log tagging."""

from __future__ import annotations

import logging
import time

from lab_orchestrator._fmt import fmt_duration

_progress_ctx: dict[str, str] = {}
"""Module-level state written by the scheduler, read by the formatter."""


class ProgressFormatter(logging.Formatter):
    """Logging formatter that prepends ``[done/total pct%]`` to every line."""

    def format(self, record: logging.LogRecord) -> str:
        result = super().format(record)
        tag = _progress_ctx.get("tag", "")
        if tag:
            bracket_end = result.find("] ", result.find("["))
            if bracket_end >= 0:
                pos = bracket_end + 2
                result = f"{result[:pos]}{tag}  {result[pos:]}"
            else:
                result = f"{tag}  {result}"
        return result


def update_progress(
    done: int,
    total: int,
    done_weight: float,
    total_weight: float,
    wall_t0: float,
    failed: int,
    *,
    pre_done_weight: float = 0.0,
) -> None:
    """Update the MDC progress tag for subsequent log lines.

    Parameters
    ----------
    done, total : int
        Task counts (including pre-completed from ``--resume``).
    done_weight, total_weight : float
        Cumulative weight (GPU-hours).
    wall_t0 : float
        ``time.monotonic()`` at scheduler start.
    failed : int
        Number of tasks that exited with non-zero return code.
    pre_done_weight : float
        Weight of tasks completed before this run (``--resume``).
    """
    pct = done_weight / total_weight * 100 if total_weight > 0 else 0

    elapsed = time.monotonic() - wall_t0
    run_done = done_weight - pre_done_weight
    if run_done > 0 and pct < 100:
        eta = elapsed * (total_weight - done_weight) / run_done
        eta_str = f" ETA {fmt_duration(eta)}"
    else:
        eta_str = ""

    fail_str = f" {failed}!" if failed else ""
    _progress_ctx["tag"] = f"[{done}/{total} {pct:.0f}%{fail_str}{eta_str}]"


def set_done_tag() -> None:
    """Set progress tag to ``[DONE]``."""
    _progress_ctx["tag"] = "[DONE]"
