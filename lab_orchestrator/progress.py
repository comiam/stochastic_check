"""Track schedule progress through a shared log-formatting tag."""

from __future__ import annotations

import logging
import time

from lab_orchestrator._fmt import fmt_duration

_progress_ctx: dict[str, str] = {}


class ProgressFormatter(logging.Formatter):
    """Insert the current completion tag into formatted log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Append the tag after the first bracketed field when present."""
        result = super().format(record)
        tag = _progress_ctx.get("tag", "")

        if tag:
            bracket_end = result.find("] ", result.find("["))

            if bracket_end >= 0:
                position = bracket_end + 2
                result = f"{result[:position]}{tag}  {result[position:]}"
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
    """Update the tag using completion counts, GPU-hour weights, and elapsed time.

    Counts include resumed work. ETA uses only weight completed in this
    schedule, excluding the pre_done_weight offset.
    """
    percent = done_weight / total_weight * 100 if total_weight > 0 else 0
    elapsed = time.monotonic() - wall_t0
    run_done = done_weight - pre_done_weight

    if run_done > 0 and percent < 100:
        eta = elapsed * (total_weight - done_weight) / run_done
        eta_text = f" ETA {fmt_duration(eta)}"
    else:
        eta_text = ""

    failures = f" {failed}!" if failed else ""
    _progress_ctx["tag"] = f"[{done}/{total} {percent:.0f}%{failures}{eta_text}]"


def set_done_tag() -> None:
    """Mark subsequent log records as belonging to a finished schedule."""
    _progress_ctx["tag"] = "[DONE]"
