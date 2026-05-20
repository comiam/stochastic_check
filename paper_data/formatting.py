"""Number formatting helpers shared by all table generators."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def fmt_mean_std(values: Sequence[float], digits: int = 3) -> str:
    """``"mean +/- std"`` with the given precision; uses LaTeX ``\\pm``."""
    finite = [v for v in values if v is not None and not math.isnan(v)]
    if not finite:
        return "---"
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_sci(value: float, digits: int = 2) -> str:
    """Scientific notation with LaTeX-friendly exponent."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    if value == 0.0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(value))))
    mant = value / 10**exp
    return f"${mant:.{digits}f} \\times 10^{{{exp}}}$"


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    return f"{value * 100:.{digits}f}\\%"


def fmt_ci(point: float, lo: float, hi: float, digits: int = 2) -> str:
    if any(math.isnan(v) for v in (point, lo, hi)):
        return "---"
    return f"{point:.{digits}f}\\,[{lo:.{digits}f}, {hi:.{digits}f}]"


def sanitise_filename(name: str) -> str:
    """Translate a LaTeX label into a filesystem-safe filename component."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)
