"""Format recorded statistics without changing their numerical conventions."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def fmt_mean_std(values: Iterable[float], digits: int = 3) -> str:
    """Format a mean and sample deviation with a LaTeX plus/minus symbol."""
    finite = [value for value in values if value is not None and not math.isnan(value)]
    if not finite:
        return "---"

    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_sci(value: float, digits: int = 2) -> str:
    """Format scientific notation with a LaTeX exponent."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    if value == 0.0:
        return "$0$"

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10**exponent
    return f"${mantissa:.{digits}f} \\times 10^{{{exponent}}}$"


def fmt_pct(value: float, digits: int = 1) -> str:
    """Format a fraction as a LaTeX percentage."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"

    return f"{value * 100:.{digits}f}\\%"


def fmt_ci(point: float, lo: float, hi: float, digits: int = 2) -> str:
    """Format a point estimate and its confidence interval."""
    if any(math.isnan(value) for value in (point, lo, hi)):
        return "---"

    return f"{point:.{digits}f}\\,[{lo:.{digits}f}, {hi:.{digits}f}]"


def sanitise_filename(name: str) -> str:
    """Replace unsafe filename characters in a LaTeX label."""
    result = []

    for character in name:
        if character.isalnum() or character in "._-":
            result.append(character)
        else:
            result.append("_")

    return "".join(result)
