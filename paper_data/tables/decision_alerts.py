"""Table ``tab:alerts`` -- online :math:`3\\sigma` detection rule (paper sec. 7.2).

For every ``(arch, dataset, noise > 0, seed)`` run we replay the
per-step trace stream through :func:`stochastic_check.decision.online_3sigma_alert`
*per layer*, then collapse the per-layer alerts to a single number per
run -- the **earliest** epoch at which any monitored layer crosses the
:math:`3\\sigma` band built on the first ``BASELINE_EPOCHS`` of the
run itself.  Aggregating over seeds gives the columns of the paper
table:

  * ``mean detection epoch``  (over seeds whose first-crossing layer
    fired at all),
  * ``alert rate``            (fraction of seeds that ever fired),
  * ``earliest layer``        (most common first-crossing layer).

This is the artefact that backs the §7.2 claim that the on-line
detector triggers well before terminal validation accuracy collapses.
Clean runs (``noise = 0``) are omitted from the rows but their alert
rate is kept as a control row to advertise the false-positive count.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stochastic_check.decision import online_3sigma_alert

from ..config import DataPaths
from ..formatting import fmt_mean_std
from ..loader import iter_run_finals, load_jsonl

# Calibration window of the baseline statistics (mu_0, s_0) and the
# critical z-score.  The paper fixes both numerically; they live in
# one place so audit-trail is single-source.
BASELINE_EPOCHS: int = 5
THRESHOLD: float = 3.0


def _layers_in(records: list[dict[str, Any]]) -> list[str]:
    """Stable layer ordering: first non-empty ``trace`` dict in the run.

    Layers introduced later in training would change this list; for the
    trace-trajectory schema produced by :class:`Trainer` the set is
    fixed at construction time, so the first record is sufficient.
    """
    for rec in records:
        trace = rec.get("trace") or {}
        if trace:
            return list(trace.keys())
    return []


def _first_alert(records: list[dict[str, Any]]) -> tuple[int | None, str | None]:
    """Earliest ``(alert_epoch, layer)`` across all layers; ``None`` if mute.

    Ties are broken by lexicographic layer name to stay deterministic.
    """
    layers = _layers_in(records)
    best_epoch: int | None = None
    best_layer: str | None = None
    for layer in layers:
        alert = online_3sigma_alert(
            records, layer, baseline_epochs=BASELINE_EPOCHS, threshold=THRESHOLD
        )
        if alert.alert_epoch is None:
            continue
        if (
            best_epoch is None
            or alert.alert_epoch < best_epoch
            or (alert.alert_epoch == best_epoch and (best_layer or "") > layer)
        ):
            best_epoch = alert.alert_epoch
            best_layer = layer
    return best_epoch, best_layer


def generate(paths: DataPaths, out_dir: Path) -> Path:
    # Per-cell accumulators keyed by (arch, dataset, noise).
    epochs: dict[tuple[str, str, float], list[int]] = defaultdict(list)
    n_seeds: dict[tuple[str, str, float], int] = defaultdict(int)
    earliest_layer: dict[tuple[str, str, float], Counter[str]] = defaultdict(Counter)

    for run_dir, final in iter_run_finals(paths.exp_b_runs_root):
        cfg = final.get("config", {})
        arch = cfg.get("arch")
        dataset = cfg.get("dataset")
        noise = cfg.get("training", {}).get("noise_rate")
        if None in (arch, dataset, noise):
            continue
        jsonl = run_dir / "steps.jsonl"
        if not jsonl.exists():
            continue
        records = load_jsonl(jsonl)
        cell = (str(arch), str(dataset), float(noise))
        n_seeds[cell] += 1
        ep, layer = _first_alert(records)
        if ep is not None:
            epochs[cell].append(ep)
            if layer is not None:
                earliest_layer[cell][layer] += 1

    cells = sorted(n_seeds)
    # ---- LaTeX rendering ----
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lllccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Arch & Dataset & $\eta$ & "
        r"Det.\ epoch (mean$\pm$std) & Alert rate & Earliest layer \\"
    )
    lines.append(r"\midrule")
    for cell in cells:
        arch, dataset, noise = cell
        seen = n_seeds[cell]
        alerted = epochs[cell]
        rate = (len(alerted) / seen) if seen > 0 else math.nan
        if earliest_layer[cell]:
            layer_str = earliest_layer[cell].most_common(1)[0][0]
        else:
            layer_str = "---"
        lines.append(
            " & ".join(
                [
                    arch.replace("_", r"\_"),
                    dataset.replace("_", r"\_"),
                    f"${noise:g}$",
                    fmt_mean_std(np.asarray(alerted, dtype=float), digits=1)
                    if alerted
                    else "---",
                    f"{rate:.2f}" if not math.isnan(rate) else "---",
                    layer_str.replace("_", r"\_"),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = out_dir / "tab_alerts.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
