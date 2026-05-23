"""Ad-hoc diagnostics for tab:alerts -- not committed to package surface."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_data.config import DataPaths
from paper_data.loader import iter_run_finals, load_jsonl
from stochastic_check.decision import (
    EnsembleBaseline,
    calibrate_threshold,
    cusum_statistic,
    online_cusum_alert,
)

K = 0.5
ARL0 = 1000.0
HEAD = {"resnet18": "fc", "resnet34": "fc", "vgg11": "classifier"}


def main() -> None:
    paths = DataPaths()
    runs_root = paths.exp_b_runs_root
    by_cell: dict[tuple[str, str, float], dict[int, Path]] = defaultdict(dict)
    val_acc: dict[tuple[str, str, float], list[float]] = defaultdict(list)

    for run_dir, final in iter_run_finals(runs_root):
        cfg = final.get("config", {})
        arch = cfg["arch"]
        dataset = cfg["dataset"]
        noise = float(cfg["training"]["noise_rate"])
        seed = int(cfg["training"]["seed"])
        jsonl = run_dir / "steps.jsonl"
        if jsonl.exists():
            by_cell[(arch, dataset, noise)][seed] = jsonl
        acc = (final.get("metrics") or {}).get("val_acc")
        if acc is not None:
            val_acc[(arch, dataset, noise)].append(float(acc))

    print("\n=== Terminal val_acc per cell ===")
    print(f"{'arch':<10}{'dataset':<10}{'eta':<8}{'val_acc mean':<14}{'std':<8}{'n':<4}")
    for key in sorted(val_acc):
        a = np.asarray(val_acc[key])
        print(
            f"{key[0]:<10}{key[1]:<10}{key[2]:<8}{a.mean():<14.4f}{a.std(ddof=1):<8.4f}{a.size:<4}"
        )

    print("\n=== Per-(arch,dataset) diagnostics ===")
    groups: dict[tuple[str, str], dict[float, dict[int, Path]]] = defaultdict(dict)
    for (arch, ds, noise), seeds in by_cell.items():
        groups[(arch, ds)][noise] = seeds

    diagnostics: dict[str, dict] = {}

    for (arch, ds), by_noise in sorted(groups.items()):
        head = HEAD[arch]
        clean = by_noise.get(0.0, {})
        clean_recs = {s: load_jsonl(p) for s, p in sorted(clean.items())}
        baseline = EnsembleBaseline.from_runs(list(clean_recs.values()), head)
        h = calibrate_threshold(
            list(clean_recs.values()),
            layer=head,
            k=K,
            target_arl0=ARL0,
            n_replicates=400,
            rng=np.random.default_rng(0),
        )

        # Pool of leave-one-out null z and per-seed FP behaviour.
        loo_z: dict[int, np.ndarray] = {}
        fp_alerts: dict[int, int | None] = {}
        for seed, recs in clean_recs.items():
            rest = [r for s, r in clean_recs.items() if s != seed]
            loo = EnsembleBaseline.from_runs(rest, head)
            _, _, z = loo.standardize(recs)
            loo_z[seed] = z
            alert = online_cusum_alert(recs, baseline=loo, k=K, h=h)
            fp_alerts[seed] = alert.alert_step

        # ACF(1) on pooled null z, plus per-run.
        per_run_acf = []
        for z in loo_z.values():
            if z.size > 5:
                z0 = z - z.mean()
                denom = float(z0 @ z0)
                if denom > 0:
                    per_run_acf.append(float(z0[:-1] @ z0[1:]) / denom)
        acf1 = float(np.mean(per_run_acf)) if per_run_acf else float("nan")

        # Effective length adjustment: with positive ACF, effective N ~ N (1-rho)/(1+rho)
        # so the IID-based ARL0 calibration is optimistic by factor (1+rho)/(1-rho).
        adj = (1 + acf1) / (1 - acf1) if -1 < acf1 < 1 else float("nan")

        # Empirical FP curve: at the calibrated h, how often does each seed
        # produce an alert anywhere in [1, 117]?
        fp_seeds = sum(1 for v in fp_alerts.values() if v is not None)
        early_fp = sum(1 for v in fp_alerts.values() if v is not None and v <= 5)

        diagnostics[f"{arch}/{ds}"] = {
            "h": h,
            "acf1": acf1,
            "iid_adjustment_factor": adj,
            "fp_count": fp_seeds,
            "fp_early_count": early_fp,
            "fp_seeds": {s: v for s, v in fp_alerts.items() if v is not None},
        }

        print(f"\n--- {arch} / {ds} (head={head}) ---")
        print(f"  h={h:.3f}")
        print(
            f"  ACF(1) of null z = {acf1:+.3f}  ->  IID-vs-AR(1) ARL inflation factor = {adj:.2f}"
        )
        print(
            f"  FP: {fp_seeds}/{len(clean_recs)} seeds alert (of which {early_fp} in first 5 steps)"
        )
        print(f"  FP first-alert step per seed: {diagnostics[f'{arch}/{ds}']['fp_seeds']}")

    out = Path("paper_data/tables_out/diagnostics.json")
    out.write_text(json.dumps(diagnostics, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
