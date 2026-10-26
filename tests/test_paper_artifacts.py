"""Verify immutable inputs and the paper's reproduced table values."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from paper_data.config import DataPaths
from paper_data.main import generate_selected
from paper_data.tables.baselines import evaluate as evaluate_baselines
from paper_data.tables.overhead import evaluate as evaluate_overhead
from stochastic_check.aggregate import aggregate_runs

FIXTURES = Path(__file__).parent / "fixtures" / "paper"


def test_artifact_checksums():
    """Detect any altered, missing, or unexpected result file."""
    root = DataPaths().results_root
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest

    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    }
    assert actual == set(expected)
    assert len(expected) == 1202

    for relative, digest in expected.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, (
            relative
        )

    assert not any(root.rglob("*.pt"))
    assert not any(root.rglob("*.pth"))
    assert not any(root.rglob("*.ckpt"))


@pytest.mark.parametrize("profile", ["exp_b", "exp_b_rerun"])
def test_artifact_run_matrix_and_records(profile):
    """Keep every seed and snapshot in the audited complete experiment matrix."""
    root = DataPaths().results_root / profile / "runs"
    cells = set()

    for path in sorted(root.glob("*/final.json")):
        final = json.loads(path.read_text())
        config = final["config"]
        training = config["training"]

        cells.add(
            (
                config["arch"],
                config["dataset"],
                training["noise_rate"],
                training["seed"],
            )
        )

        records = [
            json.loads(line)
            for line in (path.parent / "steps.jsonl").read_text().splitlines()
        ]
        assert [record["step"] for record in records] == list(range(100, 11701, 100))
        assert final["train_summary"]["global_steps"] == 11700

    expected = {
        (arch, dataset, noise, seed)
        for arch in ["resnet18", "resnet34", "vgg11"]
        for dataset in ["cifar10", "cifar100"]
        for noise in [0.0, 0.25, 0.4, 0.6]
        for seed in (range(20) if noise == 0 else range(10))
    }
    assert cells == expected


@pytest.mark.parametrize("profile", ["exp_b", "exp_b_rerun"])
def test_aggregate_matches_run_manifests(profile):
    """Keep stored indexes consistent with the original final manifests."""
    root = DataPaths().results_root / profile
    stored = json.loads((root / "aggregate.json").read_text())
    rebuilt = aggregate_runs(root / "runs")
    assert stored["runs"] == rebuilt["runs"]
    assert stored["n_runs"] == rebuilt["n_runs"] == 300


@pytest.fixture(scope="module")
def generated_tables(tmp_path_factory):
    """Run the real generators once for the module's output checks."""
    output = tmp_path_factory.mktemp("paper-tables")
    labels = ["tab:alerts", "tab:baselines", "tab:overhead"]
    assert generate_selected(DataPaths(), labels, output) == 0
    return output


@pytest.mark.parametrize("filename", ["tab_alerts.tex", "tab_overhead.tex"])
def test_tables_match_original_bytes(generated_tables, filename):
    """Preserve every published number and the established table formatting."""
    assert (generated_tables / filename).read_bytes() == (
        FIXTURES / filename
    ).read_bytes()


def test_baseline_table_matches_original_cells(generated_tables):
    """Ignore alignment whitespace while preserving all labels and numbers."""
    generated = (generated_tables / "tab_baselines.tex").read_text()
    expected = (FIXTURES / "tab_baselines.tex").read_text()
    assert re.sub(r"\s+", " ", generated) == re.sub(r"\s+", " ", expected)


def test_matched_far_values():
    """Preserve unrounded thresholds, rates, powers, and mean detection epochs."""
    expected = json.loads((FIXTURES / "expected_summary.json").read_text())["baselines"]
    points = evaluate_baselines(DataPaths().rerun_runs_root)

    for name, point in points.items():
        assert point.threshold == expected[name]["h"]
        assert point.false_alarm_rate == expected[name]["far"]
        assert point.power == expected[name]["power"]
        assert point.mean_epoch == expected[name]["mean_epoch"]


def test_recorded_overhead_values():
    """Preserve the unrounded calculation from the historical timing records."""
    expected = json.loads((FIXTURES / "expected_summary.json").read_text())["overhead"]
    points = evaluate_overhead(DataPaths().rerun_runs_root)

    for name, point in points.items():
        assert point.n_runs == expected[name]["n_runs"]
        assert point.training_ms == pytest.approx(
            expected[name]["training_ms_per_step"], abs=1e-12
        )
        assert point.trace_seconds == expected[name]["trace_s_per_snapshot"]
        assert point.overhead_percent == pytest.approx(
            expected[name]["overhead_percent"], rel=0, abs=1e-12
        )


def test_missing_inputs_report_failure(tmp_path):
    """Never turn a missing input directory into a successful empty table."""
    output = tmp_path / "output"
    assert generate_selected(DataPaths(tmp_path), ["tab:alerts"], output) == 1
    assert not output.exists()
