"""Preserve the original JSONL normalization contract."""

import pytest

from paper_data.loader import load_jsonl as load_paper_records
from stochastic_check.utils import load_jsonl as load_training_records


@pytest.mark.parametrize("load_records", [load_paper_records, load_training_records])
def test_jsonl_retains_whitespace_normalization(load_records, tmp_path):
    """Strip surrounding whitespace and omit blank records before parsing."""
    path = tmp_path / "records.jsonl"
    path.write_text(
        chr(160) + '{"value": 1}' + chr(160) + '\n \t\n{"value": 2}\n', encoding="utf-8"
    )
    assert load_records(path) == [{"value": 1}, {"value": 2}]
