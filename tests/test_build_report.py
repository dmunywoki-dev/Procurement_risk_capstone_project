import pandas as pd
from scripts.build_report import _markdown_table, _fmt

def test_markdown_table_renders_rows_and_headers():
    frame = pd.DataFrame({"metric": ["spearman"], "value": [0.87543]})
    table = _markdown_table(frame, ["metric", "value"])
    assert "| metric | value |" in table
    assert "0.8754" in table

def test_format_missing_values():
    assert _fmt(float("nan")) == "Not available"
