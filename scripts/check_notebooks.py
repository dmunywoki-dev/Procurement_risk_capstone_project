#!/usr/bin/env python3
"""Validate the two submission notebooks and parse their Python code cells."""
from __future__ import annotations
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "01_TGEE_End_to_End.ipynb",
    ROOT / "notebooks" / "02_FastAPI_Model_Service.ipynb",
]

def check() -> list[str]:
    summaries = []
    for path in NOTEBOOKS:
        if not path.is_file():
            raise FileNotFoundError(path)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise ValueError(f"{path.name}: expected nbformat 4")
        code_cells = [
            cell for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        errors = []
        for index, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source)
            except SyntaxError as exc:
                errors.append(f"code cell {index}: {exc}")
        if errors:
            raise SyntaxError(f"{path.name}: " + "; ".join(errors))
        if not code_cells:
            raise ValueError(f"{path.name}: no executable code cells")
        summaries.append(
            f"{path.relative_to(ROOT)}: {len(code_cells)} Python code cells parsed"
        )
    return summaries

if __name__ == "__main__":
    print("\n".join(check()))
