#!/usr/bin/env python3
"""Check required source, generated outputs and submission files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    required = [
        "README.md", "AI_USE.md", "requirements.txt", "requirements-api.txt",
        "Dockerfile", "docker-compose.yml", "api_service.py",
        "notebooks/01_TGEE_End_to_End.ipynb",
        "notebooks/02_FastAPI_Model_Service.ipynb",
        "reports/FINAL_REPORT.md", "reports/PRESENTATION_SLIDES.md",
        "tests/test_api.py", "scripts/monitor_drift.py",
        "docs/TECHNIQUE_JUSTIFICATION.md",
        "docs/OPERATIONS_AND_GOVERNANCE.md",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    metadata_path = ROOT / "artifacts/fy2025_2026/fy2025_2026_run_metadata.json"
    bundle_path = ROOT / "artifacts/fy2025_2026/tgee_fy2025_2026_model_bundle.joblib"
    if not metadata_path.is_file():
        missing.append(str(metadata_path.relative_to(ROOT)))
    if not bundle_path.is_file():
        missing.append(str(bundle_path.relative_to(ROOT)))
    mode = "not_run"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mode = metadata.get("data_mode", "unknown")
        if mode != "official_fy2025_2026" and not args.allow_synthetic:
            missing.append(
                "official FY2025/2026 run required (current data_mode="
                + str(mode) + "); synthetic fixture is only a software test"
            )
    if missing:
        print("SUBMISSION PREFLIGHT: NOT READY")
        for item in missing:
            print(" - " + item)
        raise SystemExit(2)
    print("SUBMISSION PREFLIGHT: required files and official run metadata found.")
    print("Data mode:", mode)
    print("Manual checks remain: inspect empirical report, API, Docker, review queue and AI disclosure.")

if __name__ == "__main__":
    main()
