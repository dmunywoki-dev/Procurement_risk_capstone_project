#!/usr/bin/env python3
"""Fail-closed promotion gate for the registered TGEE model.

The gates validate provenance, temporal evaluation artefacts and rank stability.
They do not establish predictive accuracy because independent outcome labels
are not available for this procurement snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
FY_ROOT = ROOT / "artifacts" / "fy2025_2026"
REPORT_ROOT = ROOT / "reports" / "fy2025_2026"
MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL", "tgee-procurement-prioritisation")
DEFAULT_MIN_SPEARMAN = 0.70
DEFAULT_MIN_TOPK = 0.50


def evaluate_gates(min_spearman: float, min_topk: float) -> dict:
    metadata_path = FY_ROOT / "fy2025_2026_run_metadata.json"
    bundle_path = FY_ROOT / "tgee_fy2025_2026_model_bundle.joblib"
    checks = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"gate": name, "passed": bool(passed), "evidence": evidence})

    if not metadata_path.exists() or not bundle_path.exists():
        add("model_artifact_and_metadata", False, "Required FY model files are absent.")
        return {"passed": False, "checks": checks}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    add(
        "official_source_only",
        metadata.get("data_mode") == "official_fy2025_2026",
        f"data_mode={metadata.get('data_mode')}",
    )
    add(
        "source_provenance",
        bool(metadata.get("source_manifest"))
        and all(item.get("sha256") for item in metadata.get("source_manifest", [])),
        f"source files recorded={len(metadata.get('source_manifest', []))}",
    )
    add(
        "artifact_integrity",
        digest == metadata.get("model_bundle_sha256"),
        f"sha256_match={digest == metadata.get('model_bundle_sha256')}",
    )
    train_n = int(metadata.get("train_records", 0))
    holdout_n = int(metadata.get("holdout_records", 0))
    add("chronological_partition", train_n >= 30 and holdout_n >= 10,
        f"train={train_n}; holdout={holdout_n}")
    snapshots = int(metadata.get("process_snapshots", 0))
    unique = int(metadata.get("unique_ocids", 0))
    add("one_process_per_ocid", snapshots > 0 and snapshots == unique,
        f"snapshots={snapshots}; unique_ocids={unique}")

    comparison_path = REPORT_ROOT / "tables" / "unsupervised_model_comparison.csv"
    gmm_path = REPORT_ROOT / "tables" / "gmm_holdout_comparison.csv"
    stability_path = REPORT_ROOT / "tables" / "seed_sensitivity.csv"
    ablation_path = REPORT_ROOT / "tables" / "branch_ablation.csv"
    queue_path = REPORT_ROOT / "holdout_review_queue.csv"

    required_files = [comparison_path, gmm_path, stability_path, ablation_path, queue_path]
    add("evaluation_outputs_present", all(path.exists() for path in required_files),
        "; ".join(path.name for path in required_files if path.exists()))

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        models = comparison.get("model", pd.Series(dtype=str)).astype(str)
        has_iforest = models.str.contains("Isolation Forest", case=False).any()
        has_lof = models.str.contains("Local Outlier Factor", case=False).any()
        has_tgee = models.str.contains("TGEE", case=False).any()
        add("baseline_comparison_complete", has_iforest and has_lof and has_tgee,
            f"IsolationForest={has_iforest}; LOF={has_lof}; TGEE={has_tgee}")
    else:
        add("baseline_comparison_complete", False, "Comparison table absent.")

    if gmm_path.exists():
        gmm = pd.read_csv(gmm_path)
        converged = "converged" in gmm and bool(gmm["converged"].iloc[0])
        add("gmm_converged", converged, f"converged={converged}")
    else:
        add("gmm_converged", False, "GMM holdout comparison absent.")

    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        spearman = float(stability["spearman_rank_correlation"].iloc[0])
        topk = float(stability["top_k_overlap"].iloc[0])
        add("seed_rank_stability", spearman >= min_spearman,
            f"spearman={spearman:.4f}; required>={min_spearman:.2f}")
        add("seed_topk_stability", topk >= min_topk,
            f"top_k_overlap={topk:.4f}; required>={min_topk:.2f}")
    else:
        add("seed_rank_stability", False, "Seed sensitivity table absent.")
        add("seed_topk_stability", False, "Seed sensitivity table absent.")

    if queue_path.exists():
        queue = pd.read_csv(queue_path)
        add("human_review_queue", len(queue) > 0 and
            {"tgee_priority_score", "review_band", "review_reason"}.issubset(queue.columns),
            f"rows={len(queue)}")
    else:
        add("human_review_queue", False, "Holdout review queue absent.")

    semantics = str(metadata.get("score_interpretation", "")).lower()
    add("safe_score_semantics", "not a calibrated probability" in semantics
        or "not a probability" in semantics,
        metadata.get("score_interpretation", "score semantics missing"))

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "promotion_semantics": (
            "Operational readiness and ranking robustness only; not verified "
            "fraud prediction or legal validation."
        ),
        "thresholds": {
            "minimum_seed_spearman": min_spearman,
            "minimum_seed_topk_overlap": min_topk,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-spearman", type=float, default=DEFAULT_MIN_SPEARMAN)
    parser.add_argument("--min-topk-overlap", type=float, default=DEFAULT_MIN_TOPK)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    result = evaluate_gates(args.min_spearman, args.min_topk_overlap)
    output_dir = REPORT_ROOT / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_promotion_gate.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result["passed"]:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        client = MlflowClient()
        try:
            candidate = client.get_model_version_by_alias(args.model_name, "candidate")
            client.set_registered_model_alias(
                args.model_name, "champion", str(candidate.version)
            )
            result["champion_version"] = str(candidate.version)
            result["champion_alias_set"] = True
        except Exception as exc:
            result["passed"] = False
            result["champion_alias_set"] = False
            result["registry_error"] = type(exc).__name__
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
