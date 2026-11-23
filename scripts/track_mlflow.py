#!/usr/bin/env python3
"""Track an FY2025/2026 TGEE run and register its scoring interface in MLflow."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
FY_ROOT = ROOT / "artifacts" / "fy2025_2026"
REPORT_ROOT = ROOT / "reports" / "fy2025_2026"
MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL", "tgee-procurement-prioritisation")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def _metric_from_csv(path: Path, row_mask, column: str):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    selected = frame.loc[row_mask(frame)]
    if selected.empty or column not in selected:
        return None
    value = pd.to_numeric(selected.iloc[0][column], errors="coerce")
    return float(value) if pd.notna(value) else None


def main() -> None:
    metadata_path = FY_ROOT / "fy2025_2026_run_metadata.json"
    bundle_path = FY_ROOT / "tgee_fy2025_2026_model_bundle.joblib"
    if not metadata_path.is_file() or not bundle_path.is_file():
        raise FileNotFoundError(
            "Run the FY2025/2026 Notebook 1 pipeline before MLflow tracking."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    comparison_path = REPORT_ROOT / "tables" / "unsupervised_model_comparison.csv"
    gmm_path = REPORT_ROOT / "tables" / "gmm_holdout_comparison.csv"
    stability_path = REPORT_ROOT / "tables" / "seed_sensitivity.csv"
    for required in (comparison_path, gmm_path, stability_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required evaluation output is missing: {required}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    experiment_name = "DSA8401-PPIP-FY2025-2026"
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    comparison = pd.read_csv(comparison_path)
    gmm = pd.read_csv(gmm_path)
    stability = pd.read_csv(stability_path)
    metrics = {
        "records_total": float(metadata.get("process_snapshots", 0)),
        "records_train": float(metadata.get("train_records", 0)),
        "records_holdout": float(metadata.get("holdout_records", 0)),
        "unique_ocids": float(metadata.get("unique_ocids", 0)),
    }

    if "model" in comparison:
        baseline = comparison.loc[
            comparison["model"].str.contains("Isolation Forest", case=False, na=False)
        ]
        if not baseline.empty:
            for source, target in (
                ("spearman_vs_tgee", "tgee_vs_iforest_spearman"),
                ("top_k_overlap_with_tgee", "tgee_vs_iforest_topk_overlap"),
            ):
                if source in baseline:
                    value = pd.to_numeric(baseline.iloc[0][source], errors="coerce")
                    if pd.notna(value):
                        metrics[target] = float(value)
    if not gmm.empty:
        for source, target in (
            ("spearman_vs_tgee", "tgee_vs_gmm_spearman"),
            ("top_k_overlap_with_tgee", "tgee_vs_gmm_topk_overlap"),
        ):
            if source in gmm:
                value = pd.to_numeric(gmm.iloc[0][source], errors="coerce")
                if pd.notna(value):
                    metrics[target] = float(value)
    if not stability.empty:
        for source, target in (
            ("spearman_rank_correlation", "seed_rank_spearman"),
            ("top_k_overlap", "seed_topk_overlap"),
        ):
            if source in stability:
                value = pd.to_numeric(stability.iloc[0][source], errors="coerce")
                if pd.notna(value):
                    metrics[target] = float(value)

    sys.path.insert(0, str(ROOT))
    from mlflow_tgee_model import TgeePyfuncModel

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="tgee-fy2025-2026",
    ) as run:
        params = {
            "model_version": "2.1.0-fy2025-2026",
            "data_mode": str(metadata.get("data_mode", "unknown")),
            "analysis_period": str(metadata.get("analysis_period", "unknown")),
            "cutoff_date": str(metadata.get("cutoff_date", "unknown")),
            "active_branches": ",".join(metadata.get("active_branches", [])),
            "unique_ocids": str(metadata.get("unique_ocids", 0)),
            "proxy_classifier_trained": str(metadata.get("proxy_classifier_trained", False)),
            "seed": "42",
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "problem_type": "unsupervised_anomaly_prioritisation",
            "score_semantics": "relative_review_priority_not_fraud_probability",
            "validation_labels": "no_independent_adjudicated_outcomes",
            "source": "Kenya PPIP OCDS FY2025/2026",
            "student_work": "individual_capstone",
        })
        mlflow.log_artifact(str(metadata_path), artifact_path="run_metadata")
        mlflow.log_artifact(str(bundle_path), artifact_path="serialized_bundle")
        for path in (comparison_path, gmm_path, stability_path):
            mlflow.log_artifact(str(path), artifact_path="evaluation")
        queue = REPORT_ROOT / "holdout_review_queue.csv"
        if queue.exists():
            mlflow.log_artifact(str(queue), artifact_path="review_queue")

        model_info = mlflow.pyfunc.log_model(
            name="tgee_fy2025_2026",
            python_model=TgeePyfuncModel(),
            artifacts={"model_bundle": str(bundle_path)},
            code_paths=[
                str(ROOT / "api_service.py"),
                str(ROOT / "mlflow_tgee_model.py"),
            ],
            pip_requirements=[
                "mlflow>=3.0,<4",
                "numpy>=1.26,<3",
                "pandas>=2.2,<3",
                "scikit-learn>=1.6,<2",
                "joblib>=1.4,<2",
            ],
            registered_model_name=MODEL_NAME,
        )
        version = getattr(model_info, "registered_model_version", None)
        if version is not None:
            client.set_registered_model_alias(MODEL_NAME, "candidate", str(version))

        print(json.dumps({
            "tracking_uri": TRACKING_URI,
            "experiment": experiment_name,
            "run_id": run.info.run_id,
            "registered_model": MODEL_NAME,
            "registered_version": version,
            "candidate_alias_set": version is not None,
            "metrics_logged": metrics,
        }, indent=2))


if __name__ == "__main__":
    main()
