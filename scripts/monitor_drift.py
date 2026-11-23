#!/usr/bin/env python3
"""Retrospective feature-drift and review-band monitoring.

With no independently adjudicated outcome labels, this script measures
covariate drift only. It does not claim to measure label or concept drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "data" / "processed" / "fy2025_2026"
    / "scored_process_snapshots.csv"
)
EXCLUDE = {
    "ocid", "release_id", "buyer_name", "event_date", "data_split",
    "record_type", "combined_text", "supplier_names_display",
    "tgee_priority_score", "review_band", "tabular_score", "text_score",
    "temporal_score", "graph_score", "hypergraph_score",
    "dominant_component", "review_reason",
}


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(float)
    if len(ref) < 5 or len(cur) < 5:
        return float("nan")
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        return 0.0
    edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    ref_pct = np.maximum(ref_counts / len(ref), eps)
    cur_pct = np.maximum(cur_counts / len(cur), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _categorical_psi(reference: pd.Series, current: pd.Series,
                     max_categories: int = 20) -> float:
    ref = reference.fillna("<MISSING>").astype(str)
    cur = current.fillna("<MISSING>").astype(str)
    if len(ref) < 5 or len(cur) < 5:
        return float("nan")
    top = set(ref.value_counts().head(max_categories).index)
    ref_group = ref.where(ref.isin(top), "<OTHER>")
    cur_group = cur.where(cur.isin(top), "<OTHER>")
    categories = sorted(set(ref_group) | set(cur_group))
    eps = 1e-6
    ref_pct = np.array([(ref_group == c).mean() for c in categories])
    cur_pct = np.array([(cur_group == c).mean() for c in categories])
    ref_pct = np.maximum(ref_pct, eps)
    cur_pct = np.maximum(cur_pct, eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare_frames(reference: pd.DataFrame,
                   current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common = [c for c in reference.columns if c in current.columns and c not in EXCLUDE]
    for column in common:
        ref = reference[column]
        cur = current[column]
        if pd.api.types.is_numeric_dtype(ref) or pd.api.types.is_numeric_dtype(cur):
            ref_num = pd.to_numeric(ref, errors="coerce").dropna()
            cur_num = pd.to_numeric(cur, errors="coerce").dropna()
            ks_stat = ks_p = float("nan")
            if len(ref_num) >= 5 and len(cur_num) >= 5:
                result = ks_2samp(ref_num, cur_num)
                ks_stat, ks_p = float(result.statistic), float(result.pvalue)
            psi = _psi(ref_num, cur_num)
            rows.append({
                "feature": column, "feature_type": "numeric",
                "psi": psi, "ks_statistic": ks_stat, "ks_p_value": ks_p,
                "reference_n": len(ref_num), "current_n": len(cur_num),
                "drift_flag": bool(
                    (np.isfinite(psi) and psi >= 0.20)
                    or (np.isfinite(ks_p) and ks_p < 0.01)
                ),
            })
        else:
            psi = _categorical_psi(ref, cur)
            rows.append({
                "feature": column, "feature_type": "categorical",
                "psi": psi, "ks_statistic": np.nan, "ks_p_value": np.nan,
                "reference_n": int(ref.notna().sum()),
                "current_n": int(cur.notna().sum()),
                "drift_flag": bool(np.isfinite(psi) and psi >= 0.20),
            })
    return pd.DataFrame(rows).sort_values(
        ["drift_flag", "psi"], ascending=[False, False],
        na_position="last"
    )


def build_group_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise score/review bands across observed procurement strata."""
    if "tgee_priority_score" not in frame.columns:
        return pd.DataFrame()
    rows = []
    for group_col in ("procurement_method", "category"):
        if group_col not in frame.columns:
            continue
        for group_value, part in frame.groupby(group_col, dropna=False):
            if len(part) < 10:
                continue
            rows.append({
                "group_dimension": group_col,
                "group_value": str(group_value),
                "records": int(len(part)),
                "mean_priority_score": float(
                    pd.to_numeric(part["tgee_priority_score"], errors="coerce").mean()
                ),
                "high_priority_rate": float(
                    (part.get("review_band") == "high_priority").mean()
                ) if "review_band" in part else np.nan,
                "review_or_high_rate": float(
                    part.get("review_band").isin(["review", "high_priority"]).mean()
                ) if "review_band" in part else np.nan,
            })
    return pd.DataFrame(rows)


def run_monitoring(input_path: Path, output_dir: Path) -> dict:
    frame = pd.read_csv(input_path)
    if "data_split" not in frame.columns:
        raise ValueError(
            "Input must include data_split=train_reference/chronological_holdout. "
            "Re-run the updated Notebook 1 export."
        )
    reference = frame.loc[frame["data_split"] == "train_reference"].copy()
    current = frame.loc[frame["data_split"] == "chronological_holdout"].copy()
    if len(reference) < 5 or len(current) < 5:
        raise ValueError("Both reference and current periods need at least five rows.")

    drift = compare_frames(reference, current)
    groups = build_group_diagnostics(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    drift.to_csv(output_dir / "feature_drift.csv", index=False)
    groups.to_csv(output_dir / "review_band_group_diagnostics.csv", index=False)
    summary = {
        "reference_records": int(len(reference)),
        "current_records": int(len(current)),
        "features_compared": int(len(drift)),
        "features_flagged": int(drift["drift_flag"].sum()) if len(drift) else 0,
        "psi_alert_threshold": 0.20,
        "ks_p_value_alert_threshold": 0.01,
        "group_diagnostics_are_not_fairness_proof": True,
        "label_drift_status": "not_measurable_without_verified_outcome_labels",
        "concept_drift_status": "not_measurable_without_verified_outcome_labels",
        "interpretation": (
            "Covariate drift is a monitoring signal, not evidence of misconduct "
            "or model failure. Review thresholds and data-generation changes."
        ),
    }
    (output_dir / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports" / "fy2025_2026" / "monitoring",
    )
    args = parser.parse_args()
    summary = run_monitoring(args.input, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
