#!/usr/bin/env python3
"""Build the submission report and presentation outline from a completed run."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FY_ROOT = ROOT / "artifacts" / "fy2025_2026"
PROC_ROOT = ROOT / "data" / "processed" / "fy2025_2026"
REPORT_ROOT = ROOT / "reports" / "fy2025_2026"
TABLE_ROOT = REPORT_ROOT / "tables"

def read_csv(name: str) -> pd.DataFrame:
    path = TABLE_ROOT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def _markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None,
                    max_rows: int = 30) -> str:
    if frame.empty:
        return "_No table was produced by this run._"
    selected = frame.loc[:, [c for c in (columns or frame.columns) if c in frame.columns]]
    selected = selected.head(max_rows)
    headers = [str(c) for c in selected.columns]
    rows = [[_fmt(value) for value in row] for row in selected.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)

def build() -> tuple[Path, Path]:
    metadata_path = FY_ROOT / "fy2025_2026_run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("FY2025/2026 run metadata is missing. Execute Notebook 1 first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    comparison = read_csv("unsupervised_model_comparison.csv")
    gmm = read_csv("gmm_holdout_comparison.csv")
    ablation = read_csv("branch_ablation.csv")
    stability = read_csv("seed_sensitivity.csv")
    bands = read_csv("holdout_review_band_summary.csv")
    quality = read_csv("data_quality_audit.csv")
    scores_path = PROC_ROOT / "scored_process_snapshots.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    source_rows = metadata.get("source_manifest", [])
    source_table = pd.DataFrame([
        {"source_file": item.get("file", item.get("filename", "unknown")),
         "sha256": item.get("sha256", "not recorded"),
         "bytes": item.get("size_bytes", item.get("bytes", "not recorded"))}
        for item in source_rows
    ])
    event_min = event_max = "Not available"
    if "event_date" in scores:
        dates = pd.to_datetime(scores["event_date"], errors="coerce", utc=True).dropna()
        if len(dates):
            event_min, event_max = str(dates.min().date()), str(dates.max().date())
    active = metadata.get("active_branches", [])
    weights = metadata.get("active_weights", {})
    report = f"""# Final Capstone Report

## DSA 8401 — Applied Machine Learning
**Project:** Data-Driven Procurement Risk Prioritisation Using Unsupervised Anomaly Detection and Buyer–Supplier Network Analysis  
**Programme:** MSc Data Science & Analytics, Strathmore University  
**Student:** Individual capstone submission

> **Interpretation boundary:** This system prioritises unusual procurement records for human review. Its score is not a fraud probability, does not establish misconduct, and must not be used for automatic exclusion, sanction or accusation.

## 1. Problem statement and intended use

Kenya's public procurement system generates many process records. Reviewing every record in depth is resource-intensive. The project develops an end-to-end screening workflow that ranks process snapshots by unusualness relative to an earlier reference period. Procurement analysts can use the ranking to decide which records to inspect first, while retaining responsibility for document review and independent corroboration.

The project does not attempt to predict fraud because the public dataset does not provide a sufficiently complete, independently adjudicated fraud outcome. The target is therefore a relative anomaly-prioritisation score rather than a binary misconduct label.

## 2. Data source, scope and quality

The pipeline reads the selected Kenya PPIP / OCDS source package(s), records their SHA-256 hashes and extracts nested release information. The analysis period is **{metadata.get("analysis_period", "FY2025/2026")}**, from **{metadata.get("fy_start", "1 July 2025")}** to the exclusive end date **{metadata.get("fy_end_exclusive", "1 July 2026")}**.

| Dataset property | Observed value |
|---|---:|
| Data mode | {metadata.get("data_mode", "Not recorded")} |
| Source releases loaded | {metadata.get("records_loaded", "Not recorded")} |
| FY cohort release rows | {metadata.get("fy_cohort_release_rows", "Not recorded")} |
| Process snapshots | {metadata.get("process_snapshots", "Not recorded")} |
| Unique OCIDs | {metadata.get("unique_ocids", "Not recorded")} |
| Training records | {metadata.get("train_records", "Not recorded")} |
| Chronological holdout records | {metadata.get("holdout_records", "Not recorded")} |
| Holdout cutoff | {metadata.get("cutoff_date", "Not recorded")} |
| Event-date range in scored data | {event_min} to {event_max} |

### Source provenance

{_markdown_table(source_table, ["source_file", "sha256", "bytes"])}

### Data quality audit

{_markdown_table(quality, ["check", "value", "interpretation"], 40)}

Missing values are retained as unknown where absence is not evidence of absence. Tenderer disclosure gaps are not treated as zero competition. Financial features use only explicitly KES-denominated positive values; other currencies are not silently converted. Supplier linkage prefers source identifiers; fuzzy name-based entity merging is not performed.

## 3. Feature engineering and architecture

The Temporal Graph-Enhanced Ensemble (TGEE) is built from complementary feature branches:

- **Structured procurement features:** duration, award value where comparable, procurement method, category, tenderer-disclosure indicators and other available attributes.
- **Tender text novelty:** TF-IDF with truncated SVD and an Isolation Forest novelty detector. Vocabulary and projection are fitted on training records only.
- **Temporal deviations:** current tender duration and value are compared with earlier records in the same procurement-method/category strata.
- **Buyer–supplier history:** prior pair recurrence, supplier reach and relationship concentration are computed before the current process updates the relationship history.
- **Conditional bidder co-participation:** enabled only when the source has sufficient identifiable multi-tenderer coverage; no missing bidders are inferred.

Active branches: **{", ".join(active) or "Not recorded"}**.  
Fusion weights: **{json.dumps(weights)}**.

The API accepts prepared point-in-time feature records. It does not claim to reconstruct the historical graph from an isolated raw OCDS record.

## 4. Modelling choices and justification

Isolation Forest is the primary structured anomaly detector because the task is ranking unusual observations without labels. Local Outlier Factor supplies a neighbourhood-density baseline, evaluated on a seeded sample for computational tractability. A diagonal-covariance Gaussian Mixture Model is a density-based challenger with component count selected by BIC on training data. PCA is used as a training-fitted dimensionality diagnostic, not as an outcome classifier.

The composite fuses percentile-normalised branch scores using pre-specified weights. Review thresholds are derived from the training score distribution only. The optional proxy-label classifier is disabled by default; any proxy experiment measures agreement with its own transparent review rules, not confirmed fraud.

## 5. Temporal evaluation design

Records are sorted chronologically and divided into an earlier training reference and a later holdout. Imputation, scaling, text vocabulary, anomaly detectors, score-reference distributions and review thresholds are fitted on the training period. Temporal and buyer–supplier features use only records earlier than the current process.

Because independent outcome labels are unavailable, the evaluation reports ranking behaviour rather than accuracy. No fraud precision, recall, F1, ROC-AUC, PR-AUC or probability calibration is claimed.

### Unsupervised ranking comparison

{_markdown_table(comparison, ["model", "spearman_vs_tgee", "top_k_overlap_with_tgee", "top_k", "evaluation_population"], 20)}

### Gaussian Mixture challenger

{_markdown_table(gmm, ["model", "components_selected_by_bic", "spearman_vs_tgee", "top_k_overlap_with_tgee", "converged"], 10)}

### Branch ablation and seed stability

{_markdown_table(ablation, ["omitted_branch", "spearman_vs_full_tgee", "top_k_overlap_vs_full_tgee"], 20)}

{_markdown_table(stability, ["comparison", "spearman_rank_correlation", "top_k_overlap", "top_k"], 10)}

### Holdout review bands

{_markdown_table(bands, ["review_band", "records", "share_pct"], 10)}

The ranking comparisons measure similarity to the TGEE reference ranking; they do not measure correctness against real procurement outcomes. The top-K review queue is a workload prioritisation aid for human inspection.

## 6. Deployment and reproducibility

The repository provides a scripted notebook pipeline, a standalone FastAPI application, Docker packaging and automated tests. Run:

1. Install base dependencies from requirements.txt.
2. Place the official FY source file(s) in data/raw/.
3. Run bash run_pipeline.sh from the repository root.
4. Run bash run_api.sh or docker compose up --build.
5. Run tests with pytest -q.

The API exposes GET /health, GET /model-info, POST /score and POST /score/batch. It validates required prepared-feature fields and returns the component percentiles, composite score and review band. The deployed service is not a raw OCDS ingestion service.

Optional MLOps tracking uses MLflow for run parameters, metrics, provenance, artifacts and a registered PyFunc scoring model. A fail-closed promotion gate checks official-data provenance, artifact integrity, chronological split, evaluation outputs and seed stability before assigning the champion alias.

## 7. Monitoring, responsible use and governance

The retrospective monitor compares training-reference and chronological-holdout feature distributions using Population Stability Index (PSI) and Kolmogorov–Smirnov statistics for numeric features. It also summarises score and review-band distributions across procurement method and category. These are covariate and operational triage diagnostics, not demographic fairness evidence.

Label drift and concept drift cannot be assessed without independently adjudicated outcome labels. Group differences may reflect legitimate procurement composition, disclosure completeness or institutional mandates; they must trigger contextual review, not automatic adverse action. The project excludes personal demographic attributes from the modelling feature set and recommends role-based access, source traceability, retention controls and documented human decisions under applicable institutional policy and the Kenya Data Protection Act, 2019.

## 8. Limitations

The PPIP dataset is a public disclosure snapshot and may not reproduce exactly what was visible at every historical date. Missing fields can reflect reporting practices rather than process behaviour. A single FY file with one release per OCID does not support lifecycle-transition modelling. Supplier identifiers can be incomplete, and exact-name fallback can split or conflate entities. Graph recurrence is descriptive and does not prove preferential treatment or collusion. Anomaly scores may reflect legitimate differences between categories, buyer mandates or market structures.

## 9. Conclusion

The project delivers a reproducible, temporally evaluated, unsupervised review-prioritisation system with text and buyer–supplier historical features, a conditional co-participation branch, model comparisons, stability checks, an API, Docker packaging, automated tests, monitoring and optional MLflow registry controls. Its proper use is to support accountable human review—not to make a legal or fraud determination.

## 10. References

- Open Contracting Partnership. Kenya: Public Procurement Regulatory Authority dataset profile. https://data.open-contracting.org/en/publication/147
- Public Procurement Information Portal. OCDS data. https://tenders.go.ke/ocds
- Strathmore University. DSA 8401 Applied Machine Learning Capstone brief and course materials.
"""
    report_path = ROOT / "reports" / "FINAL_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    slides = f"""# DSA 8401 Capstone — Presentation Outline

## Slide 1 — Title
Data-Driven Procurement Risk Prioritisation Using Unsupervised Anomaly Detection and Buyer–Supplier Network Analysis.

## Slide 2 — Problem and intended use
Prioritise unusual public procurement records for human review. State clearly that the output is not a fraud probability or allegation.

## Slide 3 — Dataset and real-world complications
Source: Kenya PPIP / OCDS. Data mode: {metadata.get("data_mode")}. Process snapshots: {metadata.get("process_snapshots")}; train: {metadata.get("train_records")}; chronological holdout: {metadata.get("holdout_records")}. Discuss missing tenderer disclosure, invalid tender dates, missing/mixed-currency award values, supplier identity and reporting patterns.

## Slide 4 — Data preparation and leakage controls
Explain source hashes, FY filtering, unique OCID verification, currency-aware amount handling, point-in-time graph/temporal features, train-only preprocessing and chronological holdout.

## Slide 5 — Feature and system architecture
Show the tabular, text, temporal, buyer–supplier graph and conditional co-participation branches and percentile score fusion.

## Slide 6 — Modelling choices
Isolation Forest primary; LOF baseline; GMM/BIC challenger; PCA diagnostic. Explain why no supervised fraud classifier, neural graph, CNN, RNN or Transformer was justified by the data.

## Slide 7 — Evaluation design
State cutoff {metadata.get("cutoff_date")}; train {metadata.get("train_records")}; holdout {metadata.get("holdout_records")}. Explain Spearman ranking similarity, top-K overlap, ablation and seed stability. No independently verified target labels.

## Slide 8 — Results
Use the exact tables in reports/FINAL_REPORT.md and figures in reports/fy2025_2026/figures. Explain observed values, not synthetic values.

## Slide 9 — Review queue and explanations
Show a small anonymised sample from reports/fy2025_2026/holdout_review_queue.csv. Explain review reasons and the need for source-document corroboration.

## Slide 10 — API and Docker demonstration
Demonstrate /health, /model-info, /score and /score/batch. Explain the prepared-feature contract.

## Slide 11 — MLOps and monitoring
Show MLflow run/model registration, automated CI tests, PSI/KS feature drift diagnostics and the fail-closed promotion gate.

## Slide 12 — Limitations, ethics and conclusion
Explain no independent fraud labels, disclosure limitations, snapshot limitations, human-in-the-loop safeguards, privacy and governance.
"""
    slides_path = ROOT / "reports" / "PRESENTATION_SLIDES.md"
    slides_path.write_text(slides, encoding="utf-8")
    return report_path, slides_path

if __name__ == "__main__":
    report_file, slides_file = build()
    print(f"Built {report_file.relative_to(ROOT)}")
    print(f"Built {slides_file.relative_to(ROOT)}")
