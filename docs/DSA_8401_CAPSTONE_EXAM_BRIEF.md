# DSA 8401 Capstone — Requirements Crosswalk

**Programme:** MSc Data Science & Analytics, Strathmore University  
**Mode:** Individual end-to-end ML system  
**Weight:** 60% of the course grade

| Assessment area | Evidence in this project |
|---|---|
| Problem selection and real-world messiness | Kenya PPIP/OCDS FY2025/2026; missingness, invalid tender dates, disclosure gaps, currency ambiguity and identifier quality audited in Notebook 1 |
| Sufficient system depth | Nested data ingestion, feature engineering, chronological modelling, relational history, anomaly ranking, API, Docker, MLOps and monitoring |
| Reproducible ML pipeline | Notebook 1 plus run_pipeline.sh; source hashes and run metadata; report build script |
| Data pipeline (10%) | Ingestion, FY filtering, uniqueness check, quality audit, currency-aware cleaning and point-in-time feature engineering |
| Modelling and evaluation (20%) | TGEE, Isolation Forest and LOF comparison, GMM/BIC challenger, PCA diagnostics, temporal holdout, top-K overlap, ablation and stability |
| Technique justification (10%) | docs/TECHNIQUE_JUSTIFICATION.md and docs/REFERENCE_REPO_ALIGNMENT.md |
| Deployment (10%) | api_service.py, Pydantic request contract, TestClient coverage, Dockerfile and docker-compose.yml |
| Final report and presentation (10%) | reports/FINAL_REPORT.md and reports/PRESENTATION_SLIDES.md, generated from run evidence |
| AI disclosure | AI_USE.md |
| MLOps and quality controls | GitHub Actions, pytest, optional MLflow experiment tracking/registry, PSI/KS monitoring and promotion gate |

## Critical evaluation qualification
Independent procurement-integrity outcome labels are not available in the public snapshot. The primary model is therefore unsupervised anomaly prioritisation. Ranking similarity, score stability and GMM density diagnostics are reported as such. Optional proxy-rule classification metrics are not fraud accuracy and must not be described as confirmed misconduct detection.

## Required final-run evidence
The official-source execution is a local action because raw PPIP procurement files are intentionally not committed to the repository. Run bash run_pipeline.sh, inspect the generated run metadata and report, and use python scripts/preflight_submission.py to ensure a synthetic fixture has not been mistaken for the official empirical run.
