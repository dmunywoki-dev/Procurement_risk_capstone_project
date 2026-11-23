# DSA 8401 — Applied Machine Learning Capstone: Requirements Crosswalk

**Programme:** MSc Data Science & Analytics, Strathmore University  
**Student:** Dean Maingi Munywoki (113719)

This crosswalk is a project-readiness aid. Confirm the final submission format and any updated lecturer instructions against the current course portal.

| Assessment area | Evidence in this project |
|---|---|
| Data ingestion and audit | `notebooks/01_TGEE_End_to_End.ipynb`, source hash manifest and audit CSV |
| Data cleaning and feature preparation | Currency-aware extraction, missingness handling and point-in-time feature engineering in notebook 1 |
| Modelling and evaluation | TGEE branch ensemble, tabular IF/LOF baselines, rank stability, ablation, optional proxy experiment |
| Technique justification | `README.md`, notebook methodology notes and `reports/FINAL_REPORT.md` |
| Model service and deployment | `notebooks/02_FastAPI_Model_Service.ipynb`, `Dockerfile`, `docker-compose.yml` |
| Reproducibility | `run_pipeline.sh`, pinned major dependency ranges, source hashes and saved model bundle |
| Report and presentation | `reports/FINAL_REPORT.md`, `reports/PRESENTATION_SLIDES.md` |
| AI assistance disclosure | `AI_USE.md` |

**Important data-science qualification:** verified integrity outcomes are not assumed. Unsupervised metrics assess ranking behaviour; optional supervised classification metrics assess agreement with proxy rules only. The project must not present proxy labels as legal findings or confirmed fraud outcomes.
