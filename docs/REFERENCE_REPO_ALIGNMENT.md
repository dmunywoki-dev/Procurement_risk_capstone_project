# DSA 8401 and AML_Book-Data Practical Alignment

Reference repository: https://github.com/NUELBUNDI/AML_Book-Data

The reference repository contains practicals on end-to-end ML, messy data, evaluation, trees/ensembles/Optuna, unsupervised learning, MLPs, CNNs, RNNs, retrieval-augmented generation, FastAPI, Docker, MLflow and drift monitoring. This capstone adapts the relevant engineering and modelling patterns to procurement review prioritisation. It does not copy the reference repository's datasets or attempt unrelated learning tasks.

| Course area / practical | Adapted evidence in this capstone | Decision |
|---|---|---|
| Week 1 — end-to-end ML lifecycle | Two notebooks, data pipeline, model artifacts, API, Docker, README and run scripts | Implemented for this procurement problem |
| Week 2 — messy data and leakage | Missingness audit, explicit KES handling, tenderer unknowns, IDs, point-in-time temporal/graph features, chronological split | Implemented and documented |
| Week 3 — evaluation | Chronological holdout, ranking comparisons, top-K overlap, ablation, seed stability | Adapted; supervised ROC/PR metrics excluded without independent labels |
| Week 4 — trees, ensembles and Optuna | Optional proxy-rule comparison exists but is disabled by default | Not used as primary fraud model; no genuine target for meaningful tuning |
| Week 5 — unsupervised practical | Isolation Forest, LOF, GMM/BIC, PCA diagnostic and score fusion | Implemented; K-Means/DBSCAN excluded because partition labels are not the intended output |
| Week 6 — MLP | None | Excluded: no independently labelled target; proxy-only deep learning would be misleading |
| Week 7 — CNN / X-ray practical | None | Excluded: this dataset contains procurement text and tabular/relational records, not image inputs |
| Week 8 — RNN / temporal practical | Prior-only temporal features and chronological evaluation | Adapted; no within-OCID sequence because the audited FY extract has one release per OCID |
| Week 9 — RAG / language practical | TF-IDF and truncated-SVD tender-text novelty | Lightweight NLP selected; RAG/LLM excluded because there is no document QA task or demonstrated retrieval requirement |
| Week 10 — FastAPI | Typed Pydantic request contract, single and batch scoring, health/model-info endpoints, TestClient | Implemented |
| Week 10 — Docker | Non-root Docker image, read-only model mount, compose health check | Implemented |
| Week 11 — MLflow practicals | Run parameters/metrics/artifacts, PyFunc model logging and registry candidate/champion aliases | Implemented as optional MLOps workflow |
| Week 11 — CI/CD | GitHub Actions for synthetic pipeline, notebook smoke test, pytest and Docker build | Implemented |
| Weeks 12–13 — drift and governance | PSI/KS covariate drift, procurement-stratum review summaries, promotion gates and human-review governance | Implemented with explicit label/concept drift limitations |

## Specific practices adapted

- Reproducible run metadata and source lineage.
- Train-only fitting of preprocessing and score reference distributions.
- Chronological holdout instead of random splitting.
- Model comparison and stability checks suited to unlabelled ranking.
- FastAPI request validation and TestClient tests.
- Docker packaging and a non-root runtime.
- MLflow experiment tracking and model registry.
- Automated CI tests and covariate drift diagnostics.

## Important exclusions

The practical repository's COVID X-ray CNN, COVID temporal RNN, loan/fraud supervised classifiers and RAG examples solve different tasks with different labels, modalities and assumptions. Their models are not transplanted into procurement anomaly prioritisation. The exclusion is intentional and forms part of the technique justification in docs/TECHNIQUE_JUSTIFICATION.md.
