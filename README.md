# DSA 8401 Capstone — Kenya PPIP Procurement Integrity TGEE

**Student:** Dean Maingi Munywoki (113719)  
**Programme:** MSc Data Science & Analytics, Strathmore University  
**Project:** Temporal Graph-Enhanced Ensemble for procurement-integrity risk prioritisation

## 1. What this project does

This capstone builds a reproducible screening prototype over Kenya's PPIP/Open Contracting Data Standard (OCDS) procurement process records. The Temporal Graph-Enhanced Ensemble (TGEE) combines:

- **Structured tabular anomalies:** procurement method, tender duration, KES-denominated award values, tenderer disclosure, and procurement category.
- **Tender-text novelty:** TF-IDF / truncated-SVD representation of available titles and descriptions.
- **Temporal deviations:** current tender duration and award value compared with prior records in the same method/category stratum.
- **Buyer-supplier graph history:** prior buyer-supplier recurrence, supplier reach and concentration features computed before the current process updates the relationship history.
- **Conditional bidder co-participation:** a hypergraph-inspired co-bidding branch that is enabled only when enough multi-tenderer identities are disclosed and linkable.

The output is a relative review-priority score and a queue for human inspection. It is **not** a probability of fraud and does not establish corruption, collusion or illegality.

## 2. Important methodological corrections

The submitted TGEE notebook used a `is_high_risk` target generated from the same red-flag and graph variables later supplied to supervised models. It also calculated network features across the full dataset before a random train/test split. Those choices make the reported classifier metrics unsuitable as evidence of fraud detection.

This rebuilt project therefore:

1. uses a chronological holdout rather than a random stratified split;
2. calculates temporal and graph-history features from prior records only;
3. keeps missing tenderer identities unknown rather than converting them to zero bidders;
4. aggregates award amounts only where the source currency is explicitly KES;
5. treats Benford's Law as an exploratory diagnostic, not evidence of manipulation;
6. makes unsupervised anomaly ranking the primary empirical output because verified integrity labels are unavailable;
7. retains a secondary supervised stacking experiment only as agreement with a transparent, rule-generated **proxy** label; it must not be described as validated fraud prediction;
8. does not claim a GCN: this implementation uses classical anomaly detectors and graph/co-participation statistics.

The uploaded research proposal is preserved in `docs/Procurement_Integrity_Proposal_Final.pdf`. A bidder-hypergraph neural model or GCN is not implemented in this capstone build; the optional co-participation features are a conditional, non-neural precursor.

## 3. Project structure

```text
DSA8401_Procurement_Integrity_TGEE/
├── notebooks/
│   ├── 01_TGEE_End_to_End.ipynb
│   └── 02_FastAPI_Model_Service.ipynb
├── data/
│   ├── raw/                    # Place official PPIP/OCDS data here
│   ├── sample/                 # Synthetic software smoke-test fixture only
│   └── processed/              # Generated feature and score files
├── artifacts/                  # Saved model bundle and run metadata
├── reports/
│   ├── FINAL_REPORT.md
│   ├── PRESENTATION_SLIDES.md
│   ├── figures/
│   └── tables/
├── docs/                       # Exam brief, proposal and revision notes
├── run_pipeline.sh
├── run_api.sh
├── Dockerfile
└── docker-compose.yml
```

Core implementation logic is kept in Jupyter notebooks. Shell scripts are orchestration wrappers.

## 4. Data setup

The end-to-end notebook accepts:

- OCDS JSON packages containing a top-level `releases` array (e.g. `tender-releases-*.json`);
- JSON Lines (`.jsonl`); or
- gzip-compressed JSON Lines (`.jsonl.gz`).

Copy the official files into `data/raw/`. Do not place unrelated JSON files in that folder. The notebook records a SHA-256 hash and basic file provenance for each source.

```bash
mkdir -p data/raw
# Copy official downloaded PPIP/OCDS files into data/raw/
```

When `data/raw/` is empty, the notebook automatically selects `data/sample/synthetic_ocds_releases.json` for a deterministic smoke test. The notebook prints a warning. Synthetic results must never be presented as empirical PPIP findings.

**Data-handling decisions:**

- One latest release snapshot is retained per OCID for retrospective process-level analysis.
- `event_date` uses release date, falling back to tender start date.
- Tenderer counts are considered observed only when identifiable tenderer parties are published.
- Award values enter the KES value features only where currency is explicitly `KES`.
- Buyer / supplier matching uses source IDs, falling back to exact normalised names. Fuzzy entity matching is not silently performed.

## 5. Setup (Python 3.11–3.13)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Run the capstone pipeline

From the project root:

```bash
bash run_pipeline.sh
```

Or open `notebooks/01_TGEE_End_to_End.ipynb` in Jupyter and run all cells in order.

The pipeline produces:

- `data/processed/scored_process_snapshots.csv`
- `data/processed/api_prepared_features.csv` (prepared API input contract)
- `reports/holdout_review_queue.csv`
- `reports/tables/data_quality_audit.csv`
- `reports/tables/unsupervised_model_comparison.csv`
- `reports/tables/branch_ablation.csv`
- `reports/tables/bipartite_graph_diagnostics.csv`
- `reports/tables/benford_by_method.csv`
- `reports/tables/proxy_label_model_comparison.csv` (when proxy modelling is feasible)
- `artifacts/tgee_model_bundle.joblib`
- `artifacts/run_metadata.json`
- `reports/figures/*.png`

Use the review queue as an analyst shortlist. A high score means that the record is unusual relative to the model's training reference, not that misconduct occurred.

## 7. API and Docker

Run the API after executing the pipeline:

```bash
bash run_api.sh
```

Open `http://127.0.0.1:8000/docs`.

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /score`
- `POST /score/batch`

The API accepts **prepared point-in-time feature records**, not raw nested OCDS releases. Graph and temporal history must be generated by the same feature-engineering logic as notebook 1. This prevents the service from pretending it can reconstruct historical context from one isolated raw tender.

Docker:

```bash
docker compose up --build
```

Train the model before starting the API container. Generated `artifacts/` are mounted read-only.

## 8. How to interpret evaluation

Because no verified fraud/corruption target is available in the public records, the main evaluation does not claim predictive accuracy. It reports:

- chronological holdout coverage and score distributions;
- score-rank correlation and top-K overlap against tabular-only anomaly baselines (LOF uses a fixed seeded sample for scale);
- branch ablation sensitivity;
- random-seed stability;
- data quality and tenderer-identity coverage.

The optional proxy classifier reports F1 / ROC-AUC / PR-AUC only against a deterministic rule-derived proxy flag. These results are **proxy-rule agreement**, not validation against confirmed procurement misconduct. For genuine supervised validation, obtain a lawful, independently adjudicated and sufficiently representative set of investigation/audit outcomes.

## 9. Limitations and next work

The source is a compiled public snapshot and may not reproduce exactly what was known at each historical moment. Public disclosure may be incomplete; an absent tenderer, award or contract field is not evidence that the event did not occur. Exact-name entity resolution can split the same supplier across spelling variants or combine unrelated entities with similar names. Current review bands are capacity settings and should be confirmed with procurement auditors.

The next research stage is to obtain verified review outcomes, assess buyer-disjoint and supplier-disjoint holdouts, add calibrated ranking metrics such as Precision@K / Recall@K / NDCG@K where valid relevance labels exist, and implement a true graph/hypergraph neural model only if data coverage and compute justify it.

## 10. References

- Open Contracting Partnership. Kenya: Public Procurement Regulatory Authority dataset profile. https://data.open-contracting.org/en/publication/147
- Public Procurement Information Portal. OCDS data. https://tenders.go.ke/ocds
- Strathmore University, DSA 8401 Applied Machine Learning course materials and exam brief (see `docs/`).
