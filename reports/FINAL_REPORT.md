# Final Capstone Report

## Temporal Graph-Enhanced Ensemble for Procurement-Integrity Risk Prioritisation in Kenya

**DSA 8401 — Applied Machine Learning Capstone**  
**Student:** Dean Maingi Munywoki (113719)  
**Programme:** MSc Data Science & Analytics, Strathmore University

> **Reporting status:** The project package includes a synthetic fixture for a deterministic software smoke test. No official PPIP raw data files were supplied with the two uploads. Therefore, this report describes the implemented methodology and specifies where actual empirical results must be inserted after running the project on official source files. Do not treat the synthetic fixture's output as empirical Kenya procurement findings.

## 1. Problem and intended use

Public procurement datasets contain large numbers of process records, while detailed manual review is resource-intensive. This project develops a reproducible machine-learning workflow that prioritises unusual process snapshots for human examination. It is a decision-support prototype; it does not make legal findings or determine whether a procurement process is fraudulent, corrupt or collusive.

## 2. Data and unit of analysis

The intended empirical source is Kenya's PPIP OCDS publication. The notebook ingests release packages, JSONL and gzip JSONL, records source hashes and extracts buyer, tender, party, award, supplier and contract attributes. The analysis retains the latest release snapshot per OCID to create one retrospective process-level row. Where release date is missing, tender start date is used as the event timestamp.

The public data may contain missing buyer/supplier identifiers, incomplete tenderer disclosure, missing values, mixed currencies and incomplete award/contract fields. The notebook quantifies these issues in `data_quality_audit.csv`. Amounts enter KES-specific features only when the currency is explicitly KES. An absent tenderer record is treated as unknown rather than zero competition.

**Empirical dataset details to insert after the official run:**

- Downloaded source package names and SHA-256 hashes: see `artifacts/run_metadata.json`.
- Number of source release rows: see `artifacts/run_metadata.json`.
- Number of process snapshots retained: see `artifacts/run_metadata.json`.
- Event-date coverage, identifier coverage and KES-value coverage: see `reports/tables/data_quality_audit.csv`.

## 3. Feature engineering

The project builds structured procurement attributes, title/description text, historical deviations and relational features. Temporal features compare a process with prior records in the same procurement-method/category stratum using exponentially weighted historical baselines. Buyer-supplier graph features are computed before the current process updates the historical relationship state. They include prior pair recurrence, prior supplier reach, prior buyer activity and supplier concentration across prior buyer relationships.

A conditional bidder co-participation branch is enabled only when at least 5% of records have two or more identifiable tenderers, at least 50 such records exist, and at least 10 bidder entities are linkable. Otherwise, that branch is disabled. Missing competitors are not inferred.

## 4. TGEE architecture

The TGEE composite combines percentile-normalised component scores from available branches:

1. structured tabular Isolation Forest;
2. tender-text novelty using TF-IDF, truncated SVD and Isolation Forest;
3. temporal deviations from prior records;
4. buyer-supplier historical graph features; and
5. conditional bidder co-participation statistics.

The default branch weights are fixed in the notebook and renormalised if a branch is unavailable. Review and high-priority cut-offs are calculated from the earlier training period only. The final score is a relative percentile-style prioritisation value, not a calibrated fraud probability.

## 5. Evaluation design

Records are split chronologically using `event_date`. All feature history is constructed in event order, and the later time block is held out. Preprocessing, detector fitting, text vocabulary, score-reference distributions and thresholds are learned from the earlier block.

Because the public records do not supply verified fraud/corruption labels, the main evaluation reports score stability and ranking diagnostics rather than accuracy metrics. The notebook compares the TGEE ranking with tabular-only Isolation Forest and Local Outlier Factor using Spearman rank correlation, top-K overlap, branch ablation and random-seed sensitivity.

An optional supervised experiment generates a transparent `proxy_review_flag` from four review indicators. The classifier predictors exclude the direct ingredients of those rules. Its F1, ROC-AUC and PR-AUC indicate agreement with the generated proxy rubric only. They are not evidence that the model detects actual fraud.

## 6. Results

Populate this section only after running the notebook against official PPIP/OCDS files.

| Result | Source output |
|---|---|
| Process snapshots and date coverage | `artifacts/run_metadata.json` |
| Missingness, bidder coverage and KES-value coverage | `reports/tables/data_quality_audit.csv` |
| Active branches and weights | `artifacts/run_metadata.json` |
| Unsupervised baseline ranking comparison | `reports/tables/unsupervised_model_comparison.csv` |
| Branch ablation | `reports/tables/branch_ablation.csv` |
| Random-seed sensitivity | `reports/tables/seed_sensitivity.csv` |
| Review queue | `reports/holdout_review_queue.csv` |
| Proxy-rule agreement (not fraud validation) | `reports/tables/proxy_label_model_comparison.csv` |

No numerical empirical findings are inserted here because the official source data were not included in the submitted uploads.

## 7. Deployment

The serialized model bundle is loaded by a FastAPI application defined in notebook 2. The service provides health, model-information, single-record and batch scoring endpoints. It accepts point-in-time engineered features produced by the training pipeline. The API returns branch percentiles, a composite score and a review band. It does not accept a raw isolated OCDS object and pretend to reconstruct graph history.

## 8. Limitations and ethics

Anomaly scores are sensitive to data quality, missing disclosure, legitimate differences between categories and market structure. Supplier recurrence or centrality is not proof of collusion. Exact-name entity matching can be incomplete or erroneous. A compiled snapshot does not guarantee point-in-time reconstruction of all historically public information. Benford deviations can arise for reasons unrelated to manipulation.

Scores must not be used for automatic exclusion, sanction, accusation or adverse action. Human reviewers should examine the underlying procurement documents, record reasons for review outcomes and independently verify any concern. Sensitive identifiers should be handled under applicable institutional and legal controls.

## 9. Conclusion

The capstone provides a reproducible temporal and relational anomaly-prioritisation workflow with text and conditional bidder co-participation features, model comparison, stability analysis and a model-serving API. Its empirical value depends on running the pipeline on the official source and, for supervised validation, obtaining independent and lawful outcome labels.
