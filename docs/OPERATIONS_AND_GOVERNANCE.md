# Operations, Monitoring and Responsible-Use Controls

## 1. Intended operational use

The API returns a relative review-priority percentile and a review band. The model is an analyst support tool. It must not automatically reject a tenderer, cancel a procurement, initiate a sanction or assert fraud/corruption/collusion.

Only authorised procurement reviewers should access the score queue and source records. A review should be recorded with reviewer role, date, source documents examined, rationale, outcome and any corrective action. Model score alone is not an evidentiary finding.

## 2. Data lineage and reproducibility

- Retain the original PPIP/OCDS source files under controlled local storage.
- Record the SHA-256 digest, filename, source date and extraction date for each source.
- Do not commit raw procurement records, personal data, access tokens or private environment files to Git.
- Preserve the FY selection, unique-OCID check, temporal cutoff, package versions, seed, feature schema and model bundle hash in run metadata.
- Keep the generated model bundle and prepared-feature table from the same run. Do not pair an API bundle with features from a different model version.
- Treat the synthetic fixture as a test fixture, never as empirical evidence.

## 3. Feature and score monitoring

The retrospective monitor compares training-reference and chronological-holdout feature distributions.

- Numeric feature drift: Population Stability Index (PSI) and two-sample Kolmogorov–Smirnov statistic / p-value.
- Categorical feature drift: PSI after grouping the most frequent reference categories and a residual OTHER category.
- Operational review diagnostics: mean priority score and review/high-priority rates by procurement method and category.

The default PSI alert level is 0.20; the KS p-value diagnostic threshold is 0.01. These are investigation triggers, not automatic model-failure criteria. Distribution tests can flag large samples for small practical differences, so reviewers must inspect effect size, source changes and data quality before action.

The current data does not contain independent adjudicated outcomes. Label drift and concept drift are explicitly marked not measurable. If verified labels become available, monitor label prevalence, ranking utility, calibration and error rates over time using a documented review process.

## 4. Fairness and subgroup interpretation

The source does not supply a defensible set of protected demographic attributes for fairness evaluation. The project therefore does not claim demographic parity, equalised odds or fairness certification. It reports only operational group summaries for procurement method and category.

Differences across buyers, procurement methods or categories may reflect mandate, size, market conditions, reporting completeness or legitimate procedural differences. Do not interpret score distributions as evidence of bias or wrongdoing. Any future fairness assessment requires a lawful basis, suitable attributes, a clear harm model and domain review.

## 5. Privacy and data protection

Follow institutional data-handling requirements and applicable Kenyan data-protection law, including the Data Protection Act, 2019. Use data minimisation, purpose limitation, access controls, secure local storage, limited retention and documented disclosure. Do not expose unnecessary supplier or buyer personal contact details in presentation examples. Use anonymised process identifiers for public demonstrations.

## 6. Human review protocol

1. Confirm the process identifier and retrieve the original public procurement documents.
2. Read the review reasons and component scores; do not treat them as proof.
3. Check whether missingness or disclosure patterns explain the anomaly.
4. Compare the process with appropriate peers (method, category, buyer context and time).
5. Obtain independent documentary corroboration before escalating.
6. Record the reviewer decision and reasons, including cases where the score is not actionable.
7. Feed adjudicated review outcomes into a separately governed validation dataset only where lawful.

## 7. Model release controls

The MLflow promotion gate requires official-data provenance, source hashes, artifact hash integrity, an adequate chronological split, evaluation outputs, a converged GMM, a non-empty review queue and configured seed-stability thresholds. Passing these checks means the build is operationally reproducible and ranking-stable under the checked perturbation; it does not establish fraud-prediction validity.

A failed gate blocks the champion alias. Do not lower a threshold simply to obtain a successful release. Investigate the data, modelling stability and evaluation evidence first.

## 8. Known limitations

- Compiled public snapshots may not reconstruct what was visible at every earlier date.
- Missing tenderer or financial disclosure is not proof that an event did not occur.
- Exact-name fallback is not robust entity resolution; fuzzy matching is intentionally excluded.
- One release per OCID in the FY cohort does not support lifecycle sequence modelling.
- A high anomaly score can reflect legitimate procurement heterogeneity.
- Without independent outcome labels, supervised performance and concept drift remain unknown.
