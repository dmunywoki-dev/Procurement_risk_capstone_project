# Presentation Deck Outline — DSA 8401 Capstone

## Slide 1 — Title
**Temporal Graph-Enhanced Ensemble for Procurement-Integrity Risk Prioritisation in Kenya**  
Dean Maingi Munywoki | MSc Data Science & Analytics | Strathmore University

## Slide 2 — Problem
- Large procurement datasets make exhaustive manual review difficult.
- Structured fields, tender text, time patterns and buyer-supplier relationships provide complementary context.
- Aim: prioritise unusual records for human review, not automate a finding of misconduct.

## Slide 3 — Data and constraints
- Source: Kenya PPIP OCDS release packages.
- Unit: latest available process snapshot per OCID.
- Key constraints to present from `data_quality_audit.csv`: missing buyer/supplier identifiers, tenderer disclosure, date coverage, KES amount coverage and duplicate handling.
- Clarify whether figures are from official data or the synthetic smoke-test fixture.

## Slide 4 — Data preparation
- Nested OCDS extraction and source hashes.
- Currency-aware award aggregation (KES only for KES features).
- Unknown bidder data are not treated as zero competition.
- Exact ID/name linkage with quality caveats.

## Slide 5 — Feature engineering
- Structured and text features.
- Historical temporal deviations.
- Prior buyer-supplier recurrence, supplier reach and concentration.
- Optional bidder co-participation branch, activated only after a disclosure-coverage gate.

## Slide 6 — TGEE architecture
- Tabular Isolation Forest.
- Text TF-IDF / SVD novelty branch.
- Temporal historical-deviation branch.
- Buyer-supplier graph-history branch.
- Conditional bidder co-participation branch.
- Percentile normalisation and weighted score fusion.

## Slide 7 — Evaluation design
- Chronological train / holdout split.
- Branch models and score references fit on earlier records.
- Main evaluation: rank correlation, top-K overlap, ablation and seed sensitivity.
- No fraud metrics without verified outcome labels.

## Slide 8 — Results
Insert actual values/figures from:
- `unsupervised_model_comparison.csv`
- `branch_ablation.csv`
- `seed_sensitivity.csv`
- `eda_overview.png` and `tgee_holdout_score_distribution.png`
Do not use synthetic fixture results as empirical findings.

## Slide 9 — Review queue and explainability
- Show a few anonymised review-queue examples.
- Explain branch scores, dominant component and visible rule signals.
- Emphasise score = relative unusualness / prioritisation, not guilt or fraud probability.

## Slide 10 — API and reproducibility
- Show `/health`, `/model-info`, `/score` and `/score/batch`.
- Demo request/response from the API notebook.
- Show notebook execution command and generated artifact paths.

## Slide 11 — Limitations and ethics
- Public data quality and disclosure gaps.
- Compiled snapshot limits point-in-time reconstruction.
- Name resolution uncertainty.
- No adjudicated labels; proxy experiment only.
- Human review and independent corroboration required.

## Slide 12 — Conclusion and next steps
- Run on official PPIP data and report real audit results.
- Seek lawful, independently adjudicated review outcomes.
- Add buyer-disjoint/supplier-disjoint evaluation when outcomes are available.
- Implement neural graph/hypergraph models only if data coverage and independent validation justify them.
