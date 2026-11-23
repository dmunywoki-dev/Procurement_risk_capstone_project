# Revision notes: uploaded TGEE notebook to reproducible capstone

The uploaded `113719_(1).ipynb` was retained in this folder as a provenance reference. The executable implementation is `notebooks/01_TGEE_End_to_End.ipynb`.

## Preserved from the source notebook
- PPIP / OCDS nested release ingestion.
- Procurement method, category, buyer, tender, award, contract and party extraction.
- Procurement duration, fiscal time and title-keyword features.
- Benford first-digit diagnostic.
- Buyer-supplier relational features.
- Tenderer co-participation as a conditional higher-order relationship concept.
- Baseline and stacked modelling ideas.
- API, report, presentation and reproducibility deliverables.

## Corrected or redesigned
1. **Target semantics:** a rule-generated review label is named `proxy_review_flag`; it is not a verified fraud label.
2. **Target leakage:** the optional proxy model excludes the direct inputs used to generate the proxy rules. Proxy metrics are labelled as rule-agreement metrics.
3. **Temporal validation:** a chronological holdout replaces random stratified splitting.
4. **Point-in-time graph features:** buyer-supplier features are computed from prior events before updating the graph state.
5. **Bidder disclosure:** tenderer identities are counted only when supplied and linkable; missing records are not treated as single-bidder processes.
6. **Currencies:** only amounts explicitly labelled KES enter KES-specific features; currencies are not silently summed together.
7. **Multiple suppliers:** all identified suppliers are retained for relationship features; the first supplier is not used as the sole representation.
8. **Benford interpretation:** deviations are exploratory diagnostics, not evidence of threshold manipulation.
9. **Model claims:** no GCN is claimed. The uploaded code used NetworkX graph metrics, Node2Vec and a stacking classifier; the rebuilt capstone uses classical anomaly detection and point-in-time graph statistics. A neural GCN/hypergraph model is a separate research extension.
10. **Scoring:** percentiles and review bands are not described as calibrated fraud probabilities.

## Data limitation
The uploads contained the project ZIP and the notebook, but not the official five `tender-releases*.json` packages referenced by the notebook. The rebuilt project therefore includes a small synthetic fixture for software smoke testing and expects official files in `data/raw/` for empirical analysis.
