# Data Dictionary and Feature Families

| Field / family | Meaning | Notes |
|---|---|---|
| `ocid`, `release_id` | OCDS process and release identifiers | Latest release snapshot per OCID is retained. |
| `event_date` | Release date, falling back to tender start date | Used for chronological ordering and holdout construction. |
| `buyer_key` | Buyer ID or exact normalised name fallback | No fuzzy entity resolution. |
| `supplier_keys` | All supplier IDs/names identified in award records | Multiple suppliers are retained; not reduced to the first supplier. |
| `num_tenderers` | Count of identifiable parties with the `tenderer` role | Missing disclosure remains unknown; not imputed as zero. |
| `award_value_kes` | Sum of award values explicitly denominated in KES | Other currencies are not converted or added to KES. |
| `duration_days` | Tender end minus tender start in days | Non-positive or implausibly long periods are marked invalid and set missing. |
| `duration_historical_abs_z`, `value_historical_abs_z` | Absolute deviation from earlier values in the same method/category stratum | Expanding EWMA is shifted so each record uses prior history only. |
| `buyer_prior_supplier_count` | Number of distinct suppliers previously linked to the buyer | Computed before updating with current process. |
| `buyer_supplier_prior_count` | Number of prior processes linking the buyer and a supplier | Maximum across all suppliers on the current process. |
| `supplier_prior_buyer_count` | Number of distinct buyers previously linked to a supplier | Maximum across all current suppliers. |
| `supplier_prior_buyer_hhi` | Concentration of a supplier's prior buyer relationships by historical award count | Descriptive relational signal, not proof of favouritism. |
| `tenderer_history_max` | Maximum number of prior processes associated with an identified tenderer | Available only when tenderer identities are disclosed. |
| `cobidder_repeat_max` | Maximum prior co-appearance count for an identified tenderer pair | Conditional higher-order participation feature; not a collusion finding. |
| `tabular_score`, `text_score`, etc. | Component percentile scores | Calibrated against the training score distribution, not outcome labels. |
| `tgee_priority_score` | Weighted composite of active component percentiles | Relative review priority only; not a probability. |
| `proxy_review_flag` | Rule-generated weak label used only in the optional secondary classifier | Never use as verified fraud/corruption truth. |
