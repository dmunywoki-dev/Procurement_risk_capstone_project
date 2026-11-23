# Technique Selection and Exclusion Rationale

## 1. Learning problem

The task is unsupervised anomaly prioritisation over procurement process snapshots. The PPIP/OCDS extract does not provide independently adjudicated fraud or corruption labels. The model therefore estimates how unusual a process appears relative to an earlier reference population; it does not estimate a probability of wrongdoing.

The design selects methods for the data and the decision task. It does not attempt to implement every method covered in DSA 8401.

## 2. Selected methods

| Method | Role in the system | Why it fits |
|---|---|---|
| Isolation Forest | Primary structured and branch-level anomaly detector | Supports high-dimensional, mixed operational features after appropriate preprocessing and does not require outcome labels. |
| Local Outlier Factor | Neighbourhood-density baseline | Checks whether local-density anomalies produce a materially different holdout ranking. It is fitted on a seeded sample to control computational cost. |
| Gaussian Mixture Model | Density-based challenger | Provides a probabilistic density representation of the training feature distribution. Component count is selected using BIC; negative log density is treated as unusualness, not fraud probability. |
| PCA | Dimensionality diagnostic and visualisation | Shows the effective dimensionality and feature structure after train-only preprocessing. It is not used as a decision boundary. |
| TF-IDF + truncated SVD | Tender text representation | Procurement descriptions are short and sparse. A linear bag-of-words representation is transparent, computationally modest and can surface text novelty without needing adjudicated text labels. |
| Prior-only temporal statistics | Temporal features | Each process is compared with earlier records in the same procurement method/category strata. Shifted/expanding calculations prevent future values from informing earlier records. |
| Buyer–supplier bipartite statistics | Relational features | Procurement is naturally represented by buyer and supplier entities. Prior pair recurrence, supplier reach and concentration provide relational context without claiming a neural graph model. |
| Conditional co-participation statistics | Optional higher-order relational branch | Only activated when sufficient tenderer identities and multi-tenderer processes are disclosed. Missing tenderers are not invented. |

## 3. Why the ensemble is reasonable

The branches capture different notions of unusualness: structured attributes, text novelty, historical deviation and network history. Each raw branch score is converted to a percentile against its own training-period reference distribution. Active branches are fused using pre-specified weights, renormalised if a branch is unavailable. Review and high-priority thresholds are quantiles of the training score distribution.

This is a pragmatic score-fusion ensemble, not a neural architecture and not a calibrated probability model. Its value is complementary ranking information, assessed through holdout rank correlation, top-K overlap, ablation and seed sensitivity.

## 4. Why supervised trees and boosting are not the primary approach

Logistic regression, decision trees, random forests, Extra Trees, XGBoost and LightGBM require a target for meaningful supervised evaluation. No independently adjudicated positive and negative procurement-integrity outcomes are available. A proxy flag created from review rules is not a substitute for ground truth: training a model to reproduce those rules would primarily measure agreement with the rule generator. The repository retains an optional weak-label teaching experiment, disabled by default and explicitly reported as proxy-rule agreement only.

Cost-sensitive tuning, ROC-AUC, PR-AUC and calibration are therefore not presented as real fraud-performance measures. They become appropriate only after lawful, independent outcome labels have been assembled.

## 5. Why clustering is not central

K-Means partitions observations around centroids and requires a chosen cluster count; cluster membership is not itself an anomaly ranking. DBSCAN is sensitive to distance scale and density parameters, particularly in mixed high-dimensional feature spaces where density can become hard to interpret. LOF and GMM supply more directly interpretable local- and global-density challenger rankings for the current scope. A clustering extension would need a demonstrated operational use, not inclusion for checklist coverage.

## 6. Why deep learning is excluded

- **MLP:** no independent supervised target is available; an MLP trained on generated proxy labels would reproduce the proxy construction rather than establish predictive validity.
- **CNN:** no image data are part of the procurement record task.
- **RNN/LSTM/GRU/TFT:** the audited FY extract has one release per OCID, so it does not contain a reliable within-process lifecycle sequence. Monthly volume forecasting is a distinct objective and is not required for process-level anomaly prioritisation.
- **Transformer:** current tender text is short and sparse, with no labelled text task or evidence that a language model would improve the ranking over TF-IDF/SVD. Transformer embeddings add complexity, compute and governance burden without a demonstrated benefit.
- **GCN/neural hypergraph:** the implementation uses classical graph statistics. It does not claim to train a GCN or a neural hypergraph model. Such a model would require sufficient stable identity linkage, careful temporal graph splits and independent validation.

## 7. Evaluation choices

The main split is chronological. Training-only preprocessing and score references reduce train/test and temporal leakage. The holdout is evaluated using:

- Spearman rank correlation between TGEE and challenger rankings.
- Top-K overlap at a review-capacity cutoff.
- Branch ablation to measure ranking sensitivity to each branch.
- Seed sensitivity to test reproducibility.
- GMM BIC/convergence diagnostics.
- Data-quality and coverage reporting.

These are ranking diagnostics, not accuracy measures against misconduct outcomes. Human review outcomes should be collected independently and lawfully before introducing supervised validation, Precision@K/Recall@K against true relevance, calibration or cost-sensitive model selection.

## 8. Course mapping

The mapping to DSA 8401 topics and the AML_Book-Data practicals is documented in docs/REFERENCE_REPO_ALIGNMENT.md.
