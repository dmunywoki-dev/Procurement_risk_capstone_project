# DSA 8401 Capstone – Presentation Outline

## Slide 1 – Title
**Procurement Risk Screening Using Kenya PPIP OCDS Data**  
Dean Maingi Munywoki · MSc Data Science & Analytics · Strathmore University

## Slide 2 – The problem
- Public procurement records are extensive.
- Detailed review of every process takes time.
- The project ranks unusual tender profiles to help prioritise manual review.

## Slide 3 – Data source
- PPIP OCDS data from tenders.go.ke.
- All-time compiled-release JSONL snapshot.
- The registry profile reports no tenderer objects and known identifier and completeness issues.

## Slide 4 – Data preparation
- Read the compiled releases and create one row per OCID.
- Check duplicates, missing fields, dates and values.
- Standardise procurement method, category and county.
- Keep tender-stage fields separate from award and contract information.

## Slide 5 – Features
- Estimated value and tender duration.
- Procurement method and category.
- County and calendar features.
- Tender title and description using TF-IDF.

## Slide 6 – Model
- Scikit-learn preprocessing pipeline.
- Isolation Forest for unsupervised anomaly screening.
- Earlier 70% of processes for fitting; later 30% for a chronological check.
- MLflow stores experiment settings and diagnostics.

## Slide 7 – Evaluation
- Review anomaly-score distribution and top-ranked processes.
- Check ranking stability across random seeds.
- Compare feature distributions between the two time periods.
- No supervised metrics because verified outcome labels are unavailable.

## Slide 8 – API and deployment
- FastAPI: `/health`, `/model-info`, `/score`, `/score/batch`.
- Dockerfile and Docker Compose.
- The API uses the same fitted preprocessing pipeline as training.

## Slide 9 – Limitations
- Compiled snapshot is not a full historical release stream.
- No verified integrity outcome labels.
- No tenderer objects for bidder-network analysis.
- Anomaly does not mean corruption.

## Slide 10 – Conclusion
The project produces a repeatable ranking workflow for further procurement review. Validation against independently verified findings is needed before operational use.
