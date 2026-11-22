# Kenya PPIP Procurement Screening

**DSA 8401 – Applied Machine Learning Capstone**  
**Student:** Dean Maingi Munywoki  
**Programme:** MSc Data Science & Analytics, Strathmore University

## What the project does

This project uses Kenya's Public Procurement Information Portal (PPIP) OCDS data to rank procurement processes whose tender details look unusual compared with earlier records. The ranking is intended to help a reviewer decide what to examine first. It does not determine whether a procurement process is corrupt or unlawful.

The work follows the practical sequence used in the Applied Machine Learning course: inspect the data, clean it, check for leakage, build a preprocessing and modelling pipeline, evaluate the result, and serve the model through an API.

## Notebooks

Run the notebooks in this order:

| Notebook | Work |
|---|---|
| `01_data_audit.ipynb` | Downloads the PPIP OCDS file, reads the compiled processes and checks data quality. |
| `02_data_preparation.ipynb` | Cleans tender fields and creates the features used by the model. |
| `03_model_training.ipynb` | Fits Isolation Forest, checks ranking stability and records the experiment in MLflow. |
| `04_evaluation_and_monitoring.ipynb` | Summarises the holdout ranking and compares training and later-period feature distributions. |
| `05_fastapi_service.ipynb` | Loads the saved model, tests the API and starts the FastAPI service. |

The notebooks contain the project code. There is no separate notebook source library and no generated Python module tree to maintain.

## How the course practicals are used

| Course practical | Applied in this project |
|---|---|
| Lab 2 – messy data and leakage-safe features | Missingness and duplicate checks, date and value cleaning, feature construction and exclusion of post-award fields. |
| Fraud MLOps | A fitted scikit-learn pipeline, MLflow experiment logging, FastAPI scoring and Docker deployment. |
| MLflow Simple | Recording model settings and diagnostic results for later review. |
| FastAPI and drift practical | API checks, batch scoring and a comparison of training and later-period feature distributions. |

The class fraud practical uses a labelled mobile-money dataset and evaluates classifiers with F1. The public PPIP snapshot does not provide verified integrity labels, so this project uses anomaly ranking instead. That is a change required by the data, not a claim that an unsupervised model is a substitute for a labelled fraud classifier.

## Data

The source is the [PPIP OCDS portal](https://tenders.go.ke/ocds). The [OCP Data Registry profile](https://data.open-contracting.org/en/publication/147) provides an all-time compressed JSONL download. The current profile covers July 2018 to October 2026 and reports 262,961 tender objects but no tenderer objects. The profile also identifies empty award and contract arrays, buyer and supplier identifier issues, and extreme values in contract and award amounts.

Notebook 1 keeps the original `.jsonl.gz` file in `data/raw/` and writes a process-level CSV to `data/processed/`. The archive is not included in this project folder; it is downloaded when the first notebook runs. The source profile does not show a dataset licence, so check the current terms before redistributing the downloaded data.

The data have no verified fraud or corruption outcome label. Unlabelled processes are not treated as legitimate examples. For that reason, the real-data model is unsupervised and reports relative anomaly percentiles, not fraud probabilities.

## Set up

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the full data and modelling workflow:

```bash
bash run_pipeline.sh
```

The first notebook needs internet access to download the source. If you already have the official compressed JSONL file, place it at `data/raw/ppip_ocds_all_time.jsonl.gz` before running the notebooks.

You can also open Jupyter and run notebooks 1–4 one by one:

```bash
jupyter notebook
```

## MLflow

Notebook 3 records the model parameters and diagnostic metrics locally. To view the experiment:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`.

## Run the API

After the model has been trained:

```bash
bash run_api.sh
```

Open `http://127.0.0.1:8000/docs`.

The API accepts the same prepared feature columns used by the model. It provides `/health`, `/model-info`, `/score` and `/score/batch`. For a first test, run the API checks in notebook 5 before starting the server.

## Docker

Build and run the API after `run_pipeline.sh` has created `artifacts/ppip_anomaly_model.joblib`:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000/docs`.

## Project outputs

- `data/raw/` – downloaded official PPIP JSONL archive.
- `data/processed/` – process-level and feature-level CSV files.
- `artifacts/` – trained scikit-learn pipeline.
- `reports/` – audit tables, evaluation tables, review shortlist and plots.
- `mlflow.db` – local experiment tracking database.

These files are generated when the notebooks run and are not required in the submitted source folder.

## How to read the scores

The model is fitted on the earlier 70% of processes and checked on the later 30%, ordered by tender start date. Since the source is a compiled snapshot rather than the full original publication history, this is a chronological distribution check, not a strict reconstruction of what was publicly known at each historical moment.

The `high`, `review` and `routine` bands are based on the 98th and 90th percentiles of the training anomaly scores. They are only a way to organise manual review. No precision, recall, F1-score, ROC-AUC or AUPRC is reported because verified outcome labels are not available.

## References

- Lee Bundi, [AML_Book-Data course repository](https://github.com/NUELBUNDI/AML_Book-Data), including the messy-data and leakage-safe feature practical, the MLflow examples, the FastAPI practical and drift-monitoring exercise.
- Open Contracting Partnership, [Kenya: Public Procurement Regulatory Authority (PPRA) dataset profile](https://data.open-contracting.org/en/publication/147).
- Public Procurement Information Portal, [OCDS data](https://tenders.go.ke/ocds).

See `reports/FINAL_REPORT.md` for the project write-up and `AI_USE.md` for the use of AI tools during preparation.
