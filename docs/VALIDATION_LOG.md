# Validation log

The rebuilt notebooks were executed end-to-end in the provided Python 3.13.5 environment using the included **synthetic smoke-test fixture**.

- Input: 1,400 generated OCDS-style process releases.
- Latest snapshots retained: 1,400.
- Chronological partition: 1,120 training records and 280 holdout records.
- All five score branches activated on the fixture (including text and bidder co-participation).
- The API notebook passed `GET /health`, `GET /model-info` and `POST /score` through FastAPI TestClient (`POST /score` returned HTTP 200).
- Notebook code cells and API smoke test completed without execution errors.

This validates code paths and integration only. It does **not** validate the model on official PPIP data, and no synthetic metric should be presented as an empirical procurement finding. The generated synthetic model bundle and score files are intentionally not included in the submission ZIP; run the pipeline on the official data to create your own artifacts.
