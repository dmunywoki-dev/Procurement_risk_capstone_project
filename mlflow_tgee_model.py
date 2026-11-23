"""MLflow PyFunc wrapper around the prepared-feature TGEE scoring contract."""
from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
import mlflow.pyfunc


class TgeePyfuncModel(mlflow.pyfunc.PythonModel):
    """Accept a DataFrame with one prepared record per row."""

    def load_context(self, context: Any) -> None:
        self.bundle = joblib.load(context.artifacts["model_bundle"])

    def predict(self, context: Any, model_input: pd.DataFrame,
                params: dict[str, Any] | None = None) -> pd.DataFrame:
        from api_service import score_prepared_record

        results = []
        for _, row in model_input.iterrows():
            payload = {}
            for key in ("record_id", "ocid", "context", "tabular",
                        "temporal", "graph", "text", "hypergraph"):
                value = row.get(key)
                if isinstance(value, str) and key in {
                    "context", "tabular", "temporal", "graph", "hypergraph"
                }:
                    value = json.loads(value)
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    payload[key] = value
            payload.setdefault("context", {})
            payload.setdefault("text", "")
            results.append(score_prepared_record(self.bundle, payload))
        return pd.DataFrame(results)
