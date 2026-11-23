import numpy as np
from fastapi.testclient import TestClient

from api_service import create_app


class FixedDetector:
    def __init__(self, value):
        self.value = value

    def score_samples(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def make_bundle():
    return {
        "project_name": "test bundle",
        "model_version": "test-1",
        "data_mode": "test_fixture",
        "score_semantics": "relative review-priority percentile; not a fraud probability",
        "tabular_model": FixedDetector(-0.9),
        "tabular_feature_cols": ["amount_feature"],
        "tabular_reference": np.array([0.1, 0.2, 0.3, 0.4]),
        "temporal_model": FixedDetector(-0.8),
        "temporal_feature_cols": ["duration_deviation"],
        "temporal_reference": np.array([0.1, 0.2, 0.3, 0.4]),
        "graph_model": FixedDetector(-0.7),
        "graph_feature_cols": ["prior_supplier_count"],
        "graph_reference": np.array([0.1, 0.2, 0.3, 0.4]),
        "weights": {"tabular": 0.4, "temporal": 0.3, "graph": 0.3},
        "review_threshold": 0.75,
        "high_threshold": 0.95,
        "text_active": False,
        "hypergraph_active": False,
    }


def valid_record():
    return {
        "record_id": "test-001",
        "ocid": "ocds-test-001",
        "context": {"procurement_method": "open"},
        "tabular": {"amount_feature": 0.0},
        "temporal": {"duration_deviation": 0.0},
        "graph": {"prior_supplier_count": 1},
        "text": "",
        "hypergraph": None,
    }


def test_health_and_model_info():
    client = TestClient(create_app(model_bundle=make_bundle()))
    assert client.get("/health").status_code == 200
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["model_version"] == "test-1"


def test_single_record_score_is_relative_priority_not_probability():
    client = TestClient(create_app(model_bundle=make_bundle()))
    response = client.post("/score", json=valid_record())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["tgee_priority_score"] <= 1.0
    assert body["review_band"] in {"routine", "review", "high_priority"}
    assert "not a fraud probability" in body["score_semantics"]


def test_missing_prepared_feature_is_rejected():
    client = TestClient(create_app(model_bundle=make_bundle()))
    payload = valid_record()
    payload["graph"] = {}
    response = client.post("/score", json=payload)
    assert response.status_code == 422
    assert "prior_supplier_count" in response.json()["detail"]


def test_batch_returns_one_result_per_record():
    client = TestClient(create_app(model_bundle=make_bundle()))
    response = client.post(
        "/score/batch",
        json={"records": [valid_record(), valid_record()]},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["results"]) == 2


def test_batch_rejects_empty_list():
    client = TestClient(create_app(model_bundle=make_bundle()))
    response = client.post("/score/batch", json={"records": []})
    assert response.status_code == 422
