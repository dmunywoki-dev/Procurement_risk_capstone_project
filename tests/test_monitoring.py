import numpy as np
import pandas as pd

from scripts.monitor_drift import compare_frames, _psi, _categorical_psi


def test_psi_zero_for_same_distribution():
    values = pd.Series(np.arange(100, dtype=float))
    assert _psi(values, values) == 0.0


def test_psi_detects_shifted_distribution():
    reference = pd.Series(np.linspace(0, 1, 200))
    current = pd.Series(np.linspace(3, 4, 200))
    assert _psi(reference, current) > 0.20


def test_categorical_psi_detects_composition_change():
    reference = pd.Series(["open"] * 90 + ["direct"] * 10)
    current = pd.Series(["open"] * 10 + ["direct"] * 90)
    assert _categorical_psi(reference, current) > 0.20


def test_compare_frames_returns_drift_table():
    reference = pd.DataFrame({
        "amount": np.linspace(0, 1, 100),
        "method": ["open"] * 80 + ["direct"] * 20,
    })
    current = pd.DataFrame({
        "amount": np.linspace(3, 4, 100),
        "method": ["direct"] * 80 + ["open"] * 20,
    })
    result = compare_frames(reference, current)
    assert set(result["feature"]) == {"amount", "method"}
    assert result["drift_flag"].all()
