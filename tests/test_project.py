import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "smartcare_ai_dataset_1000.csv"
DICTIONARY_PATH = PROJECT_ROOT / "smartcare_ai_dataset_data_dictionary.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "smartcare_option_c_best_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "smartcare_option_c_metadata.json"
APP_PATH = PROJECT_ROOT / "app.py"


DEMO_PATIENTS = [
    (
        {
            "age": 18,
            "gender": "Female",
            "previous_admissions": 0,
            "systolic_bp": 100,
            "diastolic_bp": 65,
            "blood_sugar_mg_dl": 75,
            "cholesterol_mg_dl": 120,
            "bmi": 18.0,
        },
        "Low",
    ),
    (
        {
            "age": 44,
            "gender": "Female",
            "previous_admissions": 1,
            "systolic_bp": 128,
            "diastolic_bp": 79,
            "blood_sugar_mg_dl": 114,
            "cholesterol_mg_dl": 204,
            "bmi": 25.6,
        },
        "Medium",
    ),
    (
        {
            "age": 85,
            "gender": "Male",
            "previous_admissions": 5,
            "systolic_bp": 170,
            "diastolic_bp": 105,
            "blood_sugar_mg_dl": 195,
            "cholesterol_mg_dl": 320,
            "bmi": 37.0,
        },
        "High",
    ),
]


def make_patient_record(values: dict) -> pd.DataFrame:
    """Reproduce the feature construction used by the Streamlit dashboard."""
    row = dict(values)
    row["pulse_pressure"] = row["systolic_bp"] - row["diastolic_bp"]
    row["mean_arterial_pressure"] = (
        row["systolic_bp"] + 2 * row["diastolic_bp"]
    ) / 3
    row["has_previous_admission"] = int(row["previous_admissions"] > 0)
    return pd.DataFrame([row])


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


def test_required_project_artifacts_exist():
    required = [DATA_PATH, DICTIONARY_PATH, MODEL_PATH, METADATA_PATH, APP_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"Missing required project files: {missing}"


def test_dataset_contract():
    data = pd.read_csv(DATA_PATH)

    assert data.shape == (1000, 33)
    assert "disease_risk_level" in data.columns
    assert set(data["disease_risk_level"].dropna().unique()) == {
        "Low",
        "Medium",
        "High",
    }


def test_saved_pipeline_contract(pipeline, metadata):
    assert metadata["algorithm"] == "Logistic Regression"
    assert metadata["feature_set"] == "Clinical only"
    assert metadata["class_names"] == ["Low", "Medium", "High"]
    assert list(pipeline.classes_) == [0, 1, 2]
    assert metadata["training_rows"] + metadata["test_rows"] == 1000


@pytest.mark.parametrize(("values", "expected_label"), DEMO_PATIENTS)
def test_dashboard_demo_profiles(pipeline, metadata, values, expected_label):
    patient = make_patient_record(values)[metadata["features"]]
    predicted_class = int(pipeline.predict(patient)[0])
    probabilities = pipeline.predict_proba(patient)[0]

    assert metadata["class_names"][predicted_class] == expected_label
    assert probabilities.shape == (3,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.isclose(probabilities.sum(), 1.0)


def test_dashboard_source_is_valid_and_keeps_safety_controls():
    source = APP_PATH.read_text(encoding="utf-8")

    ast.parse(source)
    assert "pipeline.predict_proba" in source
    assert "local_contributions" in source
    assert "st.warning" in source
    assert "Educational decision-support demonstration only" in source
    assert "replace professional clinical assessment" in source
