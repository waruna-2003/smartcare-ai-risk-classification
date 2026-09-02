"""SmartCare Option C educational disease-risk prototype.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "smartcare_option_c_best_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "smartcare_option_c_metadata.json"

RISK_COLOURS = {
    "Low": ("#15803d", "#dcfce7"),
    "Medium": ("#a16207", "#fef9c3"),
    "High": ("#b91c1c", "#fee2e2"),
}

DEMO_PATIENTS = {
    "low": {
        "age": 18,
        "gender": "Female",
        "previous_admissions": 0,
        "systolic_bp": 100,
        "diastolic_bp": 65,
        "blood_sugar": 75,
        "cholesterol": 120,
        "bmi": 18.0,
    },
    "medium": {
        "age": 44,
        "gender": "Female",
        "previous_admissions": 1,
        "systolic_bp": 128,
        "diastolic_bp": 79,
        "blood_sugar": 114,
        "cholesterol": 204,
        "bmi": 25.6,
    },
    "high": {
        "age": 85,
        "gender": "Male",
        "previous_admissions": 5,
        "systolic_bp": 170,
        "diastolic_bp": 105,
        "blood_sugar": 195,
        "cholesterol": 320,
        "bmi": 37.0,
    },
}


@st.cache_resource
def load_artifacts():
    """Load the fitted preprocessing/model pipeline and its metadata once."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "The trained model or metadata file is missing from the models folder."
        )
    pipeline = joblib.load(MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return pipeline, metadata


def make_patient_record(
    age: int,
    gender: str,
    previous_admissions: int,
    systolic_bp: int,
    diastolic_bp: int,
    blood_sugar: int,
    cholesterol: int,
    bmi: float,
) -> pd.DataFrame:
    """Create the exact feature row expected by the saved pipeline."""
    return pd.DataFrame(
        [
            {
                "age": age,
                "previous_admissions": previous_admissions,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "blood_sugar_mg_dl": blood_sugar,
                "cholesterol_mg_dl": cholesterol,
                "bmi": bmi,
                "pulse_pressure": systolic_bp - diastolic_bp,
                "mean_arterial_pressure": (systolic_bp + 2 * diastolic_bp) / 3,
                "has_previous_admission": int(previous_admissions > 0),
                "gender": gender,
            }
        ]
    )


def local_contributions(pipeline, patient: pd.DataFrame, predicted_class: int) -> pd.DataFrame:
    """Return the largest linear score contributions for the predicted class."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(patient)[0]
    class_position = list(model.classes_).index(predicted_class)
    names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in preprocessor.get_feature_names_out()
    ]
    contributions = transformed * model.coef_[class_position]
    result = pd.DataFrame(
        {
            "Feature": names,
            "Contribution": contributions,
        }
    )
    result["Effect"] = result["Contribution"].apply(
        lambda value: "Supports prediction" if value >= 0 else "Opposes prediction"
    )
    result["Magnitude"] = result["Contribution"].abs()
    return (
        result.sort_values("Magnitude", ascending=False)
        .head(5)
        .drop(columns="Magnitude")
        .reset_index(drop=True)
    )


st.set_page_config(
    page_title="SmartCare Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #0f4c5c, #167d8d);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2.15rem;}
    .hero p {margin: .45rem 0 0; opacity: .92;}
    .risk-card {
        padding: 1.25rem; border-radius: 14px; text-align: center;
        border: 1px solid rgba(0,0,0,.08); margin: .5rem 0 1rem;
    }
    .risk-label {font-size: 2rem; font-weight: 750; margin: .2rem 0;}
    .small-note {font-size: .88rem; color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>SmartCare Disease Risk Predictor</h1>
      <p>Explainable Low, Medium, and High risk classification from initial clinical information.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    pipeline, metadata = load_artifacts()
except Exception as error:
    st.error(f"The prototype could not load its model: {error}")
    st.stop()

demo_name = str(st.query_params.get("demo", "")).lower()
defaults = DEMO_PATIENTS.get(demo_name, DEMO_PATIENTS["medium"])

with st.sidebar:
    st.header("About this prototype")
    st.write(f"**Model:** {metadata['algorithm']}")
    st.write(f"**Feature set:** {metadata['feature_set']}")
    st.write(f"**Test accuracy:** {metadata['test_metrics']['Accuracy']:.1%}")
    st.info(
        "The model was trained on a synthetic coursework dataset. "
        "Its test performance does not establish clinical validity."
    )

st.subheader("1. Enter patient information")
st.caption("The input limits follow the ranges in the supplied SmartCare dataset.")

with st.form("patient_form"):
    first, second, third = st.columns(3)
    with first:
        age = st.number_input("Age (years)", 1, 90, defaults["age"], 1)
        gender = st.selectbox(
            "Gender", ["Female", "Male"],
            index=["Female", "Male"].index(defaults["gender"]),
        )
        previous_admissions = st.number_input(
            "Previous admissions", 0, 5, defaults["previous_admissions"], 1
        )
    with second:
        systolic_bp = st.number_input(
            "Systolic blood pressure (mmHg)", 85, 178, defaults["systolic_bp"], 1
        )
        diastolic_bp = st.number_input(
            "Diastolic blood pressure (mmHg)", 50, 111, defaults["diastolic_bp"], 1
        )
        bmi = st.number_input("BMI", 14.0, 38.8, defaults["bmi"], 0.1)
    with third:
        blood_sugar = st.number_input(
            "Blood sugar (mg/dL)", 65, 201, defaults["blood_sugar"], 1
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 100, 330, defaults["cholesterol"], 1
        )
        st.metric("Calculated pulse pressure", f"{systolic_bp - diastolic_bp} mmHg")

    submitted = st.form_submit_button(
        "Predict disease risk", type="primary", width="stretch"
    )

submitted = submitted or demo_name in DEMO_PATIENTS

if submitted:
    if systolic_bp <= diastolic_bp:
        st.error("Systolic blood pressure must be greater than diastolic pressure.")
        st.stop()

    patient = make_patient_record(
        age,
        gender,
        previous_admissions,
        systolic_bp,
        diastolic_bp,
        blood_sugar,
        cholesterol,
        bmi,
    )

    predicted_class = int(pipeline.predict(patient)[0])
    probabilities = pipeline.predict_proba(patient)[0]
    class_names = metadata["class_names"]
    predicted_label = class_names[predicted_class]
    probability_by_label = {
        class_names[int(class_value)]: float(probability)
        for class_value, probability in zip(pipeline.classes_, probabilities)
    }
    colour, background = RISK_COLOURS[predicted_label]

    st.divider()
    st.subheader("2. Prediction result")
    st.markdown(
        f"""
        <div class="risk-card" style="background:{background}; color:{colour};">
          <div>Predicted disease-risk level</div>
          <div class="risk-label">{predicted_label}</div>
          <div>Model confidence: {probability_by_label[predicted_label]:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_columns = st.columns(3)
    for column, label in zip(probability_columns, class_names):
        column.metric(f"{label} probability", f"{probability_by_label[label]:.1%}")

    probability_chart = pd.DataFrame(
        {"Probability": [probability_by_label[label] for label in class_names]},
        index=class_names,
    )
    st.bar_chart(probability_chart, y="Probability", horizontal=True)

    st.subheader("3. Why did the model make this prediction?")
    st.write(
        "The table shows the five largest contributions to the predicted class score. "
        "Positive values support the prediction; negative values oppose it."
    )
    contribution_table = local_contributions(
        pipeline, patient, predicted_class
    ).copy()
    contribution_table["Contribution"] = contribution_table["Contribution"].round(3)
    st.dataframe(contribution_table, hide_index=True, width="stretch")

    with st.expander("View the processed patient record"):
        display_record = patient.rename(
            columns=lambda value: value.replace("_", " ").title()
        )
        st.dataframe(display_record, hide_index=True, width="stretch")

    st.warning(
        "Educational decision-support demonstration only. This prediction must not "
        "replace professional clinical assessment, diagnosis, or treatment decisions."
    )
else:
    st.info("Complete the form and select **Predict disease risk** to see a result.")

st.divider()
st.markdown(
    '<p class="small-note">CCS3440 Artificial Intelligence Coursework · Option C · '
    "Synthetic SmartCare data · No patient information is stored by this app.</p>",
    unsafe_allow_html=True,
)
