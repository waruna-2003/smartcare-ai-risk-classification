# SmartCare Hospital AI — Option C

Group coursework project for explainable multiclass disease-risk classification.

The project predicts the supplied `disease_risk_level` target with three classes:

- Low
- Medium
- High

The prediction point is the initial patient assessment. The project is an educational prototype trained on synthetic data and is not clinically validated.

## Project contents

| File | Purpose |
|---|---|
| `smartcare_dataset_analysis.ipynb` | Problem definition, dataset understanding, data-quality checks, leakage policy and exploratory analysis |
| `smartcare_model_development.ipynb` | Preprocessing, feature engineering, model comparison, tuning, evaluation and model saving |
| `smartcare_explainable_ai.ipynb` | Global, class-specific and local SHAP explanations |
| `app.py` | Streamlit disease-risk prediction dashboard |
| `models/` | Saved Logistic Regression pipeline and model metadata |
| `tests/` | Automated dataset, model and dashboard-contract tests |

## Environment setup

Open PowerShell in the project folder:

```powershell
cd "D:\AI Final"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run the notebooks

Open the folder in VS Code and select the `.venv` Python kernel. Run the notebooks in this order:

1. `smartcare_dataset_analysis.ipynb`
2. `smartcare_model_development.ipynb`
3. `smartcare_explainable_ai.ipynb`

The model-development notebook creates the files required by the explainability notebook and dashboard.

## Run the Streamlit dashboard

With the virtual environment activated:

```powershell
python -m streamlit run app.py
```

The browser normally opens at `http://localhost:8501`.

Reproducible demonstration profiles are available at:

- `http://localhost:8501/?demo=low`
- `http://localhost:8501/?demo=medium`
- `http://localhost:8501/?demo=high`

## Run automated tests

From the project folder:

```powershell
python -m pytest -q
```

The tests verify that:

- required project artifacts exist;
- the dataset and target classes match the project contract;
- the saved pipeline loads and returns valid probabilities;
- the three dashboard demonstration profiles produce the intended risk levels;
- `app.py` is valid Python and retains prediction and safety-warning behaviour.

## Saved model

- Algorithm: Logistic Regression
- Feature set: clinical-only
- Training/test split: 800/200 records
- Best cross-validated macro-F1: approximately 0.988
- Final test accuracy: 99.5%
- Final test macro-F1: approximately 0.992

These results reproduce the supplied synthetic labels. They do not establish diagnostic accuracy, patient benefit or deployment safety.

## Team workflow

Complete each task on a separate Git branch and submit it through a pull request. Before opening a pull request, run the relevant notebook and execute `python -m pytest -q`.
