"""Input preparation and prediction for the credit risk demo app."""

from functools import lru_cache
import math
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model_data.joblib"

# Broad, bounded scenario limits for the app's portfolio demonstration.
# They are intentionally wider than the source data for some features; they
# are not lender eligibility rules and do not guarantee reliable extrapolation.
INPUT_RANGES = {
    "age": (18, 80),
    "income": (100_000, 20_000_000),
    "loan_amount": (10_000, 80_000_000),
    "avg_dpd_per_delinquency": (0.0, 60.0),
    "delinquency_ratio": (0.0, 100.0),
    "credit_utilization_ratio": (0.0, 100.0),
    "num_open_accounts": (0, 4),
    "loan_to_income": (0.0, 10.0),
}

# Repayment-tenure scenario bounds vary by loan purpose. Education excludes
# the course period and any repayment moratorium/grace period.
LOAN_TENURE_RANGES = {
    "Home": (6, 180),
    "Auto": (6, 84),
    "Education": (6, 180),
    "Personal": (6, 120),
}

EXPECTED_FEATURES = [
    "age",
    "loan_tenure_months",
    "number_of_open_accounts",
    "credit_utilization_ratio",
    "loan_to_income",
    "deliquency_ratio",  # Keep the notebook/artifact spelling.
    "avg_dpd_per_delinquency",
    "residence_type_Owned",
    "residence_type_Rented",
    "loan_purpose_Education",
    "loan_purpose_Home",
    "loan_purpose_Personal",
    "loan_type_Unsecured",
]

EXPECTED_SCALE_COLUMNS = [
    "age",
    "number_of_dependants",
    "years_at_current_address",
    "sanction_amount",
    "processing_fee",
    "gst",
    "net_disbursement",
    "loan_tenure_months",
    "principal_outstanding",
    "bank_balance_at_application",
    "number_of_open_accounts",
    "number_of_closed_accounts",
    "enquiry_count",
    "credit_utilization_ratio",
    "loan_to_income",
    "deliquency_ratio",
    "avg_dpd_per_delinquency",
]


@lru_cache(maxsize=1)
def _load_model_data():
    """Load the local artifact once and fail with an actionable message."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Keep artifacts/model_data.joblib beside this app."
        )

    try:
        data = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not load model_data.joblib. Check that the artifact is valid "
            "and that the installed scikit-learn version matches its training version."
        ) from exc

    required_keys = {"model", "scaler", "features", "cols_to_scale"}
    if not isinstance(data, dict) or not required_keys.issubset(data):
        raise RuntimeError(
            "The model artifact must contain model, scaler, features, and cols_to_scale."
        )

    features = list(data["features"])
    scale_columns = list(data["cols_to_scale"])
    if features != EXPECTED_FEATURES:
        raise RuntimeError(
            "The artifact feature order does not match the notebook schema. "
            f"Expected {EXPECTED_FEATURES}; found {features}."
        )
    if scale_columns != EXPECTED_SCALE_COLUMNS:
        raise RuntimeError(
            "The artifact scaler columns do not match the notebook schema. "
            f"Expected {EXPECTED_SCALE_COLUMNS}; found {scale_columns}."
        )
    if getattr(data["model"], "n_features_in_", None) != len(features):
        raise RuntimeError("The saved model feature count does not match its feature list.")
    if getattr(data["scaler"], "n_features_in_", None) != len(scale_columns):
        raise RuntimeError("The saved scaler column count does not match its column list.")
    if not hasattr(data["model"], "predict_proba"):
        raise RuntimeError("The saved model does not provide predict_proba().")
    if 1 not in list(data["model"].classes_):
        raise RuntimeError("The saved model has no class 1 for default.")

    return data


def _as_finite_number(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _validate_integer(name, value, minimum, maximum):
    number = _as_finite_number(name, value)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number.")
    number = int(number)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:,} and {maximum:,}.")
    return number


def _validate_decimal(name, value, minimum, maximum):
    number = _as_finite_number(name, value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
    # The notebook rounds these engineered features to one decimal place.
    return round(number, 1)


def _validate_inputs(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    avg_dpd_per_delinquency,
    delinquency_ratio,
    credit_utilization_ratio,
    num_open_accounts,
    residence_type,
    loan_purpose,
    loan_type,
):
    age = _validate_integer("Age", age, *INPUT_RANGES["age"])
    income = _validate_integer("Income", income, *INPUT_RANGES["income"])
    loan_amount = _validate_integer(
        "Loan Amount", loan_amount, *INPUT_RANGES["loan_amount"]
    )
    avg_dpd_per_delinquency = _validate_decimal(
        "Avg DPD",
        avg_dpd_per_delinquency,
        *INPUT_RANGES["avg_dpd_per_delinquency"],
    )
    delinquency_ratio = _validate_decimal(
        "Delinquency Ratio",
        delinquency_ratio,
        *INPUT_RANGES["delinquency_ratio"],
    )
    credit_utilization_ratio = _validate_integer(
        "Credit Utilization Ratio",
        credit_utilization_ratio,
        *INPUT_RANGES["credit_utilization_ratio"],
    )
    num_open_accounts = _validate_integer(
        "Open Loan Accounts", num_open_accounts, *INPUT_RANGES["num_open_accounts"]
    )

    valid_residence_types = {"Owned", "Rented", "Mortgage"}
    valid_loan_purposes = {"Education", "Home", "Auto", "Personal"}
    valid_loan_types = {"Unsecured", "Secured"}
    if residence_type not in valid_residence_types:
        raise ValueError("Choose a valid Residence Type.")
    if loan_purpose not in valid_loan_purposes:
        raise ValueError("Choose a valid Loan Purpose.")
    if loan_type not in valid_loan_types:
        raise ValueError("Choose a valid Loan Type.")

    tenure_min, tenure_max = LOAN_TENURE_RANGES[loan_purpose]
    loan_tenure_months = _validate_integer(
        f"{loan_purpose} Loan Tenure (months)",
        loan_tenure_months,
        tenure_min,
        tenure_max,
    )

    # The notebook engineers this feature as round(loan_amount / income, 2).
    loan_to_income = round(loan_amount / income, 2)
    low, high = INPUT_RANGES["loan_to_income"]
    if not low <= loan_to_income <= high:
        raise ValueError(
            f"Loan to Income Ratio must be between {low:.2f} and {high:.2f} "
            "after rounding to two decimal places."
        )

    return {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_tenure_months": loan_tenure_months,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
        "delinquency_ratio": delinquency_ratio,
        "credit_utilization_ratio": credit_utilization_ratio,
        "num_open_accounts": num_open_accounts,
        "residence_type": residence_type,
        "loan_purpose": loan_purpose,
        "loan_type": loan_type,
        "loan_to_income": loan_to_income,
    }


def prepare_input(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    avg_dpd_per_delinquency,
    delinquency_ratio,
    credit_utilization_ratio,
    num_open_accounts,
    residence_type,
    loan_purpose,
    loan_type,
):
    values = _validate_inputs(
        age,
        income,
        loan_amount,
        loan_tenure_months,
        avg_dpd_per_delinquency,
        delinquency_ratio,
        credit_utilization_ratio,
        num_open_accounts,
        residence_type,
        loan_purpose,
        loan_type,
    )

    # These fields are required by the saved scaler but are not among the
    # selected model features. The notebook did not collect them in the app.
    input_data = {
        "age": values["age"],
        "number_of_dependants": 1,
        "years_at_current_address": 1,
        "zipcode": 1,
        "sanction_amount": 1,
        "processing_fee": 1,
        "gst": 1,
        "net_disbursement": 1,
        "loan_tenure_months": values["loan_tenure_months"],
        "principal_outstanding": 1,
        "bank_balance_at_application": 1,
        "number_of_open_accounts": values["num_open_accounts"],
        "number_of_closed_accounts": 1,
        "enquiry_count": 1,
        "credit_utilization_ratio": values["credit_utilization_ratio"],
        "loan_to_income": values["loan_to_income"],
        "deliquency_ratio": values["delinquency_ratio"],
        "avg_dpd_per_delinquency": values["avg_dpd_per_delinquency"],
        "residence_type_Owned": int(values["residence_type"] == "Owned"),
        "residence_type_Rented": int(values["residence_type"] == "Rented"),
        "loan_purpose_Education": int(values["loan_purpose"] == "Education"),
        "loan_purpose_Home": int(values["loan_purpose"] == "Home"),
        "loan_purpose_Personal": int(values["loan_purpose"] == "Personal"),
        "loan_type_Unsecured": int(values["loan_type"] == "Unsecured"),
    }

    data = _load_model_data()
    scale_columns = list(data["cols_to_scale"])
    row = pd.DataFrame([input_data])
    row[scale_columns] = data["scaler"].transform(row[scale_columns])
    # Use the feature order stored in the artifact; schema is checked at load.
    return row[list(data["features"])]


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    data = _load_model_data()
    model = data["model"]
    probabilities = model.predict_proba(input_df)
    default_class_index = list(model.classes_).index(1)
    default_probability = float(probabilities[0, default_class_index])

    score = int(base_score + (1.0 - default_probability) * scale_length)
    if 300 <= score < 500:
        rating = "Poor"
    elif 500 <= score < 650:
        rating = "Average"
    elif 650 <= score < 750:
        rating = "Good"
    else:
        rating = "Excellent"

    return default_probability, score, rating


def predict(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    avg_dpd_per_delinquency,
    delinquency_ratio,
    credit_utilization_ratio,
    num_open_accounts,
    residence_type,
    loan_purpose,
    loan_type,
):
    input_df = prepare_input(
        age=age,
        income=income,
        loan_amount=loan_amount,
        loan_tenure_months=loan_tenure_months,
        avg_dpd_per_delinquency=avg_dpd_per_delinquency,
        delinquency_ratio=delinquency_ratio,
        credit_utilization_ratio=credit_utilization_ratio,
        num_open_accounts=num_open_accounts,
        residence_type=residence_type,
        loan_purpose=loan_purpose,
        loan_type=loan_type,
    )
    return calculate_credit_score(input_df)
