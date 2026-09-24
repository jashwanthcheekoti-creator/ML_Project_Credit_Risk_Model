import streamlit as st
from prediction_helper import INPUT_RANGES, LOAN_TENURE_RANGES, predict


st.set_page_config(page_title="Credit Risk Modelling", page_icon="📊")
st.title("Credit Risk Modelling")
st.caption("Explore borrower profiles and see how the model's estimated risk changes.")


def clear_previous_result():
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("prediction_error", None)


def handle_loan_purpose_change():
    clear_previous_result()
    purpose = st.session_state.get("loan_purpose", "Education")
    low, high = LOAN_TENURE_RANGES[purpose]
    tenure = st.session_state.get("loan_tenure_months", low)
    st.session_state["loan_tenure_months"] = min(max(tenure, low), high)


def range_help(name, purpose=None):
    if name == "loan_tenure_months":
        low, high = LOAN_TENURE_RANGES[purpose or "Education"]
    else:
        low, high = INPUT_RANGES[name]
    if name in {"income", "loan_amount"}:
        return f"App input range: ₹{low:,} to ₹{high:,}. This is a scenario limit, not a lender rule."
    if name in {"age", "num_open_accounts"}:
        return f"App input range: {low:,} to {high:,}. This is a scenario limit, not a lender rule."
    return f"App input range: {low:g} to {high:g}. This is a scenario limit, not a lender rule."


selected_loan_purpose = st.session_state.get("loan_purpose", "Education")
st.session_state.setdefault(
    "loan_tenure_months", LOAN_TENURE_RANGES[selected_loan_purpose][0]
)

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input(
        "Age",
        min_value=INPUT_RANGES["age"][0],
        max_value=INPUT_RANGES["age"][1],
        step=1,
        value=28,
        help=range_help("age"),
        on_change=clear_previous_result,
    )

with row1[1]:
    income = st.number_input(
        "Annual Income (₹)",
        min_value=INPUT_RANGES["income"][0],
        max_value=INPUT_RANGES["income"][1],
        step=1,
        value=1_200_000,
        help=range_help("income"),
        on_change=clear_previous_result,
    )

with row1[2]:
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=INPUT_RANGES["loan_amount"][0],
        max_value=INPUT_RANGES["loan_amount"][1],
        step=1,
        value=2_560_000,
        help=range_help("loan_amount"),
        on_change=clear_previous_result,
    )

with row2[0]:
    st.text("Loan to Income Ratio:")
    if income > 0:
        st.text(f"{round(loan_amount / income, 2):.2f}")
    else:
        st.text("Income must be greater than 0")

with row2[1]:
    loan_tenure_months = st.number_input(
        "Loan Tenure (months)",
        min_value=LOAN_TENURE_RANGES[selected_loan_purpose][0],
        max_value=LOAN_TENURE_RANGES[selected_loan_purpose][1],
        step=1,
        key="loan_tenure_months",
        help=range_help("loan_tenure_months", selected_loan_purpose),
        on_change=clear_previous_result,
    )

with row2[2]:
    avg_dpd_per_delinquency = st.number_input(
        "Avg DPD",
        min_value=INPUT_RANGES["avg_dpd_per_delinquency"][0],
        max_value=INPUT_RANGES["avg_dpd_per_delinquency"][1],
        step=0.1,
        value=3.3,
        format="%.1f",
        help=range_help("avg_dpd_per_delinquency"),
        on_change=clear_previous_result,
    )

with row3[0]:
    delinquency_ratio = st.number_input(
        "Delinquency Ratio",
        min_value=INPUT_RANGES["delinquency_ratio"][0],
        max_value=INPUT_RANGES["delinquency_ratio"][1],
        step=0.1,
        value=30.0,
        format="%.1f",
        help=range_help("delinquency_ratio"),
        on_change=clear_previous_result,
    )

with row3[1]:
    credit_utilization_ratio = st.number_input(
        "Credit Utilization Ratio",
        min_value=INPUT_RANGES["credit_utilization_ratio"][0],
        max_value=INPUT_RANGES["credit_utilization_ratio"][1],
        step=1.0,
        value=30.0,
        help=range_help("credit_utilization_ratio"),
        on_change=clear_previous_result,
    )

with row3[2]:
    num_open_accounts = st.number_input(
        "Open Loan Accounts",
        min_value=INPUT_RANGES["num_open_accounts"][0],
        max_value=INPUT_RANGES["num_open_accounts"][1],
        step=1,
        value=2,
        help=range_help("num_open_accounts"),
        on_change=clear_previous_result,
    )

with row4[0]:
    residence_type = st.selectbox(
        "Residence Type",
        ["Owned", "Rented", "Mortgage"],
        on_change=clear_previous_result,
    )

with row4[1]:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Education", "Home", "Auto", "Personal"],
        key="loan_purpose",
        on_change=handle_loan_purpose_change,
    )

with row4[2]:
    loan_type = st.selectbox(
        "Loan Type",
        ["Unsecured", "Secured"],
        on_change=clear_previous_result,
    )

if st.button("Calculate Risk"):
    # Clear old output first, including when the new values fail validation.
    clear_previous_result()
    try:
        st.session_state["prediction_result"] = predict(
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
    except ValueError as error:
        st.session_state["prediction_error"] = str(error)
    except Exception as error:
        st.session_state["prediction_error"] = (
            "Prediction failed. Check that model_data.joblib is present and "
            "the installed package versions match the model. "
            f"Details: {error}"
        )

if "prediction_error" in st.session_state:
    st.error(st.session_state["prediction_error"])

if "prediction_result" in st.session_state:
    probability, credit_score, rating = st.session_state["prediction_result"]
    st.write(f"Default Probability: {probability:.2%}")
    st.write(f"Credit Risk Score: {credit_score}")
    st.write(f"Rating: {rating}")
