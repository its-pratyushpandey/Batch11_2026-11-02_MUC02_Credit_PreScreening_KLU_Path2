import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Loan Pre-Screening",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "loan_approve_decline_pipeline.joblib"

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --blue-900: #082b63;
        --blue-700: #0759a8;
        --blue-500: #1687d9;
        --ink: #14243a;
        --muted: #5c6f86;
        --line: #d9e7f5;
    }

    .stApp {
        background: linear-gradient(
            135deg, #061b49 0%, #0759a8 48%, #46b7e8 100%
        );
        background-attachment: fixed;
        color: #ffffff;
    }

    [data-testid="stHeader"] {
        background: rgba(6, 27, 73, 0.18);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        width: 100%;
        max-width: 1180px;
        padding: 2rem clamp(0.8rem, 3vw, 2rem) 3rem;
    }

    /* Main headings and text */

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp label {
        color: #ffffff;
    }

    h1 {
        font-weight: 850 !important;
        letter-spacing: -0.035em;
        line-height: 1.12;
    }

    h2, h3, h4 {
        letter-spacing: -0.02em;
    }

    [data-testid="stCaptionContainer"] p {
        color: #e2f1ff !important;
        font-size: 0.94rem;
    }

    /* Form panel */

    [data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 22px;
        padding: clamp(1rem, 2.5vw, 1.8rem);
        box-shadow: 0 18px 48px rgba(2, 18, 48, 0.20);
    }

    [data-testid="stForm"] h1,
    [data-testid="stForm"] h2,
    [data-testid="stForm"] h3,
    [data-testid="stForm"] h4,
    [data-testid="stForm"] p,
    [data-testid="stForm"] label,
    [data-testid="stForm"] [data-testid="stMarkdownContainer"] {
        color: #14243a !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }

    /* Input fields: white background, dark visible values */

    [data-testid="stForm"] div[data-baseweb="input"] > div,
    [data-testid="stForm"] div[data-baseweb="select"] > div,
    [data-testid="stForm"] div[data-baseweb="textarea"] > div {
        background-color: #ffffff !important;
        border-color: #c4d6e9 !important;
        border-radius: 10px;
    }

    [data-testid="stForm"] input,
    [data-testid="stForm"] textarea,
    [data-testid="stForm"] [role="combobox"] {
        color: #14243a !important;
        -webkit-text-fill-color: #14243a !important;
        background-color: #ffffff !important;
        opacity: 1 !important;
    }

    [data-testid="stForm"] [data-testid="stNumberInput"] button {
        color: #14243a !important;
        background-color: #ffffff !important;
    }

    /* Dropdown options */

    [data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [role="listbox"] *,
    [role="option"] {
        color: #14243a !important;
    }

    [data-baseweb="popover"],
    [role="listbox"],
    [role="option"] {
        background-color: #ffffff !important;
    }

    [data-testid="stForm"] [data-testid="stRadio"] label,
    [data-testid="stForm"] [data-testid="stCheckbox"] label {
        color: #14243a !important;
    }

    /* Buttons */

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(100deg, #0759a8, #1687d9);
        color: #ffffff !important;
        border: 1px solid #1687d9;
        border-radius: 12px;
        min-height: 3.1rem;
        font-weight: 750;
        box-shadow: 0 7px 18px rgba(7, 89, 168, 0.2);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        color: #ffffff !important;
        border-color: #064a91;
        background: linear-gradient(100deg, #064a91, #0874c4);
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(7, 89, 168, 0.28);
    }

    /* Metric cards */

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 22px rgba(2, 18, 48, 0.12);
    }

    div[data-testid="stMetricLabel"] p {
        color: #52677f !important;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #082b63 !important;
        -webkit-text-fill-color: #082b63 !important;
        font-weight: 800;
    }

    /* Other alerts */

    [data-testid="stAlert"] {
        border-radius: 14px;
        border-width: 1px;
    }

    hr {
        border-color: rgba(255, 255, 255, 0.28);
        margin: 1.7rem 0;
    }

    /* Header card */

    .hero-card {
        background: linear-gradient(
            125deg, rgba(255,255,255,0.98), rgba(232,246,255,0.96)
        );
        border: 1px solid rgba(255,255,255,0.8);
        border-radius: 24px;
        padding: clamp(1.2rem, 3vw, 2.2rem);
        margin-bottom: 1.2rem;
        box-shadow: 0 20px 48px rgba(2,18,48,0.20);
    }

    .hero-card h1 {
        color: #0b2f66 !important;
        font-size: clamp(1.8rem, 4vw, 2.8rem);
    }

    .hero-eyebrow {
        color: #0875bd;
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        color: #52677f !important;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0.65rem 0 0;
        max-width: 760px;
    }

    .section-note {
        color: #e2f1ff !important;
        margin-top: -0.45rem;
        margin-bottom: 1rem;
    }

    /* Prediction result: explicit colors */

    .prediction-result {
        border-radius: 12px;
        padding: 18px 22px;
        margin: 12px 0 20px 0;
        font-size: 20px;
        font-weight: 700;
        border-left: 6px solid;
    }

    .prediction-approved {
        background-color: #e8f5e9 !important;
        color: #1b5e20 !important;
        border-left-color: #2e7d32 !important;
    }

    .prediction-declined {
        background-color: #fff3e0 !important;
        color: #8b1e1e !important;
        border-left-color: #d32f2f !important;
    }

    footer {
        visibility: hidden;
    }

    /* Responsive design */

    @media (max-width: 800px) {
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }

        [data-testid="stForm"] {
            border-radius: 18px;
        }

        .hero-card {
            border-radius: 18px;
        }

        div[data-testid="stMetric"] {
            padding: 0.85rem;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        .hero-card {
            padding: 1.1rem;
        }

        .hero-eyebrow {
            font-size: 0.68rem;
            letter-spacing: 0.08em;
        }

        .hero-subtitle {
            font-size: 0.92rem;
        }

        [data-testid="stForm"] {
            padding: 0.85rem;
        }

        .stFormSubmitButton > button {
            min-height: 2.9rem;
        }

        .prediction-result {
            padding: 14px;
            font-size: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">
            Credit decision support · ML demo
        </div>
        <h1 style="margin:0;">Loan Pre-Screening</h1>
        <p class="hero-subtitle">
            Explore how applicant and affordability details relate to
            a model's prediction of a historical loan decision.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Educational prototype only. This model predicts historical bank decisions, "
    "not actual repayment or default. Do not use its output as the sole basis "
    "for a real lending decision. Human review, independent validation, and "
    "appropriate policy/compliance checks are required."
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model(path_string: str):
    return joblib.load(path_string)


if not MODEL_PATH.exists():
    st.error("Trained model file not found.")
    st.markdown(
        "Run all cells in **Loan_Approve_Decline_Deployment_Model.ipynb** first. "
        "That notebook should create `loan_approve_decline_pipeline.joblib`. "
        "Place the `.joblib` file in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

# ---------------------------------------------------------
# APPLICANT DETAILS
# ---------------------------------------------------------

st.subheader("Applicant details")

st.markdown(
    '<p class="section-note">Enter the applicant information below, '
    'then run the pre-screening prediction.</p>',
    unsafe_allow_html=True,
)

with st.form("loan_application_form"):
    st.markdown("#### Personal and income profile")

    left, right = st.columns(2, gap="large")

    with left:
        age = st.number_input(
            "Age (years)", min_value=18, max_value=100, value=30, step=1
        )

        employment_type = st.selectbox(
            "Employment type",
            [
                "Salaried",
                "Self-Employed Professional",
                "Business Owner",
                "Gig / Contract",
            ],
        )

        years_in_current_job = st.number_input(
            "Years in current job/business",
            min_value=0.0, max_value=70.0, value=4.0, step=0.5
        )

        monthly_income = st.number_input(
            "Monthly income (₹)",
            min_value=0.0, max_value=100_000_000.0,
            value=65000.0, step=1000.0
        )

        existing_emi = st.number_input(
            "Existing monthly EMIs (₹)",
            min_value=0.0, max_value=100_000_000.0,
            value=8000.0, step=500.0
        )

        has_credit_score = st.radio(
            "Does the applicant have a credit score?",
            ["Yes", "No"],
            horizontal=True,
        )

        if has_credit_score == "Yes":
            credit_score = st.number_input(
                "Credit score (300–900)",
                min_value=300, max_value=900, value=735, step=1
            )
        else:
            credit_score = np.nan

        credit_history_months = st.number_input(
            "Credit history (months)",
            min_value=0,
            max_value=1200,
            value=(48 if has_credit_score == "Yes" else 0),
            step=1,
        )

    with right:
        num_active_loans = st.number_input(
            "Number of active loans",
            min_value=0, max_value=100, value=1, step=1
        )

        delinquencies_24m = st.number_input(
            "Late/missed repayments in last 24 months",
            min_value=0, max_value=100, value=0, step=1
        )

        loan_purpose = st.selectbox(
            "Loan purpose",
            [
                "Medical",
                "Debt Consolidation",
                "Education",
                "Home Improvement",
                "Travel",
                "Wedding",
                "Business",
                "Other",
            ],
        )

        loan_amount = st.number_input(
            "Requested loan amount (₹)",
            min_value=1.0,
            max_value=100_000_000.0,
            value=300000.0,
            step=5000.0,
        )

        tenure_months = st.selectbox(
            "Requested tenure (months)",
            [6, 12, 18, 24, 36, 48, 60, 72, 84, 96],
        )

        proposed_emi = st.number_input(
            "Proposed monthly EMI (₹)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=10250.0,
            step=250.0,
        )

        city_tier = st.selectbox(
            "City tier",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Tier 1 — Metro",
                2: "Tier 2 — Large city",
                3: "Tier 3 — Smaller city",
            }[x],
        )

    submitted = st.form_submit_button(
        "Predict decision",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if submitted:
    # Match the training notebook's feature engineering.
    corrected_income = monthly_income
    income_corrected = False

    if corrected_income > 600_000:
        corrected_income = corrected_income / 12
        income_corrected = True

    if corrected_income <= 0:
        st.error(
            "Monthly income must be greater than ₹0 so affordability "
            "ratios can be calculated."
        )
        st.stop()

    foir = (existing_emi + proposed_emi) / corrected_income
    loan_to_income = loan_amount / (corrected_income * 12)

    is_ntc = int(
        pd.isna(credit_score) or credit_history_months == 0
    )

    features = pd.DataFrame(
        [{
            "credit_score": credit_score,
            "is_ntc": is_ntc,
            "foir": foir,
            "loan_to_income": loan_to_income,
            "delinquencies_24m": delinquencies_24m,
            "num_active_loans": num_active_loans,
            "years_in_current_job": years_in_current_job,
            "age": age,
            "tenure_months": tenure_months,
            "city_tier": city_tier,
            "employment_type": employment_type,
            "loan_purpose": loan_purpose,
        }]
    )

    expected_features = [
        "credit_score",
        "is_ntc",
        "foir",
        "loan_to_income",
        "delinquencies_24m",
        "num_active_loans",
        "years_in_current_job",
        "age",
        "tenure_months",
        "city_tier",
        "employment_type",
        "loan_purpose",
    ]

    features = features[expected_features]

    try:
        prediction = str(model.predict(features)[0])

        class_labels = [
            str(c) for c in model.named_steps["classifier"].classes_
        ]

        probabilities = model.predict_proba(features)[0]

        probability_map = {
            label: float(prob)
            for label, prob in zip(class_labels, probabilities)
        }

    except Exception as exc:
        st.error(
            "Prediction failed. Ensure the model artifact was created "
            "by the supplied notebook and uses the same feature schema."
        )
        st.exception(exc)
        st.stop()

    # -----------------------------------------------------
    # PREDICTION RESULTS
    # -----------------------------------------------------

    st.divider()
    st.subheader("Prediction result")

    st.caption(
        "The result below reflects the model's learned historical "
        "decision patterns."
    )

    # Use a custom HTML result card instead of st.success/st.warning.
    # This prevents global CSS from making the result text white.

    if prediction.strip().lower() == "approved":
        result_class = "prediction-approved"
    else:
        result_class = "prediction-declined"

    safe_prediction = (
        prediction.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    st.markdown(
        f"""
        <div class="prediction-result {result_class}">
            Model prediction: {safe_prediction}
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(2, gap="large")

    with metric_cols[0]:
        st.metric(
            "Probability of Approved",
            f"{100 * probability_map.get('Approved', 0.0):.1f}%",
        )

    with metric_cols[1]:
        st.metric(
            "Probability of Declined",
            f"{100 * probability_map.get('Declined', 0.0):.1f}%",
        )

    st.subheader("Calculated affordability indicators")

    col1, col2, col3 = st.columns(3, gap="medium")

    col1.metric("FOIR", f"{foir:.1%}")
    col2.metric("Loan / annual income", f"{loan_to_income:.2f}×")
    col3.metric("New-to-credit", "Yes" if is_ntc else "No")

    if income_corrected:
        st.warning(
            "The entered monthly income exceeded ₹6,00,000, so the app "
            "divided it by 12 using the same correction rule as the "
            "training notebook. Please verify this value."
        )

    if is_ntc:
        st.warning(
            "This applicant is marked new-to-credit. The model may be "
            "less reliable for this group; route the application for "
            "human review rather than relying on the prediction alone."
        )

    with st.expander("View model probabilities and input features"):
        st.json({
            k: round(v, 4)
            for k, v in probability_map.items()
        })

        display_features = features.copy()

        display_features["foir"] = display_features["foir"].map(
            lambda x: f"{x:.4f}"
        )

        display_features["loan_to_income"] = (
            display_features["loan_to_income"].map(
                lambda x: f"{x:.4f}"
            )
        )

        st.dataframe(
            display_features,
            use_container_width=True,
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Model target: historical `decision` (Approved/Declined). "
    "It is not a validated default-risk model. Predictions are "
    "informational and should be reviewed by a qualified human."
)