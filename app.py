import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CreditWise",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700&display=swap" rel="stylesheet">
<style>
/* CreditWise presentation tokens. No data or model behavior is changed. */
:root {
    --cw-background: #F7F9FC;
    --cw-surface: #FFFFFF;
    --cw-foreground: #182230;
    --cw-muted: #66758B;
    --cw-primary: #2563EB;
    --cw-primary-hover: #1D4ED8;
    --cw-accent: #DCE7F7;
    --cw-border: #E1E7EF;
    --cw-sidebar: #182230;
    --cw-sidebar-muted: #AAB8CC;
    --cw-sidebar-hover: #243247;
    --cw-on-primary: #FFFFFF;
    --cw-success: #15785A;
    --cw-success-surface: #EBF7F1;
    --cw-danger: #B83B4B;
    --cw-danger-surface: #FCEDF0;
    --cw-focus: 0 0 0 3px #2563EB26;
    --cw-shadow: 0 3px 14px #18223006;
    --cw-body: 'Manrope', sans-serif;
    --cw-heading: 'Sora', sans-serif;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: var(--cw-background);
    color: var(--cw-foreground);
    font-family: var(--cw-body);
    letter-spacing: 0;
}
[data-testid="stMainBlockContainer"], .main .block-container {
    max-width: 1480px;
    padding: 3rem 3rem 4rem;
}
[data-testid="stMain"] p, [data-testid="stMain"] li,
[data-testid="stMain"] label, [data-testid="stMain"] input,
[data-testid="stMain"] button { font-family: var(--cw-body); }
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4 {
    font-family: var(--cw-heading);
    color: var(--cw-foreground);
    letter-spacing: 0;
    line-height: 1.35;
}
[data-testid="stMain"] h1 { font-size: 2.15rem; font-weight: 700; padding-bottom: .45rem; }
[data-testid="stMain"] h2 { font-size: 1.3rem; font-weight: 600; padding: 1rem 0 .7rem; }
[data-testid="stMain"] h3 { font-size: 1.05rem; font-weight: 600; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {
    line-height: 1.8;
    color: var(--cw-muted);
}
[data-testid="stHeader"] { background: var(--cw-background); }
[data-testid="stToolbar"] { color: var(--cw-muted); }
hr { border-color: var(--cw-border); margin: 1.4rem 0; }
/* Keep Streamlit's native collapse and reopen controls available. */
section[data-testid="stSidebar"] {
    background: var(--cw-sidebar);
    border-right: 1px solid var(--cw-sidebar-hover);
}
section[data-testid="stSidebar"] > div { background: var(--cw-sidebar); }
[data-testid="stSidebarUserContent"] { padding: 2rem 1.15rem; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--cw-sidebar-muted);
    font-family: var(--cw-body);
}
section[data-testid="stSidebar"] hr { border-color: var(--cw-sidebar-hover); }
section[data-testid="stSidebar"] button { color: var(--cw-on-primary); }
.sidebar-title {
    font: 700 1.65rem/1.5 var(--cw-heading);
    color: var(--cw-on-primary);
    padding: .25rem .45rem;
}
.sidebar-subtitle {
    font: 500 .76rem/1.7 var(--cw-body);
    color: var(--cw-sidebar-muted);
    padding: 0 .45rem;
}
.sidebar-nav-title {
    font: 700 .68rem/1.5 var(--cw-body);
    color: var(--cw-sidebar-muted);
    padding: .45rem .65rem;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 46px;
    justify-content: flex-start;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: .7rem .85rem;
    margin: .15rem 0;
    box-shadow: none;
    transition: background .16s ease, border-color .16s ease;
}
section[data-testid="stSidebar"] .stButton > button p {
    color: var(--cw-sidebar-muted);
    font-size: .85rem;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--cw-sidebar-hover);
    border-color: var(--cw-sidebar-hover);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--cw-primary);
    border-color: var(--cw-primary);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p,
section[data-testid="stSidebar"] .stButton > button:hover p { color: var(--cw-on-primary); }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: .72rem; padding: 0 .45rem;
}
/* Metric values wrap instead of truncating long model names. */
[data-testid="stMetric"] {
    background: var(--cw-surface);
    border: 1px solid var(--cw-border);
    border-top: 3px solid var(--cw-accent);
    border-radius: 8px;
    padding: 1.15rem 1.25rem;
    min-height: 130px;
    box-shadow: var(--cw-shadow);
}
[data-testid="stMetricLabel"] p {
    color: var(--cw-muted);
    font: 600 .76rem/1.5 var(--cw-body);
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    color: var(--cw-foreground);
    font: 600 1.65rem/1.45 var(--cw-heading);
    white-space: normal;
    overflow: visible;
    overflow-wrap: anywhere;
    text-overflow: clip;
}
[data-testid="stMetricDelta"] { font-size: .74rem; }
/* Native controls, with consistent focus and error states. */
[data-testid="stWidgetLabel"] p {
    color: var(--cw-foreground) !important;
    font-size: .8rem;
    font-weight: 600;
}
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: var(--cw-surface);
    border: 1px solid var(--cw-border);
    border-radius: 6px;
    min-height: 46px;
    color: var(--cw-foreground);
}
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
    color: var(--cw-foreground);
    background: var(--cw-surface);
    font-size: .88rem;
}
[data-testid="stNumberInput"] button {
    background: var(--cw-background);
    color: var(--cw-muted);
    border-color: var(--cw-border);
}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {
    border-color: var(--cw-primary);
    box-shadow: var(--cw-focus);
}
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: var(--cw-surface);
    color: var(--cw-foreground);
    font-family: var(--cw-body);
}
[data-testid="stMain"] .stButton > button {
    border-radius: 6px;
    min-height: 46px;
    font-weight: 700;
    border-color: var(--cw-border);
    transition: background .16s ease;
}
[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--cw-primary);
    border-color: var(--cw-primary);
    color: var(--cw-on-primary);
}
[data-testid="stMain"] .stButton > button[kind="primary"] p,
[data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"] p { color: var(--cw-on-primary); }
[data-testid="stMain"] .stButton > button[kind="primary"]:hover,
[data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"]:hover { background: var(--cw-primary-hover); }
button:focus-visible, a:focus-visible { outline: 2px solid var(--cw-primary); outline-offset: 3px; }
[data-testid="stDataFrame"] {
    border: 1px solid var(--cw-border);
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stImage"] { background: var(--cw-surface); border-radius: 8px; }
.info-card {
    background: var(--cw-surface);
    border: 1px solid var(--cw-border);
    border-radius: 8px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--cw-shadow);
}
.info-card h3 { margin: 0 0 .8rem; }
.info-card p { color: var(--cw-muted); line-height: 1.8; margin-bottom: .5rem; }
.prediction-card { padding: 2rem; border-radius: 8px; text-align: center; margin-top: 1.2rem; }
.approved { background: var(--cw-success-surface); border: 1px solid var(--cw-success); }
.rejected { background: var(--cw-danger-surface); border: 1px solid var(--cw-danger); }
.prediction-title { font: 600 1.7rem var(--cw-heading); }
.prediction-score { font: 700 2.5rem var(--cw-heading); margin-top: .8rem; }
[data-testid="stAlert"] { border-radius: 8px; }
[data-testid="stProgress"] [role="progressbar"] { accent-color: var(--cw-primary); }
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"], .main .block-container { padding: 2.5rem 1.25rem 3rem; }
    [data-testid="stMetric"] { padding: 1rem; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { font-size: 1.35rem; }
}
@media (max-width: 640px) {
    [data-testid="stMain"] h1 { font-size: 1.75rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 1rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important;
    }
    [data-testid="stMetric"] { min-height: 112px; }
    .info-card { padding: 1.2rem; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition: none !important; animation: none !important; } }
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD SAVED MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():

    model = joblib.load("naive_bayes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    onehot_encoder = joblib.load("onehot_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, scaler, onehot_encoder, feature_columns


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv("loan_approval_data.csv")


# Load everything

model, scaler, onehot_encoder, feature_columns = load_model_files()

df = load_dataset()


# ============================================================
# DATA PREPARATION FOR ANALYSIS
# ============================================================

analysis_df = df.copy()


# ============================================================
# MODEL PERFORMANCE CALCULATION
# ============================================================

@st.cache_data
def calculate_model_metrics(data):

    temp = data.copy()

    # --------------------------------------------------------
    # Remove rows with missing target labels
    # --------------------------------------------------------
    # Loan_Approved is the target, so missing target values
    # must not be imputed or used for model evaluation.
    temp = temp.dropna(
        subset=["Loan_Approved"]
    ).copy()

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    categorical_cols = temp.select_dtypes(
        include=["object"]
    ).columns

    numerical_cols = temp.select_dtypes(
        include=["float64", "int64"]
    ).columns

    numerical_cols = [
        col
        for col in numerical_cols
        if col != "Applicant_ID"
    ]

    # Numerical imputation

    num_imputer = SimpleImputer(
        strategy="mean"
    )

    temp[numerical_cols] = num_imputer.fit_transform(
        temp[numerical_cols]
    )

    # Categorical imputation

    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    temp[categorical_cols] = cat_imputer.fit_transform(
        temp[categorical_cols]
    )

    # --------------------------------------------------------
    # Remove ID
    # --------------------------------------------------------

    temp = temp.drop(
        columns=["Applicant_ID"]
    )

    # --------------------------------------------------------
    # Encode Education Level and Target
    # --------------------------------------------------------

    education_encoder = LabelEncoder()

    temp["Education_Level"] = (
        education_encoder.fit_transform(
            temp["Education_Level"]
        )
    )

    target_encoder = LabelEncoder()

    temp["Loan_Approved"] = (
        target_encoder.fit_transform(
            temp["Loan_Approved"]
        )
    )

    # --------------------------------------------------------
    # One Hot Encoding
    # --------------------------------------------------------

    categorical_columns = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded = encoder.fit_transform(
        temp[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_columns
        ),
        index=temp.index
    )

    temp = pd.concat(
        [
            temp.drop(
                columns=categorical_columns
            ),
            encoded_df
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    temp["DTI_Ratio_sq"] = (
        temp["DTI_Ratio"] ** 2
    )

    temp["Credit_Score_sq"] = (
        temp["Credit_Score"] ** 2
    )

    # --------------------------------------------------------
    # Final features
    # --------------------------------------------------------

    X = temp.drop(
        columns=[
            "Loan_Approved",
            "Credit_Score",
            "DTI_Ratio"
        ]
    )

    y = temp["Loan_Approved"]

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler_temp = StandardScaler()

    X_train_scaled = scaler_temp.fit_transform(
        X_train
    )

    X_test_scaled = scaler_temp.transform(
        X_test
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "K-Nearest Neighbors":
            KNeighborsClassifier(),

        "Gaussian Naive Bayes":
            GaussianNB()
    }

    results = []

    confusion_matrices = {}

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    for name, clf in models.items():

        clf.fit(
            X_train_scaled,
            y_train
        )

        predictions = clf.predict(
            X_test_scaled
        )

        results.append(
            {
                "Model": name,

                "Accuracy": accuracy_score(
                    y_test,
                    predictions
                ),

                "Precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            }
        )

        confusion_matrices[name] = confusion_matrix(
            y_test,
            predictions
        )

    results_df = pd.DataFrame(results)

    return results_df, confusion_matrices


# ============================================================
# CALCULATE METRICS
# ============================================================

metrics_df, confusion_matrices = calculate_model_metrics(
    df
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">💳 CreditWise</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Loan Intelligence Platform</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-nav-title">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    # Initialize page

    if "page" not in st.session_state:
        st.session_state.page = "🏠 Dashboard"

    navigation_items = [
        "🏠 Dashboard",
        "🔮 Loan Prediction",
        "📊 Data Insights",
        "🤖 Model Performance",
        "ℹ️ About"
    ]

    # Navigation buttons

    for page_name in navigation_items:

        if st.button(
            page_name,
            key=f"nav_{page_name}",
            type="primary" if st.session_state.page == page_name else "secondary",
            use_container_width=True
        ):

            st.session_state.page = page_name
            st.rerun()

    page = st.session_state.page

    st.divider()

    st.caption("CreditWise v1.0")
    st.caption("Machine Learning Loan Assessment")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("CreditWise")

    st.subheader(
        "Intelligent Loan Approval Prediction Platform"
    )

    st.write(
        "Analyze loan applications and predict approval "
        "using machine learning."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_records = len(df)

    approved = (
        df["Loan_Approved"]
        .value_counts()
        .get("Yes", 0)
    )

    rejected = (
        df["Loan_Approved"]
        .value_counts()
        .get("No", 0)
    )

    valid_decisions = approved + rejected

    approval_rate = (
        approved / valid_decisions * 100
        if valid_decisions > 0
        else 0
    )

    avg_credit_score = df[
        "Credit_Score"
    ].mean()

    avg_loan_amount = df[
        "Loan_Amount"
    ].mean()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Applications",
            f"{total_records:,}"
        )

    with col2:

        st.metric(
            "Approved",
            f"{approved:,}"
        )

    with col3:

        st.metric(
            "Rejected",
            f"{rejected:,}"
        )

    with col4:

        st.metric(
            "Approval Rate",
            f"{approval_rate:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # SECONDARY METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Average Credit Score",
            f"{avg_credit_score:.0f}"
        )

    with col2:

        st.metric(
            "Average Loan Amount",
            f"{avg_loan_amount:,.0f}"
        )

    with col3:

        best_model = metrics_df.loc[
            metrics_df["Accuracy"].idxmax(),
            "Model"
        ]

        best_accuracy = metrics_df.loc[
            metrics_df["Accuracy"].idxmax(),
            "Accuracy"
        ]

        st.metric(
            "Best Accuracy Model",
            best_model,
            f"{best_accuracy * 100:.2f}% accuracy"
        )

    st.divider()

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    st.header("📌 Platform Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-card">

            <h3>🎯 What CreditWise Does</h3>

            <p>
            CreditWise analyzes applicant information such as
            income, credit score, existing loans, savings,
            employment status and loan details.
            </p>

            <p>
            The system processes these features through a
            machine learning pipeline and predicts whether
            a loan application is likely to be approved.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">

            <h3>🧠 Machine Learning Pipeline</h3>

            <p>
            Raw Data
            <br>↓<br>
            Data Cleaning
            <br>↓<br>
            Encoding & Feature Engineering
            <br>↓<br>
            Feature Scaling
            <br>↓<br>
            Machine Learning Models
            <br>↓<br>
            Loan Prediction
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # DATASET SNAPSHOT
    # --------------------------------------------------------

    st.header("📊 Dataset Snapshot")

    snapshot_col1, snapshot_col2 = st.columns(2)

    with snapshot_col1:

        st.write(
            "The application is currently working with "
            "the original loan approval dataset."
        )

        st.dataframe(
            df.head(8),
            use_container_width=True,
            hide_index=True
        )

    with snapshot_col2:

        missing_values = df.isnull().sum()

        missing_summary = pd.DataFrame(
            {
                "Column": missing_values.index,
                "Missing Values": missing_values.values
            }
        )

        missing_summary = (
            missing_summary[
                missing_summary["Missing Values"] > 0
            ]
            .sort_values(
                "Missing Values",
                ascending=False
            )
        )

        st.write("Missing-value overview")

        st.dataframe(
            missing_summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# LOAN PREDICTION
# ============================================================

elif page == "🔮 Loan Prediction":

    st.title("🔮 Loan Approval Prediction")

    st.write(
        "Enter applicant and loan information to generate "
        "an AI-powered loan decision."
    )

    st.divider()

    # --------------------------------------------------------
    # APPLICANT INFORMATION
    # --------------------------------------------------------

    st.header("👤 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=0
        )

    with col2:

        coapplicant_income = st.number_input(
            "Co-applicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300.0,
            max_value=850.0,
            value=650.0,
            step=10.0
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0.0,
            value=1.0,
            step=1.0
        )

    with col3:

        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.01
        )

        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        collateral_value = st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=30000.0,
            step=1000.0
        )

    # --------------------------------------------------------
    # LOAN INFORMATION
    # --------------------------------------------------------

    st.header("🏦 Loan Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

    with col2:

        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=1.0,
            value=48.0,
            step=6.0
        )

    with col3:

        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Car",
                "Business",
                "Education",
                "Home",
                "Personal"
            ]
        )

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.header("📋 Personal & Employment Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        employment_status = st.selectbox(
            "Employment Status",
            [
                "Salaried",
                "Self-employed",
                "Unemployed"
            ]
        )

    with col2:

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married"
            ]
        )

    with col3:

        property_area = st.selectbox(
            "Property Area",
            [
                "Urban",
                "Semiurban",
                "Rural"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        education_level = st.selectbox(
            "Education Level",
            [
                "Graduate",
                "Not Graduate"
            ]
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

    employer_category = st.selectbox(
        "Employer Category",
        [
            "Private",
            "Government",
            "MNC",
            "Unemployed"
        ]
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Loan Approval",
        use_container_width=True,
        type="primary"
    ):

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            {
                "Applicant_Income": [applicant_income],

                "Coapplicant_Income": [
                    coapplicant_income
                ],

                "Employment_Status": [
                    employment_status
                ],

                "Age": [age],

                "Marital_Status": [
                    marital_status
                ],

                "Dependents": [
                    dependents
                ],

                "Credit_Score": [
                    credit_score
                ],

                "Existing_Loans": [
                    existing_loans
                ],

                "DTI_Ratio": [
                    dti_ratio
                ],

                "Savings": [
                    savings
                ],

                "Collateral_Value": [
                    collateral_value
                ],

                "Loan_Amount": [
                    loan_amount
                ],

                "Loan_Term": [
                    loan_term
                ],

                "Loan_Purpose": [
                    loan_purpose
                ],

                "Property_Area": [
                    property_area
                ],

                "Education_Level": [
                    education_level
                ],

                "Gender": [
                    gender
                ],

                "Employer_Category": [
                    employer_category
                ]
            }
        )

        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        input_data["DTI_Ratio_sq"] = (
            input_data["DTI_Ratio"] ** 2
        )

        input_data["Credit_Score_sq"] = (
            input_data["Credit_Score"] ** 2
        )

        # ----------------------------------------------------
        # DROP ORIGINAL FEATURES
        # ----------------------------------------------------

        input_data = input_data.drop(
            columns=[
                "Credit_Score",
                "DTI_Ratio"
            ]
        )

        # ----------------------------------------------------
        # EDUCATION ENCODING
        # ----------------------------------------------------

        education_mapping = {
            "Graduate": 0,
            "Not Graduate": 1
        }

        input_data["Education_Level"] = (
            input_data["Education_Level"]
            .map(education_mapping)
        )

        # ----------------------------------------------------
        # ONE-HOT ENCODING
        # ----------------------------------------------------

        categorical_columns = [
            "Employment_Status",
            "Marital_Status",
            "Loan_Purpose",
            "Property_Area",
            "Gender",
            "Employer_Category"
        ]

        encoded = onehot_encoder.transform(
            input_data[categorical_columns]
        )

        if hasattr(encoded, "toarray"):

            encoded = encoded.toarray()

        encoded_df = pd.DataFrame(
            encoded,
            columns=onehot_encoder.get_feature_names_out(
                categorical_columns
            ),
            index=input_data.index
        )

        input_data = input_data.drop(
            columns=categorical_columns
        )

        input_data = pd.concat(
            [
                input_data,
                encoded_df
            ],
            axis=1
        )

        # ----------------------------------------------------
        # FEATURE ORDER
        # ----------------------------------------------------

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_scaled
        )[0]

        probabilities = model.predict_proba(
            input_scaled
        )[0]

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        approved_probability = (
            probabilities[1] * 100
        )

        rejected_probability = (
            probabilities[0] * 100
        )

        st.divider()

        st.header("📊 Prediction Result")

        # ----------------------------------------------------
        # APPROVAL RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.success(
                "## ✅ Loan Approved"
            )

            st.write(
                "The model predicts that this application "
                "is likely to be approved."
            )

        else:

            st.error(
                "## ❌ Loan Not Approved"
            )

            st.write(
                "The model predicts that this application "
                "is unlikely to be approved."
            )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Approval Probability",
                f"{approved_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Rejection Probability",
                f"{rejected_probability:.2f}%"
            )

        st.progress(
            approved_probability / 100
        )


# ============================================================
# DATA INSIGHTS
# ============================================================

elif page == "📊 Data Insights":

    st.title("📊 Data Insights")

    st.write(
        "Explore patterns and relationships in the original "
        "loan approval dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET KPIs
    # --------------------------------------------------------

    valid_df = analysis_df[
        analysis_df["Loan_Approved"].notna()
    ]

    approved = (
        valid_df["Loan_Approved"] == "Yes"
    ).sum()

    rejected = (
        valid_df["Loan_Approved"] == "No"
    ).sum()

    approval_rate = (
        approved / len(valid_df) * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            f"{len(analysis_df):,}"
        )

    with col2:

        st.metric(
            "Approved",
            f"{approved:,}"
        )

    with col3:

        st.metric(
            "Rejected",
            f"{rejected:,}"
        )

    with col4:

        st.metric(
            "Approval Rate",
            f"{approval_rate:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # APPROVAL DISTRIBUTION
    # --------------------------------------------------------

    st.header("🎯 Loan Approval Distribution")

    approval_counts = (
        valid_df["Loan_Approved"]
        .value_counts()
        .reindex(
            ["Yes", "No"],
            fill_value=0
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        bars = ax.bar(
            ["Approved", "Rejected"],
            [
                approval_counts["Yes"],
                approval_counts["No"]
            ]
        )

        ax.set_ylabel(
            "Number of Applications"
        )

        ax.set_title(
            "Loan Approval Distribution",
            color="#0f172a"
        )

        ax.tick_params(
            colors="#334155"
        )

        ax.xaxis.label.set_color("#334155")
        ax.yaxis.label.set_color("#334155")

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                color="#111827",
                fontweight="bold"
            )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        approval_table = pd.DataFrame(
            {
                "Decision": [
                    "Approved",
                    "Rejected"
                ],

                "Count": [
                    approval_counts["Yes"],
                    approval_counts["No"]
                ],

                "Percentage": [
                    approval_counts["Yes"]
                    / len(valid_df) * 100,

                    approval_counts["No"]
                    / len(valid_df) * 100
                ]
            }
        )

        st.dataframe(
            approval_table.style.format(
                {
                    "Percentage": "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # CATEGORICAL DISTRIBUTIONS
    # --------------------------------------------------------

    st.header(
        "📋 Applicant & Loan Characteristics"
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_category = st.selectbox(
            "Select Category",
            [
                "Employment_Status",
                "Loan_Purpose",
                "Property_Area",
                "Education_Level",
                "Gender",
                "Employer_Category"
            ]
        )

    category_counts = (
        valid_df[selected_category]
        .value_counts()
    )

    with col2:

        st.write(
            f"Distribution of {selected_category}"
        )

        st.dataframe(
            category_counts.rename(
                "Count"
            ),
            use_container_width=True
        )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    bars = ax.bar(
        category_counts.index.astype(str),
        category_counts.values
    )

    ax.set_ylabel("Count")

    ax.set_title(
        f"{selected_category} Distribution",
        color="#0f172a"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
        colors="#334155"
    )

    ax.tick_params(
        axis="y",
        colors="#334155"
    )

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            color="#111827",
            fontweight="bold"
        )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # CREDIT SCORE ANALYSIS
    # --------------------------------------------------------

    st.header("💳 Credit Score Analysis")

    credit_stats = (
        valid_df
        .groupby("Loan_Approved")[
            "Credit_Score"
        ]
        .agg(
            ["mean", "median", "min", "max"]
        )
    )

    st.dataframe(
        credit_stats.style.format(
            "{:.2f}"
        ),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    valid_df.boxplot(
        column="Credit_Score",
        by="Loan_Approved",
        ax=ax
    )

    ax.set_title(
        "Credit Score by Loan Decision",
        color="#0f172a"
    )

    ax.set_xlabel(
        "Loan Decision"
    )

    ax.set_ylabel(
        "Credit Score"
    )

    ax.tick_params(
        colors="#334155"
    )

    plt.suptitle("")

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # INCOME ANALYSIS
    # --------------------------------------------------------

    st.header("💰 Income Analysis")

    income_data = (
        valid_df
        .groupby("Loan_Approved")[
            [
                "Applicant_Income",
                "Coapplicant_Income"
            ]
        ]
        .mean()
    )

    st.dataframe(
        income_data.style.format(
            "{:,.2f}"
        ),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    income_data.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Income by Loan Decision",
        color="#0f172a"
    )

    ax.set_ylabel(
        "Average Income"
    )

    ax.set_xlabel(
        "Loan Decision"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
        colors="#334155"
    )

    ax.tick_params(
        axis="y",
        colors="#334155"
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------------

    st.header(
        "🔥 Numerical Feature Correlation"
    )

    st.write(
        "Correlation values range from -1 to +1. "
        "Values closer to +1 indicate a strong positive "
        "relationship, while values closer to -1 indicate "
        "a strong negative relationship."
    )

    numeric_columns = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term"
    ]

    correlation = (
        valid_df[numeric_columns]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(13, 10)
    )

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(
        range(len(correlation.columns))
    )

    ax.set_yticks(
        range(len(correlation.columns))
    )

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right",
        fontsize=9,
        color="#111827"
    )

    ax.set_yticklabels(
        correlation.columns,
        fontsize=9,
        color="#111827"
    )

    # Add actual correlation values

    for i in range(
        len(correlation.columns)
    ):

        for j in range(
            len(correlation.columns)
        ):

            value = correlation.iloc[i, j]

            # Choose readable text color

            text_color = (
                "white"
                if abs(value) > 0.55
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                fontweight="bold"
            )

    ax.set_title(
        "Correlation Matrix",
        color="#0f172a",
        fontsize=14,
        fontweight="bold"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation"
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Performance of the machine learning models evaluated "
        "using the project's dataset and preprocessing pipeline."
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    st.header("📋 Model Evaluation Metrics")

    display_metrics = metrics_df.copy()

    display_metrics[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ] *= 100

    st.dataframe(
        display_metrics.style.format(
            {
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1 Score": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # ACCURACY CHART
    # --------------------------------------------------------

    st.header("📈 Model Accuracy")

    accuracy_chart = (
        metrics_df
        .set_index("Model")[
            "Accuracy"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        accuracy_chart.index,
        accuracy_chart.values * 100
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        "Model Accuracy Comparison",
        color="#0f172a"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
        colors="#334155"
    )

    ax.tick_params(
        axis="y",
        colors="#334155"
    )

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            color="#111827",
            fontweight="bold"
        )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # ALL METRICS
    # --------------------------------------------------------

    st.header("📊 Performance Comparison")

    chart_metrics = (
        metrics_df
        .set_index("Model")[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ] * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    chart_metrics.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel(
        "Score (%)"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        "Model Performance Metrics",
        color="#0f172a"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
        colors="#334155"
    )

    ax.tick_params(
        axis="y",
        colors="#334155"
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.header("🔲 Confusion Matrix")

    selected_model = st.selectbox(
        "Select Model",
        list(confusion_matrices.keys())
    )

    cm = confusion_matrices[
        selected_model
    ]

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        cm,
        cmap="Blues"
    )

    ax.set_title(
        f"{selected_model} — Confusion Matrix",
        color="#0f172a"
    )

    ax.set_xlabel(
        "Predicted",
        color="#334155"
    )

    ax.set_ylabel(
        "Actual",
        color="#334155"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        ["No", "Yes"],
        color="#334155"
    )

    ax.set_yticklabels(
        ["No", "Yes"],
        color="#334155"
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#111827"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model_row = metrics_df.loc[
        metrics_df["Accuracy"].idxmax()
    ]

    st.success(
        f"Best accuracy: "
        f"{best_model_row['Model']} "
        f"({best_model_row['Accuracy'] * 100:.2f}%)"
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About CreditWise")

    st.subheader(
        "Machine Learning Based Loan Approval Prediction"
    )

    st.divider()

    # --------------------------------------------------------
    # PROBLEM STATEMENT
    # --------------------------------------------------------

    st.header("📌 Problem Statement")

    st.write(
        """
        CreditWise is designed to assist in the initial assessment
        of loan applications using machine learning.

        The system considers financial, demographic, employment
        and loan-related attributes to predict whether an
        application is likely to be approved.
        """
    )

    # --------------------------------------------------------
    # OBJECTIVES
    # --------------------------------------------------------

    st.header("🎯 Objectives")

    st.markdown(
        """
        - Analyze loan application data.
        - Perform data preprocessing.
        - Engineer useful features.
        - Compare classification algorithms.
        - Identify a suitable prediction model.
        - Build an interactive prediction application.
        - Present meaningful data insights.
        """
    )

    # --------------------------------------------------------
    # MACHINE LEARNING WORKFLOW
    # --------------------------------------------------------

    st.header("🧠 Machine Learning Workflow")

    st.markdown(
        """
        **Raw Dataset**

        ↓

        **Data Cleaning**

        ↓

        **Categorical Encoding**

        ↓

        **Feature Engineering**

        ↓

        **Train/Test Split**

        ↓

        **Feature Scaling**

        ↓

        **Model Training**

        ↓

        **Model Evaluation**

        ↓

        **Loan Prediction**
        """
    )

    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.header("🛠️ Technologies")

    st.markdown(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Matplotlib
        - Streamlit
        - Joblib
        """
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    st.header("🤖 Models Evaluated")

    st.markdown(
        """
        - Logistic Regression
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        """
    )

    # --------------------------------------------------------
    # PROJECT FILES
    # --------------------------------------------------------

    st.header("📁 Project Components")

    project_files = pd.DataFrame(
        {
            "File": [
                "credit_wise.ipynb",
                "loan_approval_data.csv",
                "train_model.py",
                "naive_bayes_model.pkl",
                "scaler.pkl",
                "onehot_encoder.pkl",
                "feature_columns.pkl",
                "app.py"
            ],

            "Purpose": [
                "Original ML notebook",
                "Raw loan dataset",
                "Model training script",
                "Saved Gaussian Naive Bayes model",
                "Saved feature scaler",
                "Saved categorical encoder",
                "Saved feature ordering",
                "Streamlit application"
            ]
        }
    )

    st.dataframe(
        project_files,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.caption(
        "CreditWise — Loan Approval Prediction System"
    )
