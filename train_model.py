import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("loan_approval_data.csv")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ==========================================
# 2. Handle Missing Values
# ==========================================

categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["float64"]).columns

# Numerical columns
from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

# Categorical columns
cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])


# ==========================================
# 3. Remove Applicant ID
# ==========================================

df = df.drop(columns=["Applicant_ID"])


# ==========================================
# 4. Encode Target and Education Level
# ==========================================

le = LabelEncoder()

df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])


# ==========================================
# 5. One-Hot Encode Categorical Features
# ==========================================

cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]

onehot = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

encoded = onehot.fit_transform(df[cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=onehot.get_feature_names_out(cols),
    index=df.index
)

df = pd.concat(
    [
        df.drop(columns=cols),
        encoded_df
    ],
    axis=1
)


# ==========================================
# 6. Feature Engineering
# ==========================================

df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2


# ==========================================
# 7. Remove Original Credit Score and DTI
# ==========================================

X = df.drop(
    columns=[
        "Loan_Approved",
        "Credit_Score",
        "DTI_Ratio"
    ]
)

y = df["Loan_Approved"]


# ==========================================
# 8. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 9. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 10. Train Naive Bayes Model
# ==========================================

model = GaussianNB()

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 11. Evaluate Model
# ==========================================

accuracy = model.score(
    X_test_scaled,
    y_test
)

print("\nModel Training Completed!")
print("--------------------------------")
print("Model: Gaussian Naive Bayes")
print("Accuracy:", accuracy)
print("--------------------------------")


# ==========================================
# 12. Save Model Components
# ==========================================

joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(onehot, "onehot_encoder.pkl")
joblib.dump(le, "label_encoder.pkl")

# Save the exact feature order
joblib.dump(list(X.columns), "feature_columns.pkl")


print("\nFiles saved successfully!")
print("✓ naive_bayes_model.pkl")
print("✓ scaler.pkl")
print("✓ onehot_encoder.pkl")
print("✓ label_encoder.pkl")
print("✓ feature_columns.pkl")