
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Prediction Using Machine Learning")
st.write("Enter health measurements to predict the dataset class.")

st.warning(
    "Educational project only. This prediction is not a medical diagnosis."
)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset of Diabetes.csv")
    df.columns = df.columns.str.strip()
    df["CLASS"] = df["CLASS"].astype(str).str.strip()
    df = df[df["CLASS"].isin(["Y", "N", "P"])].copy()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# Prepare data
X = df.drop(columns=["ID", "No_Pation", "CLASS"], errors="ignore")
y = df["CLASS"]

# Convert categorical columns to numeric
X = pd.get_dummies(X, dummy_na=True)
X = X.fillna(X.median(numeric_only=True)).fillna(0)

# Train model automatically when app starts
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

st.divider()
st.subheader("Enter Health Measurements")

# Create input fields
inputs = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(X.columns):
    with (col1 if i % 2 == 0 else col2):
        if feature in df.columns and (
            df[feature].dtype == "object"
            or str(df[feature].dtype) == "category"
        ):
            choices = df[feature].dropna().astype(str).unique().tolist()
            inputs[feature] = st.selectbox(feature, choices)
        else:
            if feature in df.columns:
                values = pd.to_numeric(df[feature], errors="coerce").dropna()
                default = float(values.median()) if not values.empty else 0.0
            else:
                default = 0.0

            inputs[feature] = st.number_input(
                feature,
                value=default,
                step=0.1
            )

if st.button("🔍 Predict Class", use_container_width=True):
    input_df = pd.DataFrame([inputs])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted dataset class: {prediction}")

    probabilities = model.predict_proba(input_df)[0]
    st.write("Model probability estimates:")

    st.bar_chart(
        pd.DataFrame({
            "Class": model.classes_,
            "Probability (%)": probabilities * 100
        }).set_index("Class")
    )

st.divider()
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)