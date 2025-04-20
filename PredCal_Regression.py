# Packages/Libraries
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve, classification_report)
import seaborn as sns

# Disable deprecated Streamlit warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Absolute paths to datasets and image
DIABETES_DATA_PATH = "/Users/emmanuelawuni/PycharmProjects/NLPProject/DiabetesData.csv"
IMAGE_PATH = "/Users/emmanuelawuni/PycharmProjects/NLPProject/image.jpeg"

# Load the dataset
try:
    data = pd.read_csv(DIABETES_DATA_PATH)
except FileNotFoundError as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# Data Cleaning Function
def clean_data(df):
    # Convert percentage strings to numeric values
    percentage_cols = ['HBA1C', 'FBS', 'Cholesterol', 'Triglycerides', 'HDLCholesterol', 'LDLCholesterol']
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '').astype(float)

    # Convert other numeric columns
    numeric_cols = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    return df


# Clean the data
data_clean = clean_data(data.copy())

# Insert image
try:
    st.image(IMAGE_PATH, width=200)
except FileNotFoundError:
    st.warning("Image not found. Skipping display.")


# Page 1: Dataset Explorer
def page1():
    st.title("Dataset Explorer")

    if st.checkbox("Show Raw Diabetes Dataset"):
        st.write(data)

    if st.checkbox("Show Cleaned Diabetes Dataset"):
        st.write(data_clean)

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        try:
            uploaded_data = clean_data(pd.read_csv(uploaded_file))
            st.write("Uploaded and cleaned data:")
            st.write(uploaded_data)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")


# Page 2: Classification Analysis (Logistic Regression)
def page2():
    st.title("Diabetes Classification (Logistic Regression)")

    # Model features and target
    features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'HBA1C', 'FBS',
                'Cholesterol', 'Triglycerides', 'HDLCholesterol', 'LDLCholesterol']
    target = 'target'

    # Ensure we're using cleaned data
    df = data_clean.copy()

    # Ensure binary classification (convert if necessary)
    if df[target].nunique() > 2:
        st.warning("Target variable has more than 2 classes. Converting to binary by thresholding at median.")
        df[target] = (df[target] > df[target].median()).astype(int)

    # Train-test split (90:10)
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df[target],
        test_size=0.1,
        random_state=42,
        stratify=df[target]
    )

    # Create and train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediction Interface with medically valid ranges
    st.header("Make a Prediction")
    st.write("Enter patient details to assess diabetes risk:")

    # Define medically valid ranges for each feature
    feature_ranges = {
        'Age': {'min': 0, 'max': 120, 'step': 1, 'default': 30, 'unit': 'years'},
        'BMI': {'min': 10.0, 'max': 50.0, 'step': 0.1, 'default': 25.0, 'unit': 'kg/m²'},
        'SystolicBP': {'min': 70, 'max': 250, 'step': 1, 'default': 120, 'unit': 'mmHg'},
        'DiastolicBP': {'min': 40, 'max': 150, 'step': 1, 'default': 80, 'unit': 'mmHg'},
        'HBA1C': {'min': 3.0, 'max': 15.0, 'step': 0.1, 'default': 5.5, 'unit': '%'},
        'FBS': {'min': 50, 'max': 300, 'step': 1, 'default': 90, 'unit': 'mg/dL'},
        'Cholesterol': {'min': 100, 'max': 400, 'step': 1, 'default': 200, 'unit': 'mg/dL'},
        'Triglycerides': {'min': 50, 'max': 500, 'step': 1, 'default': 150, 'unit': 'mg/dL'},
        'HDLCholesterol': {'min': 20, 'max': 100, 'step': 1, 'default': 50, 'unit': 'mg/dL'},
        'LDLCholesterol': {'min': 50, 'max': 300, 'step': 1, 'default': 100, 'unit': 'mg/dL'}
    }

    # Display the valid ranges for all features
    st.subheader("Valid Ranges for Input Features")
    ranges_df = pd.DataFrame.from_dict(feature_ranges, orient='index')
    ranges_df = ranges_df[['min', 'max', 'unit']]
    ranges_df.columns = ['Minimum Value', 'Maximum Value', 'Unit']
    st.table(ranges_df)

    input_values = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            params = feature_ranges[feature]
            input_val = st.number_input(
                f"{feature} ({params['unit']})",
                min_value=params['min'],
                max_value=params['max'],
                value=params['default'],
                step=params['step'],
                format="%d" if params['step'] == 1 else "%.1f"
            )
            input_values[feature] = input_val

    if st.button('Predict Diabetes Risk', help="Click to make prediction"):
        try:
            new_data = pd.DataFrame([input_values])
            probability = model.predict_proba(new_data)[0][1]
            prediction = model.predict(new_data)[0]

            st.success(f"""
            **Prediction Results:**
            - Probability of Diabetes: {probability:.1%}
            - Classification: {'**Diabetic**' if prediction == 1 else '**Non-Diabetic**'}
            """)

            # Show probability gauge
            fig, ax = plt.subplots(figsize=(6, 0.5))
            ax.barh([0], [probability], color='salmon')
            ax.barh([0], [1 - probability], left=[probability], color='lightgreen')
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_title('Diabetes Risk Probability')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    # Model Evaluation
    st.header("Model Evaluation")

    col1, col2 = st.columns(2)
    with col1:
        train_acc = accuracy_score(y_train, model.predict(X_train))
        st.metric("Training Accuracy", f"{train_acc:.2%}")
    with col2:
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        st.metric("Test Accuracy", f"{test_acc:.2%}")

    # Classification report
    if st.checkbox("Show Classification Report"):
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.json(report)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    st.pyplot(fig)


# Page Navigation
pages = {
    'Dataset Explorer': page1,
    'Diabetes Classifier': page2
}

selected_page = st.sidebar.selectbox("Navigate", list(pages.keys()))
pages[selected_page]()