
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

## Page configuration
st.set_page_config(
    page_title="Hypertension Prediction | DataQuest Solutions",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

## Application directory and model path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best_gb_model.joblib")

## Load the complete model bundle
@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)

try:
    model_bundle = load_model_bundle()
    model = model_bundle["model"]
    encoders = model_bundle["encoders"]
    target_encoder = model_bundle["target_encoder"]
    scaler = model_bundle["scaler"]
    feature_columns = model_bundle["feature_columns"]

except FileNotFoundError:
    st.error(
        "The model file was not found. Please place "
        "'best_gb_model.joblib' in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load the model bundle: {error}")
    st.stop()

## DataQuest Solutions brand colors
NAVY = "#0d2853"
ORANGE = "#ff7a38"
WHITE = "#ffffff"
LIGHT_BG = "#f4f7fb"
DARK_TEXT = "#243247"
MUTED_TEXT = "#64748b"

## Custom application styling
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {LIGHT_BG};
    }}

    [data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}

    [data-testid="stSidebar"] * {{
        color: {WHITE};
    }}

    [data-testid="stSidebar"] .stSelectbox label {{
        color: {WHITE};
    }}

    [data-testid="stSidebar"] .stRadio label {{
        color: {WHITE};
    }}

    .main-header {{
       background: linear-gradient(115deg, #0d2853, #173b62);
       margin-top: 0rem !important;
       width:100%;
       padding: 2px 35px;
       border-radius: 0 0 16px 16px;
       color: #ffffff;
       margin-bottom: 25px;
       border-left: 7px solid #ff7a38;
       position: sticky;
       top: 0;
       z-index: 999;
       box-shadow: 0 4px 15px rgba(13, 40, 83, 0.15);
    }}

    .main-header h1 {{
        color: {WHITE};
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 8px 0;
    }}

    .main-header p {{
        color: #e6edf7;
        font-size: 14px;
        margin: 0;
    }}

    .section-heading {{
        color: {NAVY};
        font-size: 21px;
        font-weight: 750;
        margin-top: 12px;
        margin-bottom: 15px;
    }}

    .metric-card {{
        background: {WHITE};
        padding: 20px;
        border-radius: 13px;
        border: 1px solid #e5eaf1;
        box-shadow: 0 3px 12px rgba(13, 40, 83, 0.05);
        min-height: 112px;
    }}

    .metric-label {{
        color: {MUTED_TEXT};
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 9px;
    }}

    .metric-value {{
        color: {NAVY};
        font-size: 24px;
        font-weight: 800;
    }}

    .metric-description {{
        color: {MUTED_TEXT};
        font-size: 11px;
        margin-top: 5px;
    }}

    .result-card {{
        background: {WHITE};
        border-radius: 15px;
        padding: 25px;
        border: 1px solid #e5eaf1;
        box-shadow: 0 4px 15px rgba(13, 40, 83, 0.06);
        margin-top: 15px;
    }}

    .result-positive {{
        color: #b42318;
        font-size: 27px;
        font-weight: 800;
    }}

    .result-negative {{
        color: #15803d;
        font-size: 27px;
        font-weight: 800;
    }}

    .result-caption {{
        color: {MUTED_TEXT};
        font-size: 13px;
    }}

    .info-card {{
        background: {WHITE};
        padding: 20px;
        border-radius: 13px;
        border: 1px solid #e5eaf1;
        margin-bottom: 15px;
    }}

    .info-card h4 {{
        color: {NAVY};
        margin-top: 0;
    }}

    .info-card p {{
        color: {DARK_TEXT};
        font-size: 13px;
        line-height: 1.7;
    }}

    .stButton > button {{
        background-color: {ORANGE};
        color: {WHITE};
        border: none;
        border-radius: 8px;
        padding: 11px 22px;
        font-weight: 700;
        width: 100%;
    }}

    .stButton > button:hover {{
        background-color: #e96727;
        color: {WHITE};
        border: none;
    }}

    div[data-testid="stFormSubmitButton"] button {{
        background-color: {ORANGE};
        color: {WHITE};
        border: none;
        border-radius: 8px;
        font-weight: 700;
        width: 100%;
        min-height: 45px;
    }}

    div[data-testid="stFormSubmitButton"] button:hover {{
        background-color: #e96727;
        color: {WHITE};
    }}

    .footer {{
        text-align: center;
        color: #ffffff;
        font-size: 12px;
        padding: 3px 1px;
        margin-top: 35px;
        border-top: 1px solid #e1e7ef;
        background-color: #0d2853;
        position: sticky;
        bottom: 0;
        z-index: 998;
        box-shadow: 0 -4px 12px rgba(13, 40, 83, 0.06);
    }}

    .disclaimer {{
        background: #fff7ed;
        border-left: 5px solid {ORANGE};
        padding: 15px 18px;
        border-radius: 8px;
        color: #7c3f13;
        font-size: 12px;
        line-height: 1.7;
        margin-top: 20px;
    }}

    [data-testid="stMetric"] {{
        background: {WHITE};
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e5eaf1;
    }}

    /* DataQuest sidebar logo */
    [data-testid="stSidebar"] [data-testid="stImage"] {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto 10px auto;
        padding: 10px;
    }}

    /* Control logo dimensions */
    [data-testid="stSidebar"] [data-testid="stImage"] img {{
        width: 130px;
        height: 130px;
        object-fit: contain;
        border-radius: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


## Application navigation
with st.sidebar:

    ## Display DataQuest Solutions logo
    st.image(
        os.path.join(APP_DIR, "favicon.png"),
        width=130
    )

    ## DataQuest Solutions branding
    st.markdown(
        """
        <div style="text-align:center; padding:0px 5px 20px 5px;">
            <h2 style="color:white; margin:5px 0 0 0;">
                DataQuest Solutions
            </h2>
            <p style="color:#dce6f4; font-size:11px; margin-top:5px;">
                Health Intelligence & Analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    ## Application navigation menu
    page = st.radio(
        "NAVIGATION",
        [
            "Prediction Dashboard",
            "About the Model"
        ]
    )

    st.markdown("---")

    ## Application description
    st.markdown(
        """
        **About this application**

        This application uses a trained
        Gradient Boosting classification model
        to estimate hypertension status from
        selected demographic, lifestyle and
        health-related characteristics.
        """
    )

    st.markdown("---")

    ## Developer information
    st.caption("Developed by DataQuest Solutions")
    st.caption("Data Science | AI | Health Analytics")

## Main application header
st.markdown(
    """
    <div class="main-header">
        <h1>Hypertension Prediction System</h1>
        <p>
            Machine Learning-Based Hypertension Classification
            and Health Risk Assessment
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

## Helper function for categorical input options
def get_category_options(column):
    encoder = encoders[column]
    options = []

    for value in encoder.classes_:
        if pd.isna(value):
            options.append(("Not recorded", value))
        else:
            options.append((str(value), value))

    return options

## Helper function to obtain the selected encoded category
def encode_category(column, selected_label):
    options = get_category_options(column)

    for label, original_value in options:
        if label == selected_label:
            return encoders[column].transform([original_value])[0]

    raise ValueError(
        f"Selected category '{selected_label}' is not available "
        f"for {column}."
    )

## Prediction dashboard
if page == "Prediction Dashboard":

    ## Dashboard overview
    st.markdown(
        '<div class="section-heading">Prediction Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Prediction Model</div>
                <div class="metric-value">Gradient Boosting</div>
                <div class="metric-description">
                    Selected classification algorithm
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Input Variables</div>
                <div class="metric-value">10</div>
                <div class="metric-description">
                    Demographic, lifestyle and health variables
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Model Type</div>
                <div class="metric-value">Classification</div>
                <div class="metric-description">
                    Hypertension status prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Patient Assessment</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the individual's characteristics below to generate "
        "a hypertension classification prediction."
    )

    ## Input form
    with st.form("hypertension_prediction_form"):

        st.markdown("#### 1. Demographic Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=84,
                value=50,
                step=1
            )

        with col2:
            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=11.9,
                max_value=41.9,
                value=25.9,
                step=0.1,
                format="%.1f"
            )

        st.markdown("#### 2. Lifestyle and Behavioural Characteristics")

        col1, col2 = st.columns(2)

        with col1:
            salt_intake = st.number_input(
                "Salt Intake",
                min_value=2.5,
                max_value=16.4,
                value=8.5,
                step=0.1,
                format="%.1f",
                help="Enter salt intake using the same unit as the training dataset."
            )

            stress_score = st.slider(
                "Stress Score",
                min_value=0,
                max_value=10,
                value=5
            )

            sleep_duration = st.number_input(
                "Sleep Duration (hours)",
                min_value=1.5,
                max_value=11.4,
                value=6.5,
                step=0.1,
                format="%.1f"
            )

        with col2:
            bp_options = get_category_options("bp_history")
            bp_history = st.selectbox(
                "Blood Pressure History",
                options=[item[0] for item in bp_options]
            )

            exercise_options = get_category_options("exercise_level")
            exercise_level = st.selectbox(
                "Exercise Level",
                options=[item[0] for item in exercise_options]
            )

            smoking_options = get_category_options("smoking_status")
            smoking_status = st.selectbox(
                "Smoking Status",
                options=[item[0] for item in smoking_options]
            )

        st.markdown("#### 3. Medical and Family History")

        col1, col2 = st.columns(2)

        with col1:
            medication_options = get_category_options("medication")
            medication = st.selectbox(
                "Medication Status",
                options=[item[0] for item in medication_options],
                help="Select the medication category represented in the training data."
            )

            family_options = get_category_options("family_history")
            family_history = st.selectbox(
                "Family History of Hypertension",
                options=[item[0] for item in family_options]
            )

        with col2:
            st.markdown(
                """
                <div class="info-card">
                    <h4>Assessment Information</h4>
                    <p>
                        Ensure the information entered is accurate.
                        The prediction is based on the variables and
                        categories used during model development.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        submitted = st.form_submit_button(
            "Generate Hypertension Prediction"
        )

    ## Process the prediction
    if submitted:

        try:
            ## Prepare raw input values
            input_values = {
                "age": age,
                "salt_intake": salt_intake,
                "stress_score": stress_score,
                "bp_history": bp_history,
                "sleep_duration": sleep_duration,
                "bmi": bmi,
                "medication": medication,
                "family_history": family_history,
                "exercise_level": exercise_level,
                "smoking_status": smoking_status
            }

            ## Create input dataframe in the original feature order
            input_df = pd.DataFrame(
                [{column: input_values[column] for column in feature_columns}],
                columns=feature_columns
            )

            ## Apply the saved categorical encoders
            for column in encoders:
                selected_label = input_values[column]
                input_df[column] = encode_category(
                    column,
                    selected_label
                )

            ## Ensure numeric variables are numeric
            input_df = input_df.astype(float)

            ## Apply the saved MinMaxScaler
            scaled_input = scaler.transform(input_df)

            ## Generate prediction
            prediction_encoded = model.predict(scaled_input)[0]

            ## Decode the predicted target category
            prediction = target_encoder.inverse_transform(
                [prediction_encoded]
            )[0]

            ## Generate class probabilities where supported
            probabilities = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled_input)[0]

            ## Display prediction results
            st.markdown("---")

            st.markdown(
                '<div class="section-heading">Prediction Results</div>',
                unsafe_allow_html=True
            )

            prediction_text = str(prediction).strip().lower()

            if prediction_text == "yes":
                result_label = "Hypertension Predicted"
                result_class = "result-positive"
                result_description = (
                    "The model classified this assessment as "
                    "hypertension positive."
                )
            else:
                result_label = "No Hypertension Predicted"
                result_class = "result-negative"
                result_description = (
                    "The model classified this assessment as "
                    "hypertension negative."
                )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-caption">
                        MODEL CLASSIFICATION
                    </div>
                    <div class="{result_class}">
                        {result_label}
                    </div>
                    <p style="color:{DARK_TEXT}; font-size:14px;">
                        {result_description}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            ## Display probability information
            if probabilities is not None:
                st.markdown("#### Prediction Probability")

                probability_data = []

                for index, class_value in enumerate(
                    target_encoder.classes_
                ):
                    probability_data.append({
                        "Predicted Class": str(class_value),
                        "Probability": float(probabilities[index])
                    })

                probability_df = pd.DataFrame(probability_data)
                probability_df["Probability (%)"] = (
                    probability_df["Probability"] * 100
                ).round(2)

                for _, row in probability_df.iterrows():
                    st.write(
                        f"**{row['Predicted Class']}** — "
                        f"{row['Probability (%)']:.2f}%"
                    )
                    st.progress(float(row["Probability"]))

            ## Patient assessment summary
            st.markdown("#### Patient Assessment Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Age", f"{age} years")
                st.metric("BMI", f"{bmi:.1f}")

            with summary_col2:
                st.metric("Stress Score", f"{stress_score}/10")
                st.metric("Sleep Duration", f"{sleep_duration:.1f} hours")

            with summary_col3:
                st.metric("Salt Intake", f"{salt_intake:.1f}")
                st.metric("Exercise Level", exercise_level)

            ## Prepare downloadable assessment report
            report_data = {
                "Assessment Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Age": [age],
                "Salt Intake": [salt_intake],
                "Stress Score": [stress_score],
                "Blood Pressure History": [bp_history],
                "Sleep Duration": [sleep_duration],
                "BMI": [bmi],
                "Medication": [medication],
                "Family History": [family_history],
                "Exercise Level": [exercise_level],
                "Smoking Status": [smoking_status],
                "Predicted Hypertension Status": [str(prediction)]
            }

            if probabilities is not None:
                for index, class_value in enumerate(
                    target_encoder.classes_
                ):
                    report_data[
                        f"Probability_{class_value}"
                    ] = [round(float(probabilities[index]), 4)]

            report_df = pd.DataFrame(report_data)

            csv_report = report_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Assessment Report (CSV)",
                data=csv_report,
                file_name="hypertension_prediction_report.csv",
                mime="text/csv"
            )

            st.markdown(
                """
                <div class="disclaimer">
                    <strong>Important Clinical Disclaimer:</strong>
                    This application provides a machine learning-based
                    classification for research, educational and
                    decision-support purposes. It is not a medical
                    diagnosis, does not replace clinical blood pressure
                    measurement or professional assessment, and should
                    not be used independently to make treatment decisions.
                    Consult a qualified healthcare professional for
                    appropriate evaluation and management.
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as error:
            st.error(
                f"An error occurred while generating the prediction: {error}"
            )

    else:
        st.info(
            "Complete the patient assessment form and click "
            "'Generate Hypertension Prediction' to view the result."
        )

## About the model page
elif page == "About the Model":

    st.markdown(
        '<div class="section-heading">About the Prediction Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>Model Overview</h4>
            <p>
                The Hypertension Prediction System uses a Gradient
                Boosting Classifier developed to classify individuals
                according to their hypertension status.
            </p>
            <p>
                The model was selected following comparative evaluation
                and hyperparameter optimization of machine learning
                classification algorithms.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Model Information")

    model_info = pd.DataFrame({
        "Model Attribute": [
            "Algorithm",
            "Learning Task",
            "Target Variable",
            "Number of Predictors",
            "Training Dataset",
            "Cross-validation"
        ],
        "Description": [
            "Gradient Boosting Classifier",
            "Binary Classification",
            "has_hypertension",
            str(len(feature_columns)),
            "1,985 observations",
            "5-fold Stratified Cross-Validation"
        ]
    })

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Model Predictors")

    predictor_descriptions = {
        "age": "Age of the individual in years",
        "salt_intake": "Salt intake",
        "stress_score": "Stress score",
        "bp_history": "Blood pressure history",
        "sleep_duration": "Sleep duration in hours",
        "bmi": "Body Mass Index",
        "medication": "Medication category",
        "family_history": "Family history",
        "exercise_level": "Exercise level",
        "smoking_status": "Smoking status"
    }

    predictor_df = pd.DataFrame({
        "Predictor": feature_columns,
        "Description": [
            predictor_descriptions.get(
                column,
                "Predictor used during model training"
            )
            for column in feature_columns
        ]
    })

    st.dataframe(
        predictor_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Model Preprocessing")

    st.markdown(
        """
        The application applies the same preprocessing components
        saved during model development:

        1. **Label Encoding:** Converts categorical predictors into
           numerical representations using the fitted encoders.

        2. **MinMax Scaling:** Transforms the input features using the
           saved scaler fitted on the training data.

        3. **Gradient Boosting Classification:** Generates the
           hypertension status prediction.

        4. **Target Decoding:** Converts the predicted numerical class
           back to its original category.
        """
    )

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Research and Clinical Use:</strong>
            Model predictions should be interpreted in the context of
            the model's training data, validation performance and
            intended use. The system is not a substitute for clinical
            diagnosis or professional medical advice.
        </div>
        """,
        unsafe_allow_html=True
    )

## Application footer
st.markdown(
    f"""
    <div class="footer">
        <strong style="color:{NAVY};">
            DataQuest Solutions
        </strong>
        <br>
        Data Science | Artificial Intelligence | Health Analytics
        <br>
        Empowering Smarter Healthcare Decisions
        <br><br>
        © {datetime.now().year} DataQuest Solutions.
        All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True
)
