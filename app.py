
import streamlit as st
from predict import load_model, predict_diabetes

# Configure the webpage
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load the saved machine learning model
@st.cache_resource
def get_model():
    return load_model()


model, features = get_model()

# Page heading
st.title("Diabetes Risk Prediction")
st.write(
    "This application uses machine learning to classify "
    "diabetes risk based on health information."
)

st.info(
    "Academic demonstration only. This tool does not "
    "provide a medical diagnosis."
)

st.divider()
st.subheader("Enter Your Health Information")

# Binary health indicators
binary_features = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Checked in Past 5 Years",
    "Smoker": "Smoked at Least 100 Cigarettes",
    "Stroke": "History of Stroke",
    "HeartDiseaseorAttack": "Heart Disease or Heart Attack",
    "PhysActivity": "Physical Activity in Past 30 Days",
    "Fruits": "Eat Fruit at Least Once Daily",
    "Veggies": "Eat Vegetables at Least Once Daily",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Have Health Coverage",
    "NoDocbcCost": "Unable to See Doctor Due to Cost",
    "DiffWalk": "Difficulty Walking or Climbing Stairs"
}

health_data = {}

# Create the form
with st.form("diabetes_prediction_form"):

    for feature in features:

        # Binary features: 0 = No, 1 = Yes
        if feature in binary_features:
            health_data[feature] = st.selectbox(
                binary_features[feature],
                options=[0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No"
            )

        # BMI
        elif feature == "BMI":
            health_data[feature] = st.number_input(
                "Body Mass Index (BMI)",
                min_value=10.0,
                max_value=80.0,
                value=25.0,
                step=0.1
            )

        # Age category
        elif feature == "Age":
            age_categories = {
                1: "18–24",
                2: "25–29",
                3: "30–34",
                4: "35–39",
                5: "40–44",
                6: "45–49",
                7: "50–54",
                8: "55–59",
                9: "60–64",
                10: "65–69",
                11: "70–74",
                12: "75–79",
                13: "80 or older"
            }

            health_data[feature] = st.selectbox(
                "Age",
                options=list(age_categories.keys()),
                format_func=lambda value: age_categories[value]
            )

        # Sex
        elif feature == "Sex":
            health_data[feature] = st.selectbox(
                "Sex",
                options=[0, 1],
                format_func=lambda value: (
                    "Female" if value == 0 else "Male"
                )
            )

        # General health rating
        elif feature == "GenHlth":
            health_data[feature] = st.selectbox(
                "General Health",
                options=[1, 2, 3, 4, 5],
                format_func=lambda value: {
                    1: "Excellent",
                    2: "Very Good",
                    3: "Good",
                    4: "Fair",
                    5: "Poor"
                }[value]
            )

        # Mental health days
        elif feature == "MentHlth":
            health_data[feature] = st.slider(
                "Days of Poor Mental Health (Past 30 Days)",
                min_value=0,
                max_value=30,
                value=0
            )

        # Physical health days
        elif feature == "PhysHlth":
            health_data[feature] = st.slider(
                "Days of Poor Physical Health (Past 30 Days)",
                min_value=0,
                max_value=30,
                value=0
            )

        # Education level
        elif feature == "Education":
            education_levels = {
                1: "Never Attended School / Kindergarten",
                2: "Elementary School",
                3: "Some High School",
                4: "High School Graduate",
                5: "Some College",
                6: "College Graduate"
            }

            health_data[feature] = st.selectbox(
                "Education Level",
                options=list(education_levels.keys()),
                format_func=lambda value: education_levels[value]
            )

        # Income category
        elif feature == "Income":
            health_data[feature] = st.selectbox(
                "Annual Household Income Category",
                options=list(range(1, 9)),
                format_func=lambda value: {
                    1: "Less than $10,000",
                    2: "$10,000–$14,999",
                    3: "$15,000–$19,999",
                    4: "$20,000–$24,999",
                    5: "$25,000–$34,999",
                    6: "$35,000–$49,999",
                    7: "$50,000–$74,999",
                    8: "$75,000 or more"
                }[value]
            )

        else:
            # Fallback for unexpected feature names
            health_data[feature] = st.number_input(
                feature,
                value=0.0
            )

    # Submit button
    submitted = st.form_submit_button(
        "Predict Diabetes Risk",
        use_container_width=True
    )

# Generate prediction after form submission
if submitted:

    result = predict_diabetes(health_data)

    st.divider()
    st.subheader("Prediction Results")

    if result["prediction"] == 1:
        st.warning(
            "The model predicts a higher diabetes risk classification."
        )
    else:
        st.success(
            "The model predicts a lower diabetes risk classification."
        )

    # Display positive-class model score
    if "model_score" in result:
        score = result["model_score"] * 100

        st.metric(
            label="Positive-Class Model Score",
            value=f"{score:.2f}%"
        )

    st.caption(
        "Results are based on a machine learning model "
        "trained using CDC health indicators. The model "
        "score is not a medically validated probability "
        "of diabetes."
    )
