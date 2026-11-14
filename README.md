
# Medical Disease Risk Prediction Model

This project uses machine learning to predict diabetes risk based on a person's health information. We trained and compared three different models using the CDC Diabetes Health Indicators dataset.

## Dataset

We used the CDC Diabetes Health Indicators dataset, which contains 253,680 records and 21 health-related features, including BMI, age, blood pressure, cholesterol, and physical activity.

The dataset was split into training and testing sets, with an additional validation split used to compare models.

## Models

We trained and evaluated three machine learning models:

- Logistic Regression
- Random Forest
- Gradient Boosting

Random Forest performed the best based on validation F1-score (0.4526), so we selected it as our final model.

## Running the Project

1. Clone the repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Train and save the model:

```bash
python -m src.train
```

4. Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application allows users to enter their health information and receive a diabetes risk classification and model score.

## Note

This project was created for educational purposes. Predictions are based on machine learning and should not be used as a medical diagnosis.
