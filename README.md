# Diabetes Prediction 

## Overview

This project predicts whether an individual is diabetic or not using a machine learning model. It utilizes the PIMA Diabetes dataset and implements a Support Vector Machine (SVM) classifier to make predictions.

## Dataset

The dataset contains 768 entries with the following features:
- `Pregnancies`: Number of times pregnant.
- `Glucose`: Plasma glucose concentration.
- `BloodPressure`: Diastolic blood pressure (mm Hg).
- `SkinThickness`: Triceps skinfold thickness (mm).
- `Insulin`: 2-Hour serum insulin (mu U/ml).
- `BMI`: Body mass index (weight in kg/(height in m)^2).
- `DiabetesPedigreeFunction`: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history).
- `Age`: Age of the person.
- `Outcome`: Target variable (0 - Non-Diabetic, 1 - Diabetic).

## Project Workflow

1. **Data Collection and Analysis**:
   - The dataset is loaded and statistical summaries are computed.
   - Outcome distribution and feature relationships are analyzed.

2. **Data Preprocessing**:
   - StandardScaler is used to standardize the dataset for better model performance.

3. **Train-Test Split**:
   - The dataset is split into training (80%) and testing (20%) sets.

4. **Model Training**:
   - A Support Vector Machine (SVM) classifier with a linear kernel is trained on the dataset.

5. **Model Evaluation**:
   - Accuracy scores for training and testing data are computed.

6. **Predictive System**:
   - A simple predictive system is created to predict diabetes status based on user inputs.

## Tools and Technologies

- Python
- Pandas, NumPy
- scikit-learn
- Support Vector Machine (SVM)

## Results

The model achieves high accuracy scores on both training and test datasets as 78% and 77% approximately, demonstrating its effectiveness in predicting diabetes.

## Future Scope

- Extend the predictive system to a web application using frameworks like Streamlit or Flask.
- Experiment with different machine learning models for better performance.

## Acknowledgments

This project uses the PIMA Diabetes dataset, which is publicly available for research purposes.
