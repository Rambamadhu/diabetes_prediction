import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model_path = 'svm_diabetes_model.pkl'  # Adjust path if needed
scaler_path = 'diabetes_scaler.pkl'

svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Title and description
st.title("Diabetes Prediction System")
st.write("""
This app predicts whether a person is diabetic based on medical diagnostic measurements.
Provide the following details to get predictions:
""")

# Input fields for user data
pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0)
insulin = st.number_input("Insulin Level (IU/mL)", min_value=0.0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function (DPF)", min_value=0.0, format="%.4f")
age = st.number_input("Age (years)", min_value=0, step=1)

# Button to predict
if st.button("Predict"):
    # Collect input into a single array
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the user input data
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction
    prediction = svm_model.predict(user_data_scaled)
    prediction_label = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    # Display result
    st.write(f"The model predicts: **{prediction_label}**")

# Footer
st.write("\n---")
st.write("Developed using SVM and Streamlit")
