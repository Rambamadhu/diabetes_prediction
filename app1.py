import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model_path = 'svm_diabetes_model.pkl'  # Ensure this file is in the same directory
scaler_path = 'diabetes_scaler.pkl'

svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# App title and introduction
st.set_page_config(page_title="Diabetes Prediction System", layout="centered", initial_sidebar_state="expanded")
st.title("🔍 Diabetes Prediction System")
st.write("""
Welcome to the **Diabetes Prediction App**! This app uses a **Support Vector Machine (SVM)** model to predict the likelihood of diabetes based on medical diagnostic measurements.  
Simply enter the details below and click **Predict** to get your result.
""")

# Sidebar for user input
st.sidebar.header("🔢 Input Patient Data")
st.sidebar.write("Enter the following health parameters:")

# Input fields for user data (sidebar)
pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, step=1, help="Number of times the patient has been pregnant")
glucose = st.sidebar.number_input("Glucose Level (mg/dL)", min_value=0.0, help="Plasma glucose concentration")
blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0.0, help="Diastolic blood pressure")
skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0.0, help="Triceps skin fold thickness")
insulin = st.sidebar.number_input("Insulin Level (IU/mL)", min_value=0.0, help="2-hour serum insulin")
bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f", help="Weight in kg/(Height in m)^2")
dpf = st.sidebar.number_input("Diabetes Pedigree Function (DPF)", min_value=0.0, format="%.4f", help="Diabetes hereditary influence score")
age = st.sidebar.number_input("Age (years)", min_value=0, step=1, help="Age of the patient")

# Button to predict
if st.sidebar.button("Predict"):
    # Collect input into a single array
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the user input data
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction
    prediction = svm_model.predict(user_data_scaled)
    prediction_label = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    # Display result with formatting
    st.markdown("## 🩺 Prediction Result")
    if prediction_label == "Diabetic":
        st.error(f"⚠️ The patient is predicted to be **{prediction_label}**.")
    else:
        st.success(f"✅ The patient is predicted to be **{prediction_label}**.")
        
    # Display user input summary
    st.write("### Patient Details:")
    st.table({
        "Parameter": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "DPF", "Age"],
        "Value": [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    })

# Footer
st.write("---")
st.write("Developed with ❤️ using **SVM** and **Streamlit**. Want to learn more about diabetes? Visit [WHO Diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes).")
