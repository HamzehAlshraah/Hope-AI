import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
Logistic_r=joblib.load("HopeAI/logistic_regression.pkl")
random_forest=joblib.load(r"random_forset.pkl")
decision_tree=joblib.load(r"decision_tree_classifier.pkl")
svm=joblib.load(r"support_vector_classifier.pkl")
encoders = joblib.load("label_encoders.pkl")
st.title("Smart System for Academic Mental Health Monitoring")

model_option=st.sidebar.radio("Chosse the Model",["Logistic Regression","Random Forest","Decision Tree","Support Vector Classifier","Metrics"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=10, max_value=100, step=1)
Profession = st.selectbox("Profession", [
    "Student", "Employee", "Unemployed", "Freelancer", "Other"
])
Academic_Pressure = st.slider("Academic Pressure (1-10)", 1, 10, 5)
Work_Pressure = st.slider("Work Pressure (1-10)", 1, 10, 5)
CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
Study_Satisfaction = st.slider("Study Satisfaction (1-10)", 1, 10, 5)
Job_Satisfaction = st.slider("Job Satisfaction (1-10)", 1, 10, 5)
Sleep_Duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
Dietary_Habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])
Degree = st.selectbox("Degree", ["High School", "Bachelor", "Master", "PhD", "Other"])
Suicidal_Thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
Work_Study_Hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=24.0, step=0.5)
Financial_Stress = st.number_input(
    "Financial Stress (1–5)",min_value=1.0,max_value=5.0,step=1.0)

Family_History = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

input_data = pd.DataFrame({
                                "Gender": [Gender],
                                "Age": [Age],
                                "Profession": [Profession],
                                "Academic Pressure": [Academic_Pressure],
                                "Work Pressure": [Work_Pressure],
                                "CGPA": [CGPA],
                                "Study Satisfaction": [Study_Satisfaction],
                                "Job Satisfaction": [Job_Satisfaction],
                                "Sleep Duration": [Sleep_Duration],
                                "Dietary Habits": [Dietary_Habits],
                                "Degree": [Degree],
                                "Have you ever had suicidal thoughts ?": [Suicidal_Thoughts],
                                "Work/Study Hours": [Work_Study_Hours],
                                "Financial Stress": [Financial_Stress],
                                "Family History of Mental Illness": [Family_History],})


for col, le in encoders.items():
    if col in input_data.columns:
        val = input_data.at[0, col]

        # handle unseen labels safely
        if val not in le.classes_:
            input_data.at[0, col] = le.transform([le.classes_[0]])[0]  # default
        else:
            input_data.at[0, col] = le.transform([val])[0]

if model_option=="Logistic Regression":
    if st.button("Predict Depression lg"):
        prediction_lg = Logistic_r.predict(input_data)
        if prediction_lg[0] == 1 :    
            st.success(f"The predicted depression status is: depressed")
        else :
            st.success(f"The predicted depression status is: not depressed")
elif model_option =="Random Forest":
    if st.button("Predict Depression rf"):
        prediction_rf = random_forest.predict(input_data)
        if prediction_rf[0] == 1 :    
            st.success(f"The predicted depression status is: depressed")
        else :
            st.success(f"The predicted depression status is: not depressed")
elif model_option=="Decision Tree":
    if st.button("Predict Depression dt"):
         prediction_dt = decision_tree.predict(input_data)
         if prediction_dt[0] == 1 :
             st.success(f"The predicted depression status is: depressed")
         else :
             st.success(f"The predicted depression status is: not depressed")
elif model_option=="Support Vector Classifier":
    if st.button("Predict Depression svm"):
         prediction_svm = svm.predict(input_data)
         if prediction_svm[0] == 1 :
             st.success(f"The predicted depression status is: depressed")
         else :
             st.success(f"The predicted depression status is: not depressed")
elif model_option=="Metrics":
    if st.button("Show the Metrics"):
        st.table(pd.read_csv(r"C:\Users\user\Desktop\SDK\Project\HopeAI\Accuracy.csv"))
        st.table(pd.read_csv(r"C:\Users\user\Desktop\SDK\Project\HopeAI\classification_report.csv"))
        st.table(pd.read_csv(r"C:\Users\user\Desktop\SDK\Project\HopeAI\confusion_matrix.csv"))




