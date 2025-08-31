import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model

# ===== Load model =====
try:
    model = joblib.load("xgb_student_model.pkl")
except:
    model = RandomForestRegressor()  # default placeholder

# ===== Sidebar Login =====
st.sidebar.title("Login / User")
user_type = st.sidebar.radio("Select User Type", ["Guest", "Admin"])

if user_type == "Admin":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        if username == "admin" and password == "admin123":  # Change to secure credentials
            st.sidebar.success("Logged in as Admin")
            admin_logged_in = True
        else:
            st.sidebar.error("Invalid Credentials")
            admin_logged_in = False
    else:
        admin_logged_in = False
else:
    admin_logged_in = False

# ===== Main App =====
st.title("Student Result Prediction App")

# ===== Input Section =====
st.header("Enter Student Details")
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=20.0, step=0.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_score = st.number_input("Previous Score", min_value=0, max_value=100, step=1)
assignment_score = st.number_input("Assignment Score", min_value=0, max_value=100, step=1)
writing_skills = st.slider("Writing Skills (0-10)", 0, 10, 5)
reading_skills = st.slider("Reading Skills (0-10)", 0, 10, 5)
computer_skills = st.slider("Computer Skills (0-10)", 0, 10, 5)

# ===== Predict Button =====
if st.button("Predict Result"):
    input_data = [[study_hours, attendance, previous_score, assignment_score, 
                   writing_skills, reading_skills, computer_skills]]
    
    percent_score = model.predict(input_data)[0]
    
    # Determine Grade
    if percent_score >= 90:
        grade = "A+"
    elif percent_score >= 80:
        grade = "A"
    elif percent_score >= 70:
        grade = "B+"
    elif percent_score >= 60:
        grade = "B"
    elif percent_score >= 50:
        grade = "C"
    else:
        grade = "D"
    
    # Determine Pass/Fail
    status = "Pass" if percent_score >= 40 else "Fail"
    
    # Display Results
    st.success(f"Predicted Percent Score: {percent_score:.2f}%")
    st.info(f"Grade: {grade}")
    st.warning(f"Status: {status}")

# ===== Admin Retrain Feature =====
if admin_logged_in:
    st.header("Admin: Retrain Model")
    uploaded_file = st.file_uploader("Upload New Dataset (CSV) for Retraining", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview", data.head())
        
        # Assume last column is target (PercentScore)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        joblib.dump(model, "student_result_model.pkl")
        st.success("Model retrained and saved successfully!")

