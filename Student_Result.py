import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# ===== Use Wide Layout =====
st.set_page_config(page_title="Student Result Prediction", layout="wide")

# ===== Load model =====
try:
    model = joblib.load("xgb_student_model.pkl")
except:
    model = RandomForestRegressor()

# ===== Sidebar Login =====
st.sidebar.title("Login / User")
user_type = st.sidebar.radio("Select User Type", ["Guest", "Admin"])
admin_logged_in = False

if user_type == "Admin":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "admin" and password == "admin123":
            st.sidebar.success("Logged in as Admin")
            admin_logged_in = True
        else:
            st.sidebar.error("Invalid Credentials")

# ===== App Title =====
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>Student Result Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# ===== Input Section =====
with st.container():
    st.subheader("Enter Student Details")
    
    left_col, right_col = st.columns(2)

    with left_col:
        study_hours = st.number_input("Study Hours", min_value=0.0, max_value=20.0, step=0.5)
        previous_score = st.number_input("Previous Score", min_value=0, max_value=100, step=1)
        writing_skills = st.slider("Writing Skills (0-10)", 0, 10, 5)
        computer_skills = st.slider("Computer Skills (0-10)", 0, 10, 5)

    with right_col:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        assignment_score = st.number_input("Assignment Score", min_value=0, max_value=100, step=1)
        reading_skills = st.slider("Reading Skills (0-10)", 0, 10, 5)

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
    
    status = "Pass" if percent_score >= 40 else "Fail"
    
    # ===== Display Results in Dynamic Columns =====
    st.markdown("### Prediction Result")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Percent Score", f"{percent_score:.2f}%")
    col2.metric("Grade", grade)
    col3.metric("Status", status)

    # ===== Skills Bar Chart =====
    skills = pd.DataFrame({
        "Skill": ["Writing", "Reading", "Computer"],
        "Score": [writing_skills, reading_skills, computer_skills]
    })
    fig = px.bar(skills, x="Skill", y="Score", range_y=[0,10], color="Score", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

# ===== Admin Retrain Feature =====
if admin_logged_in:
    with st.expander("Admin: Retrain Model 🔧"):
        uploaded_file = st.file_uploader("Upload New Dataset (CSV) for Retraining", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview", data.head())
            
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            
            joblib.dump(model, "xgb_student_model.pkl")
            st.success("Model retrained and saved successfully!")
