import streamlit as st
import pandas as pd
import joblib

# ===== Load your model =====
model = joblib.load("xgb_student_model.pkl")  # replace with your model file

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

# ===== Predict =====
if st.button("Predict Result"):
    input_data = [[study_hours, attendance, previous_score, assignment_score, 
                   writing_skills, reading_skills, computer_skills]]
    
    percent_score = model.predict(input_data)[0]  # Assuming model predicts PercentScore
    
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

    # Optional: create dataframe to download
    result_df = pd.DataFrame({
        "StudyHours":[study_hours],
        "Attendance":[attendance],
        "PreviousScore":[previous_score],
        "AssignmentScore":[assignment_score],
        "WritingSkills":[writing_skills],
        "ReadingSkills":[reading_skills],
        "ComputerSkills":[computer_skills],
        "PercentScore":[percent_score],
        "Grade":[grade],
        "Status":[status]
    })
    
    st.download_button("Download Result as CSV", result_df.to_csv(index=False), file_name="student_result.csv")
