# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import plotly.io as pio
import os
import io
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
from fpdf import FPDF
from ics import Calendar, Event
import tempfile

# ---------- Config ----------
DATA_PATH = "StudentDataset_100k.csv"
XGB_MODEL_FILE = "xgb_student_model.pkl"
HISTORY_FILE = "prediction_history.csv"
USERS_FILE = "users_local.json"

# ---------- Helper functions ----------
def fig_to_png_bytes(fig, pred_value=None, dataset_series=None):
    try:
        return pio.to_image(fig, format="png")
    except:
        return fallback_generic_plot_bytes(pred_value, dataset_series)

def fallback_generic_plot_bytes(pred_value=None, dataset_series=None):
    buf = BytesIO()
    if dataset_series is not None and len(dataset_series)>0:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(dataset_series.dropna(), bins=40)
        if pred_value is not None:
            ax.axvline(pred_value, color='r', linestyle='--', label='Predicted')
            ax.legend()
        ax.set_xlabel("PercentScore")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6,1.4))
        value = float(pred_value) if pred_value is not None else 0.0
        ax.barh([0], [value], height=0.6)
        ax.set_xlim(0,100)
        ax.set_yticks([])
        ax.set_xlabel(f"Predicted: {value:.2f}%")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
    buf.seek(0)
    return buf.read()

def load_or_train_xgb():
    if os.path.exists(XGB_MODEL_FILE):
        model = joblib.load(XGB_MODEL_FILE)
        return model, None
    if not os.path.exists(DATA_PATH):
        st.warning("Model file not found and dataset not present to train. Please upload dataset in Admin > Retrain.")
        return None, None
    df = pd.read_csv(DATA_PATH)
    X = df[['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']]
    y = df['PercentScore']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, objective='reg:squarederror')
    model.fit(X_train, y_train)
    joblib.dump(model, XGB_MODEL_FILE)
    return model, (X_test, y_test)

def predict_percent(model, input_df):
    preds = model.predict(input_df)
    preds = np.clip(preds, 0, 100)
    return preds

def grade_from_score(score):
    if score >= 90: return 'A+'
    elif score >= 80: return 'A'
    elif score >= 70: return 'B+'
    elif score >= 60: return 'B'
    elif score >= 50: return 'C'
    elif score >= 40: return 'D'
    else: return 'F'

def passfail_from_score(score):
    return "Pass" if score >= 40 else "Fail"

def save_history(row_dict):
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
        df_new.to_csv(HISTORY_FILE, index=False)
    else:
        df_row.to_csv(HISTORY_FILE, index=False)

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    else:
        return pd.DataFrame(columns=['timestamp','user','StudyHours','Attendance','PreviousScore',
                                     'AssignmentScore','WritingSkills','ReadingSkills',
                                     'ComputerSkills','PredictedPercent','Grade','PassFail','TargetScore','ExamDate'])

def generate_weekly_plan(current_vals, predicted, target, exam_date):
    days_left = (exam_date - datetime.now().date()).days
    days_left = max(1, days_left)
    weeks = max(1, (days_left + 6)//7)
    diff = max(0, target - predicted)
    plan = []
    sh = current_vals['StudyHours']
    asg = current_vals['AssignmentScore']
    ws = current_vals['WritingSkills']
    rs = current_vals['ReadingSkills']
    cs = current_vals['ComputerSkills']
    needed_hours_inc = max(0, diff/2)
    hours_each_week = needed_hours_inc / weeks
    needed_assignment_inc = max(0, diff*0.4)
    assignment_each_week = needed_assignment_inc / weeks
    skill_each_week = min(1.0, diff/20)
    skill_each_week_value = skill_each_week / weeks
    for w in range(1, weeks+1):
        plan.append({
            'Week': w,
            'StudyHours_target': round(sh + hours_each_week * w, 1),
            'Assignment_target': round(min(100, asg + assignment_each_week * w), 1),
            'Writing_target': min(10, round(ws + skill_each_week_value * w,1)),
            'Reading_target': min(10, round(rs + skill_each_week_value * w,1)),
            'Computer_target': min(10, round(cs + 0.1 * w,1)),
        })
    return pd.DataFrame(plan)

def create_pdf_report(details, charts_bytes_list):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for key, val in details.items():
        pdf.multi_cell(0, 8, f"{key}: {val}")
    pdf.ln(5)
    for chart_bytes in charts_bytes_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(chart_bytes)
            tmpfile.flush()
            pdf.add_page()
            pdf.image(tmpfile.name, x=10, y=25, w=180)
    return pdf.output(dest="S").encode("latin1")

def create_ics_plan(plan_df, student_name="student", start_date=None):
    cal = Calendar()
    for _, row in plan_df.iterrows():
        week_no = int(row['Week'])
        ev_date = start_date + relativedelta(weeks=week_no-1) if start_date else datetime.now().date() + relativedelta(weeks=week_no-1)
        e = Event()
        e.name = f"Week {week_no} Study Plan"
        e.begin = datetime.combine(ev_date, datetime.min.time())
        e.duration = {"days":7}
        e.description = f"Targets: StudyHours={row['StudyHours_target']}, Assignment={row['Assignment_target']}, Writing={row['Writing_target']}, Reading={row['Reading_target']}"
        cal.events.add(e)
    return str(cal)

# ---------- App UI ----------
st.set_page_config(page_title="Shoppers - Student Predictor", layout="wide")
st.title("📚 Student Performance Predictor (Marathi/English)")

# language toggle
lang = st.radio("Language / भाषा:", ("Marathi", "English"))
M = {
    "Predict": "भविष्यवाणी करा" if lang=="Marathi" else "Predict",
    "Upload_CSV": "CSV अपलोड करा" if lang=="Marathi" else "Upload CSV",
    "Admin_Retrain": "Admin - Retrain" if lang=="Marathi" else "Admin - Retrain",
    "Model_Compare": "Model तुलना" if lang=="Marathi" else "Model Comparison"
}

# Simple local auth (demo)
st.sidebar.header("User / Login (Demo)")
if os.path.exists(USERS_FILE):
    with open(USERS_FILE,"r") as f:
        users = json.load(f)
else:
    users = {"admin":{"password":"admin123","role":"admin"}}
    with open(USERS_FILE,"w") as f:
        json.dump(users,f)
username = st.sidebar.text_input("Username", value="guest")
password = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Login")
user_role = "guest"
if login_btn:
    if username in users and users[username]["password"] == password:
        st.sidebar.success(f"Logged in as {username}")
        user_role = users[username].get("role","user")
    else:
        st.sidebar.error("Invalid credentials (demo users file). Using guest mode.")
        user_role = "guest"

# ---------- Student Input & Prediction ----------
with st.expander("Student Input & Prediction", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        study_hours = st.slider("Study Hours per Day:", 0.0, 12.0, 2.5, step=0.5)
        attendance = st.slider("Attendance (%):", 0, 100, 75)
        previous_score = st.slider("Previous Score:", 0, 100, 60)
    with col2:
        assignment_score = st.slider("Assignment Score (%):", 0, 100, 60)
        writing_skills = st.slider("Writing Skills (0-10):",0,10,5)
        reading_skills = st.slider("Reading Skills (0-10):",0,10,5)
    with col3:
        computer_skills = st.slider("Computer Skills (0-10):",0,10,5)
        target_score = st.slider("Target Score:", 0, 100, 80)
        exam_date = st.date_input("Exam Date:", value=datetime.now().date() + timedelta(days=30))

    if st.button(M["Predict"]):
        model, test_data = load_or_train_xgb()
        if model:
            input_df = pd.DataFrame([{
                "StudyHours": study_hours,
                "Attendance": attendance,
                "PreviousScore": previous_score,
                "AssignmentScore": assignment_score,
                "WritingSkills": writing_skills,
                "ReadingSkills": reading_skills,
                "ComputerSkills": computer_skills
            }])
            pred_percent = float(predict_percent(model, input_df)[0])
            grade = grade_from_score(pred_percent)
            pf = passfail_from_score(pred_percent)
            st.success(f"Predicted Percent: {pred_percent:.2f}% | Grade: {grade} | {pf}")
            history_row = {
                'timestamp': datetime.now().isoformat(),
                'user': username,
                'StudyHours': study_hours,
                'Attendance': attendance,
                'PreviousScore': previous_score,
                'AssignmentScore': assignment_score,
                'WritingSkills': writing_skills,
                'ReadingSkills': reading_skills,
                'ComputerSkills': computer_skills,
                'PredictedPercent': pred_percent,
                'Grade': grade,
                'PassFail': pf,
                'TargetScore': target_score,
                'ExamDate': exam_date.isoformat()
            }
            save_history(history_row)
            
            # Weekly plan
            current_vals = {"StudyHours":study_hours,"AssignmentScore":assignment_score,
                            "WritingSkills":writing_skills,"ReadingSkills":reading_skills,"ComputerSkills":computer_skills}
            plan_df = generate_weekly_plan(current_vals, pred_percent, target_score, exam_date)
            st.table(plan_df)

            # Generate ICS
            ics_file = create_ics_plan(plan_df, student_name=username, start_date=datetime.now().date())
            st.download_button("Download Weekly Plan (.ics)", ics_file, file_name="weekly_plan.ics")

            # PDF Report
            charts_bytes_list = []
            # Gauge
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=pred_percent, title={'text':"Predicted Percent"}, gauge={'axis':{'range':[0,100]}}))
            charts_bytes_list.append(fig_gauge.to_image(format="png"))
            pdf_bytes = create_pdf_report(history_row, charts_bytes_list)
            st.download_button("Download PDF Report", pdf_bytes, file_name="student_report.pdf", mime="application/pdf")

# ---------- Batch Prediction Section ----------
with st.expander("Batch Predictions (CSV Upload)", expanded=False):
    st.markdown("Upload a CSV containing columns: StudyHours, Attendance, PreviousScore, AssignmentScore, WritingSkills, ReadingSkills, ComputerSkills")
    uploaded = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
    
    if uploaded is not None:
        df_batch = pd.read_csv(uploaded)
        required_cols = ['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']
        
        if not all([c in df_batch.columns for c in required_cols]):
            st.error(f"CSV missing required columns. Ensure columns: {required_cols}")
        else:
            model, _ = load_or_train_xgb()
            if model:
                preds = predict_percent(model, df_batch[required_cols])
                df_batch['PredictedPercent'] = np.round(preds,2)
                df_batch['Grade'] = df_batch['PredictedPercent'].apply(grade_from_score)
                df_batch['PassFail'] = df_batch['PredictedPercent'].apply(passfail_from_score)
                st.dataframe(df_batch.head(200))

                csv_bytes = df_batch.to_csv(index=False).encode()
                st.download_button("Download Predictions CSV", csv_bytes, file_name="batch_predictions.csv", mime="text/csv")

                # PDF Example
                if st.button("Download Example Batch PDF"):
                    charts_bytes_list = []
                    for _, r in df_batch.head(1).iterrows():
                        details = {
                            "Student Index": r.name,
                            "PredictedPercent": r['PredictedPercent'],
                            "Grade": r['Grade'],
                            "PassFail": r['PassFail']
                        }
                        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=r['PredictedPercent'], title={'text': "Predicted Percent"}, gauge={'axis':{'range':[0,100]}}))
                        charts_bytes_list.append(fig_gauge.to_image(format="png"))
                        pdf_bytes_student = create_pdf_report(details, charts_bytes_list)
                        st.download_button("Download Example Batch PDF", pdf_bytes_student, file_name="batch_student.pdf", mime="application/pdf")

                # Save batch to history
                if st.button("Save Batch to History"):
                    for _, r in df_batch.iterrows():
                        row = {
                            'timestamp': datetime.now().isoformat(),
                            'user': username,
                            'StudyHours': r['StudyHours'],
                            'Attendance': r['Attendance'],
                            'PreviousScore': r['PreviousScore'],
                            'AssignmentScore': r['AssignmentScore'],
                            'WritingSkills': r['WritingSkills'],
                            'ReadingSkills': r['ReadingSkills'],
                            'ComputerSkills': r['ComputerSkills'],
                            'PredictedPercent': r['PredictedPercent'],
                            'Grade': r['Grade'],
                            'PassFail': r['PassFail'],
                            'TargetScore': '',
                            'ExamDate': ''
                        }
                        save_history(row)
                    st.success("Batch rows saved to history.")

# ---------- History Viewer ----------
with st.expander("Prediction History", expanded=False):
    hist_df = load_history()
    st.dataframe(hist_df.tail(100))
