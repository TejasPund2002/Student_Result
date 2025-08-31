# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import plotly.io as pio
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
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
        try:
            return fallback_generic_plot_bytes(pred_value, dataset_series)
        except:
            buf = BytesIO()
            plt.figure(figsize=(4,2))
            plt.text(0.5,0.5,"Image Unavailable", ha='center')
            plt.axis('off')
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.read()

def fallback_generic_plot_bytes(pred_value=None, dataset_series=None):
    buf = BytesIO()
    if dataset_series is not None and len(dataset_series) > 0:
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
    if days_left <= 0: days_left = 1
    weeks = max(1, (days_left + 6)//7)
    diff = max(0, target - predicted)
    plan = []
    sh = current_vals['StudyHours']
    asg = current_vals['AssignmentScore']
    ws = current_vals['WritingSkills']
    rs = current_vals['ReadingSkills']
    cs = current_vals['ComputerSkills']
    hours_each_week = max(0, diff/2) / weeks
    assignment_each_week = max(0, diff*0.4) / weeks
    skill_each_week_value = min(1.0, diff/20) / weeks
    for w in range(1, weeks+1):
        plan.append({
            'Week': w,
            'StudyHours_target': round(sh + hours_each_week * w,1),
            'Assignment_target': round(min(100, asg + assignment_each_week * w),1),
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
        pdf.multi_cell(0,8,f"{key}: {val}")
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
        if start_date:
            ev_date = start_date + relativedelta(weeks=week_no-1)
        else:
            ev_date = datetime.now().date() + relativedelta(weeks=week_no-1)
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

# Language toggle
lang = st.radio("Language / भाषा:", ("Marathi", "English"))
M = {
    "Predict": "भविष्यवाणी करा" if lang=="Marathi" else "Predict",
    "Upload_CSV": "CSV अपलोड करा" if lang=="Marathi" else "Upload CSV",
    "Admin_Retrain": "Admin - Retrain" if lang=="Marathi" else "Admin - Retrain",
    "Model_Compare": "Model तुलना" if lang=="Marathi" else "Model Comparison"
}

# ---------- Simple user auth ----------
if os.path.exists(USERS_FILE):
    with open(USERS_FILE,"r") as f: users = json.load(f)
else:
    users = {"admin":{"password":"admin123","role":"admin"}}
    with open(USERS_FILE,"w") as f: json.dump(users,f)

st.sidebar.header("User / Login (Demo)")
username = st.sidebar.text_input("Username", value="guest")
password = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Login")
user_role = "guest"
if login_btn:
    if username in users and users[username]["password"] == password:
        st.sidebar.success(f"Logged in as {username}")
        user_role = users[username].get("role","user")
    else:
        st.sidebar.error("Invalid credentials. Using guest mode.")
        user_role = "guest"

# ---------- Main UI ----------
# ... [keep all sections exactly as in your original code] ...
# This includes: Student Input & Prediction, What-If, Guidance plan, PDF & ICS download,
# Visualization & History, Batch Predictions with CSV/PDF download, Admin retrain & comparison
# All the plotting, PDF generation, batch handling fixed as above

# ---------- End of App ----------
st.markdown("---")
st.info("End of App. Created By Shekhar Shelke Contact-shekharshelke45@gmail.com")
