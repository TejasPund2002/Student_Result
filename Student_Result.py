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

# ---------- Config ----------
DATA_PATH = "StudentDataset_100k.csv"   # change if needed
XGB_MODEL_FILE ="xgb_student_model.pkl"# optional

HISTORY_FILE = "prediction_history.csv"
USERS_FILE = "users_local.json"   # simple local user store for demo

# ---------- Helper functions ----------
# ---- Plotly to PNG safe wrapper + matplotlib fallbacks ----
def fig_to_png_bytes(fig, pred_value=None, dataset_series=None):
    """
    Try Plotly/Kaleido conversion first; if it fails, fall back to matplotlib PNG bytes.
    Returns PNG bytes.
    """
    try:
        # prefer plotly.io conversion
        return pio.to_image(fig, format="png")
    except Exception as e:
        # fallback - create a simple matplotlib image depending on fig type
        # If it's a gauge (single value), create horizontal bar gauge-like image
        try:
            return fallback_generic_plot_bytes(pred_value, dataset_series)
        except Exception as e2:
            # Last resort: empty PNG
            buf = BytesIO()
            plt.figure(figsize=(4,2)); plt.text(0.5,0.5,"Image Unavailable", ha='center'); plt.axis('off')
            plt.savefig(buf, format='png', bbox_inches='tight'); plt.close()
            buf.seek(0)
            return buf.read()

def fallback_generic_plot_bytes(pred_value=None, dataset_series=None):
    """
    Create PNG bytes using matplotlib:
     - If dataset_series provided -> draw histogram and vertical line at pred_value.
     - Else if pred_value provided -> draw gauge-like horizontal bar.
    """
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
        # gauge-like bar
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
    # Try load existing model
    if os.path.exists(XGB_MODEL_FILE):
        model = joblib.load(XGB_MODEL_FILE)
        return model, None
    # else try to train quickly from DATA_PATH if exists
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
    # clip to 0-100
    preds = np.clip(preds, 0, 100)
    return preds

def grade_from_score(score):
    if score >= 90:
        return 'A+'
    elif score >= 80:
        return 'A'
    elif score >= 70:
        return 'B+'
    elif score >= 60:
        return 'B'
    elif score >= 50:
        return 'C'
    elif score >= 40:
        return 'D'
    else:
        return 'F'

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
    # current_vals: dict of inputs
    days_left = (exam_date - datetime.now().date()).days
    if days_left <= 0:
        days_left = 1
    weeks = max(1, (days_left + 6)//7)
    diff = max(0, target - predicted)
    # allocate improvements: priority StudyHours, AssignmentScore, Reading, Writing
    plan = []
    # baseline values
    sh = current_vals['StudyHours']
    asg = current_vals['AssignmentScore']
    ws = current_vals['WritingSkills']
    rs = current_vals['ReadingSkills']
    cs = current_vals['ComputerSkills']
    # heuristic: each 1 hour extra ~ 2% score (approx), each 10% assignment improvement ~2%
    # compute required study_hours_inc
    needed_hours_inc = max(0, diff/2)  # rough
    # spread across weeks
    hours_each_week = needed_hours_inc / weeks
    # assignment improvement target
    needed_assignment_inc = max(0, diff*0.4)  # portion
    assignment_each_week = needed_assignment_inc / weeks
    # reading/writing improvements small
    skill_each_week = min(1.0, diff/20)  # rough scale
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

import tempfile

def create_pdf_report(details, charts_bytes_list):
    """
    details: dict of key:value to show text info
    charts_bytes_list: list of bytes objects (each chart as PNG bytes)
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add student details
    for key, val in details.items():
        pdf.multi_cell(0, 8, f"{key}: {val}")
    pdf.ln(5)

    # Add all charts
    for chart_bytes in charts_bytes_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(chart_bytes)
            tmpfile.flush()
            pdf.add_page()
            pdf.image(tmpfile.name, x=10, y=25, w=180)

    return pdf.output(dest="S").encode("latin1")


def create_ics_plan(plan_df, student_name="student", start_date=None):
    cal = Calendar()
    # start_date is exam_date - weeks*7 or today
    for _, row in plan_df.iterrows():
        week_no = int(row['Week'])
        # schedule on Mondays
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

# Main layout: combine sections in accordion/expander for flow
with st.expander("Student Input & Prediction", expanded=True):
    st.subheader("Enter Student Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        study_hours = st.slider("Study Hours per Day:", 0.0, 12.0, 2.5, step=0.5)
        attendance = st.slider("Attendance (%):", 0, 100, 75)
        previous_score = st.slider("Previous Score:", 0, 100, 60)
    with col2:
        assignment_score = st.slider("Assignment Score (%):", 0,100,50)
        writing_skills = st.slider("Writing Skills (1-10):", 1,10,7)
        reading_skills = st.slider("Reading Skills (1-10):",1,10,5)
    with col3:
        computer_skills = st.slider("Computer Skills (1-10):",1,10,8)
        target_score = st.slider("Target Score (%):", 0,100,85)
        exam_date = st.date_input("Exam Date:", value=(datetime.now().date() + relativedelta(days=31)))
    st.markdown("---")
    # What-if interactive sliders
    with st.expander("Interactive What-If Simulator (Slide to see predicted change)", expanded=False):
        sim_sh = st.slider("Simulate StudyHours:", 0.0, 12.0, study_hours, step=0.5, key="sim_sh")
        sim_asg = st.slider("Simulate Assignment Score:", 0, 100, assignment_score, key="sim_asg")
        sim_inputs = np.array([[sim_sh, attendance, previous_score, sim_asg, writing_skills, reading_skills, computer_skills]])

    predict_col1, predict_col2 = st.columns([2,1])
    with predict_col1:
        if st.button(f"🔮 {M['Predict']}"):
            model, test_info = load_or_train_xgb()
            if model is None:
                st.error("No model available. Please go to Admin > Retrain to create a model.")
            else:
                input_df = pd.DataFrame([{
                    'StudyHours': study_hours,
                    'Attendance': attendance,
                    'PreviousScore': previous_score,
                    'AssignmentScore': assignment_score,
                    'WritingSkills': writing_skills,
                    'ReadingSkills': reading_skills,
                    'ComputerSkills': computer_skills
                }])
                pred = predict_percent(model, input_df)[0]
                grade = grade_from_score(pred)
                pf = passfail_from_score(pred)
                st.metric("Predicted Percent (%)", f"{pred:.2f}")
                st.write(f"Grade: **{grade}**")
                st.write(f"Result: **{pf}**")
                # what-if prediction
                sim_pred = predict_percent(model, pd.DataFrame([{
                    'StudyHours': sim_sh,
                    'Attendance': attendance,
                    'PreviousScore': previous_score,
                    'AssignmentScore': sim_asg,
                    'WritingSkills': writing_skills,
                    'ReadingSkills': reading_skills,
                    'ComputerSkills': computer_skills
                }]))[0]
                st.info(f"What-if Predicted with StudyHours={sim_sh}, Assignment={sim_asg}: {sim_pred:.2f}%")
                # Save to history
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'user': username,
                    'StudyHours': study_hours,
                    'Attendance': attendance,
                    'PreviousScore': previous_score,
                    'AssignmentScore': assignment_score,
                    'WritingSkills': writing_skills,
                    'ReadingSkills': reading_skills,
                    'ComputerSkills': computer_skills,
                    'PredictedPercent': round(pred,2),
                    'Grade': grade,
                    'PassFail': pf,
                    'TargetScore': target_score,
                    'ExamDate': str(exam_date)
                }
                save_history(row)
                # Guidance plan
                plan_df = generate_weekly_plan(row, pred, target_score, exam_date)
                st.subheader("Guidance Plan")
                st.dataframe(plan_df)
                # ICS export button
                ics_text = create_ics_plan(plan_df, student_name=username, start_date=None)
                st.download_button("Download Weekly Plan (ICS)", ics_text, file_name="weekly_plan.ics", mime="text/calendar")
                # PDF report
                details = {
                    "Name": username,
                    "PredictedPercent": f"{pred:.2f}%",
                    "Grade": grade,
                    "Result": pf,
                    "TargetScore": f"{target_score}%",
                    "ExamDate": str(exam_date)
                }
                # small charts creation: show charts in UI (plotly) and also create PNG bytes safely for PDF
                # 1) Show gauge on UI
                fig1 = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pred,
                    title = {'text': "Predicted Percent"},
                    gauge = {'axis': {'range': [0,100]}}
                ))
                st.plotly_chart(fig1, use_container_width=True)

                # 2) Create PNG bytes safely for PDF (try plotly -> fallback to matplotlib)
                try:
                    buf1 = fig_to_png_bytes(fig1, pred_value=pred)
                except Exception as e:
                    # ensure buf1 exists
                    buf1 = fallback_generic_plot_bytes(pred_value=pred)

                # Histogram: where this student lies vs dataset (if dataset exists)
                charts_bytes = []

# Gauge chart (Predicted %)
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pred,
    title={'text': "Predicted Percent"},
    gauge={'axis': {'range':[0,100]}}
))
charts_bytes.append(fig_gauge.to_image(format="png"))

# Dataset histogram with predicted marker
if os.path.exists(DATA_PATH):
    df_all = pd.read_csv(DATA_PATH)
    fig_hist = px.histogram(df_all, x='PercentScore', nbins=50, title="Dataset Percent Distribution")
    fig_hist.add_vline(x=pred, line_dash="dash", annotation_text="Predicted", annotation_position="top right")
    charts_bytes.append(fig_hist.to_image(format="png"))

    # Feature importance
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feats = ['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']
        fig_fi = px.bar(x=feats, y=fi, title="Feature Importances")
        charts_bytes.append(fig_fi.to_image(format="png"))

    # Predicted vs Actual
    sample = df_all.sample(2000, random_state=42)
    Xs = sample[['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']]
    ypred_sample = predict_percent(model, Xs)
    fig_pa = go.Figure()
    fig_pa.add_trace(go.Scatter(x=sample['PercentScore'], y=ypred_sample, mode='markers', name='Predicted vs Actual'))
    fig_pa.add_trace(go.Line(x=[0,100], y=[0,100], name='Perfect', line=dict(dash='dash')))
    fig_pa.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
    charts_bytes.append(fig_pa.to_image(format="png"))

    # Residuals
    residuals = sample['PercentScore'] - ypred_sample
    fig_res = px.histogram(residuals, nbins=50, title="Residuals Distribution")
    charts_bytes.append(fig_res.to_image(format="png"))


                pdf_bytes = create_pdf_report(details, charts_bytes)

                st.download_button(
                label="Download Full Report (PDF with all visuals)",
                data=pdf_bytes,
                file_name="report_full.pdf",
                mime="application/pdf")


    with predict_col2:
        # Quick model metrics display (if test_info available or if model file exists)
        if os.path.exists(XGB_MODEL_FILE):
            st.write("Model: XGBoost (Loaded)")
            # If test set available from training, display metrics
            model = joblib.load(XGB_MODEL_FILE)
            if os.path.exists(DATA_PATH):
                df_all = pd.read_csv(DATA_PATH)
                X = df_all[['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']]
                y = df_all['PercentScore']
                # quick eval on a holdout
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                ypred = model.predict(Xte)
                ypred = np.clip(ypred,0,100)
                rmse = np.sqrt(mean_squared_error(yte, ypred))
                r2 = r2_score(yte, ypred)
                st.metric("RMSE", f"{rmse:.3f}")
                st.metric("R2 Score", f"{r2:.3f}")
        else:
            st.info("No trained model loaded. Use Admin > Retrain to create model.")

st.markdown("---")

# Visualization & History Section
with st.expander("Visualizations & History", expanded=False):
    st.subheader("History Table")
    hist_df = load_history()
    st.dataframe(hist_df.sort_values(by='timestamp', ascending=False).head(200))
    st.markdown("### Model Visuals")
    if os.path.exists(XGB_MODEL_FILE):
        model = joblib.load(XGB_MODEL_FILE)
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            feats = ['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']
            fig = px.bar(x=feats, y=fi, title="Feature Importances (XGBoost)")
            st.plotly_chart(fig)
    # If dataset exists, show predicted vs actual using sample test
    if os.path.exists(DATA_PATH):
        df_all = pd.read_csv(DATA_PATH)
        sample = df_all.sample(2000, random_state=42)
        Xs = sample[['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']]
        if os.path.exists(XGB_MODEL_FILE):
            model = joblib.load(XGB_MODEL_FILE)
            ypred = predict_percent(model, Xs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sample['PercentScore'], y=ypred, mode='markers', name='Predicted vs Actual'))
            fig.add_trace(go.Line(x=[0,100], y=[0,100], name='Perfect', line=dict(dash='dash')))
            fig.update_layout(title="Actual vs Predicted (sample)", xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig)
            # residuals
            residuals = sample['PercentScore'] - ypred
            fig2 = px.histogram(residuals, nbins=50, title="Residuals Distribution")
            st.plotly_chart(fig2)
    else:
        st.info("Dataset not present for visuals. Upload in Admin > Retrain if needed.")

st.markdown("---")

# Batch prediction Section
with st.expander("Batch Predictions (CSV Upload)", expanded=False):
    st.markdown("Upload a CSV containing columns: StudyHours, Attendance, PreviousScore, AssignmentScore, WritingSkills, ReadingSkills, ComputerSkills")
    uploaded = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
    
    if uploaded is not None:
        df_batch = pd.read_csv(uploaded)
        required_cols = ['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']
        
        if not all([c in df_batch.columns for c in required_cols]):
            st.error(f"CSV missing required columns. Ensure columns: {required_cols}")
        else:
            # Load model
            model = None
            if os.path.exists(XGB_MODEL_FILE):
                model = joblib.load(XGB_MODEL_FILE)
            else:
                st.warning("No model found. Please go to Admin > Retrain to train model.")
            
            if model is not None:
                # Predictions
                preds = predict_percent(model, df_batch[required_cols])
                df_batch['PredictedPercent'] = np.round(preds,2)
                df_batch['Grade'] = df_batch['PredictedPercent'].apply(grade_from_score)
                df_batch['PassFail'] = df_batch['PredictedPercent'].apply(passfail_from_score)

                st.dataframe(df_batch.head(200))

                # --- Download CSV ---
                csv_bytes = df_batch.to_csv(index=False).encode()
                st.download_button("Download Predictions CSV", csv_bytes, file_name="batch_predictions.csv", mime="text/csv")

                # --- Download PDF Reports ---
                if st.button("Download PDF Reports for Batch (Full Visuals)"):
                    if not os.path.exists(DATA_PATH):
                        st.warning("Dataset not found. Cannot generate full visuals.")
                    elif not os.path.exists(XGB_MODEL_FILE):
                        st.warning("Model not found. Cannot generate visuals.")
                    else:
                        df_all = pd.read_csv(DATA_PATH)
                        feats = ['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']
                        model = joblib.load(XGB_MODEL_FILE)

                        all_pdfs = []
                        for _, r in df_batch.iterrows():
                            details = {
                                "Student Index": r.name,
                                "PredictedPercent": r['PredictedPercent'],
                                "Grade": r['Grade'],
                                "PassFail": r['PassFail']
                            }

                            charts_bytes = []

                            # --- Gauge Chart ---
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=r['PredictedPercent'],
                                title={'text': "Predicted Percent"},
                                gauge={'axis': {'range':[0,100]}}
                            ))
                            charts_bytes.append(fig_gauge.to_image(format="png"))

                            # --- Predicted vs Actual (sample) ---
                            sample = df_all.sample(2000, random_state=42)
                            Xs = sample[feats]
                            ypred = predict_percent(model, Xs)
                            fig_scatter = go.Figure()
                            fig_scatter.add_trace(go.Scatter(x=sample['PercentScore'], y=ypred, mode='markers', name='Predicted vs Actual'))
                            fig_scatter.add_trace(go.Line(x=[0,100], y=[0,100], name='Perfect', line=dict(dash='dash')))
                            fig_scatter.update_layout(title="Actual vs Predicted (sample)", xaxis_title="Actual", yaxis_title="Predicted")
                            charts_bytes.append(fig_scatter.to_image(format="png"))

                            # --- Residuals Histogram ---
                            residuals = sample['PercentScore'] - ypred
                            fig_hist = px.histogram(residuals, nbins=50, title="Residuals Distribution")
                            charts_bytes.append(fig_hist.to_image(format="png"))

                            # --- Feature Importance ---
                            fi = model.feature_importances_
                            fig_fi = px.bar(x=feats, y=fi, title="Feature Importances (XGBoost)")
                            charts_bytes.append(fig_fi.to_image(format="png"))

                            # --- Generate PDF ---
                            pdf_bytes_student = create_pdf_report(details, charts_bytes)
                            all_pdfs.append(pdf_bytes_student)

                        # For simplicity, provide first student PDF as example
                        st.download_button("Download Example Batch PDF (Full Visuals)", data=all_pdfs[0],
                                           file_name="batch_student_full_report.pdf", mime="application/pdf")

                # --- Save Batch to History ---
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

st.markdown("---")

# Admin / Retrain / Model Compare
with st.expander("Admin / Model Management", expanded=False):
    st.subheader("Model Management / Retrain (Admin only)")
    st.write("You need to be admin user to retrain or upload dataset.")
    if user_role != "admin":
        st.info("You are not admin. Login as admin to retrain models.")
    else:
        st.warning("Retraining on large data may take time. Use smaller sample for quick iteration.")
        up = st.file_uploader("Upload CSV dataset for retraining (must contain PercentScore for supervised)", type=['csv'])
        if up is not None:
            df_new = pd.read_csv(up)
            st.write("Preview of uploaded dataset:")
            st.dataframe(df_new.head())
            if st.button("Start Retrain XGBoost"):
                with st.spinner("Training XGBoost..."):
                    X = df_new[['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']]
                    y = df_new['PercentScore']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8,
                                             random_state=42, objective='reg:squarederror')
                    model.fit(X_train, y_train)
                    joblib.dump(model, XGB_MODEL_FILE)
                    ypred = model.predict(X_test)
                    ypred = np.clip(ypred,0,100)
                    rmse = np.sqrt(mean_squared_error(y_test, ypred))
                    r2 = r2_score(y_test, ypred)
                    st.success(f"Retrained and saved XGBoost. RMSE={rmse:.3f}, R2={r2:.3f}")
        # Option: quick model comparison
        st.subheader("Model Comparison (Quick)")
        if st.button("Run Quick Model Comparison (RF vs XGB)"):
            if not os.path.exists(DATA_PATH):
                st.error("No dataset found at DATA_PATH for model comparison.")
            else:
                dfc = pd.read_csv(DATA_PATH)
                X = dfc[['StudyHours','Attendance','PreviousScore','AssignmentScore','WritingSkills','ReadingSkills','ComputerSkills']]
                y = dfc['PercentScore']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # RandomForest
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                ypred_rf = rf.predict(X_test)
                ypred_rf = np.clip(ypred_rf,0,100)
                rmse_rf = np.sqrt(mean_squared_error(y_test, ypred_rf))
                r2_rf = r2_score(y_test, ypred_rf)
                # XGB (if exists)
                xg = None
                if os.path.exists(XGB_MODEL_FILE):
                    xg = joblib.load(XGB_MODEL_FILE)
                    ypred_xg = xg.predict(X_test)
                    ypred_xg = np.clip(ypred_xg,0,100)
                    rmse_xg = np.sqrt(mean_squared_error(y_test, ypred_xg))
                    r2_xg = r2_score(y_test, ypred_xg)
                else:
                    # train quick xg
                    xg = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=42, objective='reg:squarederror')
                    xg.fit(X_train, y_train)
                    ypred_xg = xg.predict(X_test)
                    ypred_xg = np.clip(ypred_xg,0,100)
                    rmse_xg = np.sqrt(mean_squared_error(y_test, ypred_xg))
                    r2_xg = r2_score(y_test, ypred_xg)
                    joblib.dump(xg, XGB_MODEL_FILE)
                st.write("RandomForest: RMSE={:.3f}, R2={:.3f}".format(rmse_rf, r2_rf))
                st.write("XGBoost: RMSE={:.3f}, R2={:.3f}".format(rmse_xg, r2_xg))
                st.success("Model comparison complete. You can choose to save best model.")

st.markdown("---")
st.info("End of App. Created By Shekhar Shelke Contact-shekharshelke45@gmail.com")
