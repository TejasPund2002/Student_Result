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

# ===== App Title =====
import streamlit as st

# Custom CSS for design and layout
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(45deg, #4A90E2, #50E3C2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'Segoe UI', sans-serif;
        }
        .subtitle {
            text-align: center;
            font-size: 1rem;
            color: #2C3E50;
            margin-top: -10px;
            font-family: 'Segoe UI', sans-serif;
        }
        .divider {
            border: none;
            height: 3px;
            background: linear-gradient(to right, #4A90E2, #50E3C2);
            margin: 20px 0;
        }
        .logo-container {
            display: flex;
            justify-content: center;
        }
        img.logo {
            width: 80px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Layout: Logo centered above header
st.markdown("<div class='logo-container'><img src='https://cdn-icons-png.flaticon.com/512/8833/8833036.png' class='logo'></div>", unsafe_allow_html=True)
st.markdown("<h1 class='title'>Student Result Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>An AI-powered tool to forecast student outcomes</p>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

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
    
    # Store values in session_state for persistence
    st.session_state.percent_score = percent_score
    st.session_state.writing_skills = writing_skills
    st.session_state.reading_skills = reading_skills
    st.session_state.computer_skills = computer_skills
    st.session_state.study_hours = study_hours
    st.session_state.attendance = attendance
    st.session_state.previous_score = previous_score
    st.session_state.assignment_score = assignment_score
    
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
            
# ===== Advanced Visualizations (After Prediction) =====
if 'percent_score' in locals():  # Ensure prediction is done
    with st.expander("📊 Show Advanced Visualizations"):
        st.subheader("Advanced Student Analytics")
        
        # Prepare data for visualizations
        data_dict = {
            "Skills": ["Writing", "Reading", "Computer", "Attendance", "Assignments"],
            "Scores": [writing_skills, reading_skills, computer_skills, attendance, assignment_score]
        }
        df_skills = pd.DataFrame(data_dict)
        
        # Row 1: Bar Chart + Pie Chart
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(
                df_skills, x="Skills", y="Scores", text="Scores", 
                color="Scores", color_continuous_scale="Plasma", title="Skills & Scores"
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(yaxis=dict(range=[0,100]))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                df_skills, names="Skills", values="Scores", 
                color="Skills", color_discrete_sequence=px.colors.sequential.Viridis,
                title="Skill Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Row 2: Radar Chart + Line Chart
        col3, col4 = st.columns(2)
        with col3:
            fig_radar = px.line_polar(
                df_skills, r="Scores", theta="Skills", line_close=True, 
                color_discrete_sequence=px.colors.sequential.Plasma, title="Skill Radar Chart"
            )
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col4:
            fig_line = px.line(
                df_skills, x="Skills", y="Scores", markers=True, 
                title="Score Trend", color_discrete_sequence=["#FF6F61"]
            )
            fig_line.update_traces(marker=dict(size=12))
            st.plotly_chart(fig_line, use_container_width=True)

# ===== Weekly Study Plan (Dynamic UI) ===== 
if 'percent_score' in st.session_state:  # Ensure prediction is done
    st.markdown("## Personal Study Plan")
    st.markdown("Plan Your Weekly Study to Reach Target Score with Improved Visuals")

    # Inputs
    expected_score = st.number_input(
        "Enter Your Expected Score (%)", min_value=0, max_value=100, value=80, step=1, key="expected_score"
    )
    exam_date = st.date_input("Select Exam Date", key="exam_date")

    from datetime import date
    import math
    days_left = (exam_date - date.today()).days

    if days_left <= 0:
        st.warning("Exam date should be in the future!")
    else:
        st.info(f"Days left until exam: {days_left} days")
        weeks_left = math.ceil(days_left / 7)

        # Improvement calculation
        percent_score = st.session_state.percent_score
        improvement_needed = max(0, expected_score - percent_score)
        weekly_improvement = improvement_needed / weeks_left if weeks_left > 0 else 0

        # Current attributes
        base_values = {
            "Study Hours": st.session_state.study_hours,
            "Attendance": st.session_state.attendance,
            "Assignment Score": st.session_state.assignment_score,
            "Writing": st.session_state.writing_skills,
            "Reading": st.session_state.reading_skills,
            "Computer": st.session_state.computer_skills
        }

        # Weekly Plan Data
        weekly_plan = []
        for week in range(1, weeks_left + 1):
            plan = {
                "Week": f"Week {week}",
                "Study Hours": round(base_values["Study Hours"] + weekly_improvement * 0.5 * week, 2),
                "Attendance": round(min(100, base_values["Attendance"] + weekly_improvement * 0.2 * week), 2),
                "Assignment Score": round(base_values["Assignment Score"] + weekly_improvement * 0.3 * week, 2),
                "Writing": round(min(10, base_values["Writing"] + weekly_improvement * 0.1 * week), 2),
                "Reading": round(min(10, base_values["Reading"] + weekly_improvement * 0.1 * week), 2),
                "Computer": round(min(10, base_values["Computer"] + weekly_improvement * 0.1 * week), 2)
            }
            weekly_plan.append(plan)

        df_weekly = pd.DataFrame(weekly_plan)

        # ===== Chart: Week vs Study Hours (Red-Black Theme) =====
        st.subheader("📊 Weekly Study Hours Progress")

        fig_study = px.bar(
            df_weekly, x="Week", y="Study Hours",
            text="Study Hours",
            color="Study Hours",
            color_continuous_scale=["#FF3B3B", "#FF5C5C", "#FF7676"]
        )

        fig_study.update_traces(
            textposition="outside",
            marker=dict(line=dict(width=1, color="black"))
        )

        fig_study.update_layout(
            yaxis=dict(title="Study Hours", color="white"),
            xaxis=dict(title="Weeks", color="white"),
            font=dict(color="white"),
            template="plotly_dark",
            plot_bgcolor="black",
            paper_bgcolor="black",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_study, use_container_width=True)

        # ===== Stylish Cards for Other Attributes =====
        st.subheader("🎯 Skills & Attributes (Need to Improve)")
        col1, col2, col3, col4, col5 = st.columns(5)

        card_style = """
            <div style="
                background: linear-gradient(135deg, #1c1c1c, #000000);
                padding: 15px;
                border-radius: 15px;
                text-align: center;
                color: #FF3B3B;
                font-family: 'Segoe UI', sans-serif;
                box-shadow: 0 4px 12px rgba(255,0,0,0.4);
                border: 1px solid rgba(255,59,59,0.7);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            ">
                <h4 style="margin:0; font-size:16px; font-weight:bold; color:#FF5C5C;">{}</h4>
                <p style="font-size:22px; font-weight:bold; margin:0; color:#ffffff;">{}</p>
            </div>
        """
        col1.markdown(card_style.format("Attendance", f"{df_weekly['Attendance'].iloc[-1]}%"), unsafe_allow_html=True)
        col2.markdown(card_style.format("Assignments", df_weekly["Assignment Score"].iloc[-1]), unsafe_allow_html=True)
        col3.markdown(card_style.format("Writing Skill", df_weekly["Writing"].iloc[-1]), unsafe_allow_html=True)
        col4.markdown(card_style.format("Reading Skill", df_weekly["Reading"].iloc[-1]), unsafe_allow_html=True)
        col5.markdown(card_style.format("Computer Skill", df_weekly["Computer"].iloc[-1]), unsafe_allow_html=True)

# ===== Download Report Section =====
if 'percent_score' in st.session_state:  # Only after prediction
    st.markdown("## 📥 Download Detailed Report")

    student_name = st.text_input("Enter Student Name for Report")
    from datetime import datetime
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if st.button("Generate Report (PDF & Excel)"):
        if student_name.strip() == "":
            st.error("⚠ Please enter student name before generating report.")
        else:
            # ===== Prepare Data =====
            summary_data = {
                "Student Name": student_name,
                "Generated On": report_time,
                "Predicted Score": f"{st.session_state.percent_score:.2f}%",
                "Grade": "A+" if st.session_state.percent_score >= 90 else
                         "A" if st.session_state.percent_score >= 80 else
                         "B+" if st.session_state.percent_score >= 70 else
                         "B" if st.session_state.percent_score >= 60 else
                         "C" if st.session_state.percent_score >= 50 else "D",
                "Status": "Pass" if st.session_state.percent_score >= 40 else "Fail",
                "Study Hours": st.session_state.study_hours,
                "Attendance": st.session_state.attendance,
                "Assignment Score": st.session_state.assignment_score,
                "Writing Skill": st.session_state.writing_skills,
                "Reading Skill": st.session_state.reading_skills,
                "Computer Skill": st.session_state.computer_skills,
            }

            # Weekly Plan Table
            weekly_summary = pd.DataFrame(df_weekly)

            # ===== Suggestions / Roadmap =====
            suggestions = []
            if st.session_state.percent_score < 50:
                suggestions.append("🔴 Focus on basics daily for at least 2 hours.")
                suggestions.append("🔴 Increase assignment completion rate.")
            elif st.session_state.percent_score < 70:
                suggestions.append("🟠 Revise notes regularly and practice writing skills.")
                suggestions.append("🟠 Improve reading speed and comprehension.")
            else:
                suggestions.append("🟢 Maintain consistency in study hours.")
                suggestions.append("🟢 Focus on advanced problem-solving.")

            suggestions.append("✅ Follow the weekly plan strictly.")
            suggestions.append("✅ Keep a balanced sleep and study routine.")

            # ===== Save as Excel =====
            excel_path = f"{student_name}_Report.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                pd.DataFrame([summary_data]).to_excel(writer, sheet_name="Summary", index=False)
                weekly_summary.to_excel(writer, sheet_name="Weekly Plan", index=False)
                pd.DataFrame(suggestions, columns=["Study Suggestions"]).to_excel(writer, sheet_name="Suggestions", index=False)

            # ===== Save as PDF =====
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors

            pdf_path = f"{student_name}_Report.pdf"
            doc = SimpleDocTemplate(pdf_path)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph(f"<para align='center'><font size=18 color='red'><b>Student Result Report</b></font></para>", styles["Title"]))
            story.append(Spacer(1, 12))

            # Summary Table
            data_summary = [[k, v] for k, v in summary_data.items()]
            table = Table(data_summary, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.black),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(table)
            story.append(Spacer(1, 15))

            # Weekly Plan Table 
            story.append(Paragraph("<b>📅 Weekly Study Plan</b>", styles["Heading2"]))

            # Convert values to 2 decimal places if numeric
            data_weekly = [list(weekly_summary.columns)]
            for row in weekly_summary.values.tolist():
                new_row = []
                for val in row:
                    if isinstance(val, (int, float)):  # जर नंबर असेल तर
                        new_row.append(round(val, 2))  # 2 decimal places
                    else:
                        new_row.append(val)  # string तसेच ठेवायचे
                data_weekly.append(new_row)
    
            table2 = Table(data_weekly)
            table2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.red),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table2)
            story.append(Spacer(1, 15))

            # Suggestions
            story.append(Paragraph("<b>📌 Study Suggestions & Roadmap</b>", styles["Heading2"]))
            for s in suggestions:
                story.append(Paragraph(f"- {s}", styles["Normal"]))
                story.append(Spacer(1, 5))

            story.append(Spacer(1, 30))

            # Footer
            footer_text = "<para align='center'><font size=10 color='grey'>Student Result Prediction | Developed by <b>Shekhar Shelke</b> | Powered by AI</font></para>"
            story.append(Paragraph(footer_text, styles["Normal"]))

            doc.build(story)

            # ===== Download Buttons =====
            with open(excel_path, "rb") as f:
                st.download_button("⬇ Download Excel Report", f, file_name=excel_path, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with open(pdf_path, "rb") as f:
                st.download_button("⬇ Download PDF Report", f, file_name=pdf_path, mime="application/pdf")

            st.success("✅ Report generated successfully!")

# ===== Batch Prediction for Multiple Students =====
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

st.markdown("## 👥 Batch Prediction (Multiple Students)")
st.markdown("Upload a CSV or Excel containing multiple students. See sample format below and download the template to fill your data.")

# ---- SAMPLE FORMAT ----
sample_cols = [
    "student_id", "student_name", "study_hours", "attendance",
    "previous_score", "assignment_score", "writing_skills",
    "reading_skills", "computer_skills"
]
sample_df = pd.DataFrame([{
    "student_id": "S001",
    "student_name": "Sample Student",
    "study_hours": 4.5,
    "attendance": 85,
    "previous_score": 65,
    "assignment_score": 70,
    "writing_skills": 6,
    "reading_skills": 6,
    "computer_skills": 7
}])
st.markdown("**Required columns (exact or case-insensitive match):** `student_id`, `study_hours`, `attendance`, `previous_score`, `assignment_score`, `writing_skills`, `reading_skills`, `computer_skills`. Optional: `student_name`.")
st.download_button("⬇ Download Template CSV", data=sample_df.to_csv(index=False).encode('utf-8'), file_name="batch_template.csv", mime="text/csv")
st.write("Sample row preview:")
st.dataframe(sample_df, use_container_width=True)

# ---- Upload ----
uploaded = st.file_uploader("Upload CSV / Excel file with students data", type=["csv", "xlsx", "xls"])
if uploaded is not None:
    try:
        # Try CSV first, then Excel (covers both)
        if uploaded.name.lower().endswith(".csv"):
            df_batch = pd.read_csv(uploaded)
        else:
            try:
                df_batch = pd.read_excel(uploaded)
            except Exception as e:
                st.error("Unable to read Excel — make sure 'openpyxl' is installed in environment.")
                raise e

        st.success(f"File loaded — {df_batch.shape[0]} rows found.")
    except Exception as e:
        st.error("Error reading file. Make sure it's a valid CSV or Excel.")
        st.stop()

    # Normalize column names (lower case, strip)
    df_batch.columns = [c.strip() for c in df_batch.columns]
    colmap = {c.lower(): c for c in df_batch.columns}

    # Required mapping keys (lowercase)
    required = ["student_id", "study_hours", "attendance", "previous_score",
                "assignment_score", "writing_skills", "reading_skills", "computer_skills"]

    missing = [r for r in required if r not in colmap]
    if missing:
        st.error(f"Missing required columns (case-insensitive): {missing}. Make sure your file includes them.")
        st.stop()

    # Reindex / create working df with consistent column order
    df_work = pd.DataFrame()
    for r in required:
        df_work[r] = df_batch[colmap[r]]
    # student_name optional
    if "student_name" in colmap:
        df_work["student_name"] = df_batch[colmap["student_name"]]
    else:
        # if no name column, create placeholder from id
        df_work["student_name"] = df_work["student_id"].astype(str)

    # ---- Predictions ----
    feature_order = ["study_hours", "attendance", "previous_score", "assignment_score",
                     "writing_skills", "reading_skills", "computer_skills"]
    X_batch = df_work[feature_order].astype(float).values
    try:
        preds = model.predict(X_batch)
    except Exception as e:
        st.error("Model prediction failed. Check model compatibility with input feature shapes.")
        st.stop()

    df_work["predicted_score"] = np.round(preds, 2)

    def grade_from_score(s):
        if s >= 90: return "A+"
        if s >= 80: return "A"
        if s >= 70: return "B+"
        if s >= 60: return "B"
        if s >= 50: return "C"
        return "D"
    df_work["grade"] = df_work["predicted_score"].apply(grade_from_score)
    df_work["status"] = df_work["predicted_score"].apply(lambda x: "Pass" if x >= 40 else "Fail")

    # ---- Show first 10 results in a styled "ChatGPT-UI" table ----
    st.markdown("### 🔎 First 10 Predictions (Preview)")
    preview = df_work.head(10).copy()
    preview_display = preview[["student_id", "student_name", "predicted_score", "grade", "status"]].rename(columns={
        "student_id": "ID", "student_name": "Name", "predicted_score": "Predicted (%)",
        "grade": "Grade", "status": "Status"
    })
    # Create nice HTML table with theme (red-black)
    def render_html_table(df):
        header_bg = "#b30000"  # dark red
        header_color = "white"
        row_bg1 = "#111111"
        row_bg2 = "#1a1a1a"
        html = f"""
        <style>
        .batch-table {{border-collapse:collapse; width:100%; font-family: 'Segoe UI', sans-serif;}}
        .batch-table th {{background:{header_bg}; color:{header_color}; padding:8px; text-align:left;}}
        .batch-table td {{padding:8px; color:white; font-size:14px;}}
        .batch-table tr:nth-child(even) {{background:{row_bg2};}}
        .batch-table tr:nth-child(odd) {{background:{row_bg1};}}
        .id-pill {{background:#ff4d4d; color:white; padding:4px 8px; border-radius:8px; font-weight:bold;}}
        </style>
        <table class="batch-table">
            <thead><tr>"""
        for c in df.columns:
            html += f"<th>{c}</th>"
        html += "</tr></thead><tbody>"
        for i, row in df.iterrows():
            html += "<tr>"
            for c in df.columns:
                val = row[c]
                if c == "Predicted (%)":
                    val = f"{float(val):.2f}%"
                if c == "ID":
                    html += f"<td><span class='id-pill'>{val}</span></td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        return html

    st.markdown(render_html_table(preview_display), unsafe_allow_html=True)

    # ---- Key Metrics & Useful Insights ----
    st.markdown("### 📈 Batch Summary & Insights")
    total_students = len(df_work)
    pass_count = (df_work["status"] == "Pass").sum()
    avg_passing_pct = np.round((pass_count / total_students) * 100, 2) if total_students else 0
    top_student = df_work.sort_values("predicted_score", ascending=False).iloc[0]
    top_pct = top_student["predicted_score"]

    # Metrics cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Students", total_students)
    mcol2.metric("Avg Passing %", f"{avg_passing_pct}%")
    mcol3.metric("Top Student %", f"{top_pct:.2f}%")
    mcol4.metric("Passing Count", pass_count)

    # Useful insights (simple auto-generated)
    st.markdown("#### 🔍 Useful Insights")
    insights = []
    avg_study = df_work["study_hours"].mean()
    avg_att = df_work["attendance"].mean()
    insights.append(f"Average study hours: {avg_study:.2f} hrs")
    insights.append(f"Average attendance: {avg_att:.2f}%")
    low_perf = df_work[df_work["predicted_score"] < 40]
    insights.append(f"Students predicted to fail: {len(low_perf)}")
    top_ids = ", ".join(df_work.sort_values("predicted_score", ascending=False).head(3)["student_id"].astype(str).tolist())
    insights.append(f"Top 3 student IDs: {top_ids}")

    for it in insights:
        st.markdown(f"- {it}")

    # ---- Two-column layout of charts (overall stats) -> precisely 2 per row ----
    st.markdown("### 📊 Batch Visualizations")
    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(df_work, x="predicted_score", nbins=12, title="Predicted Score Distribution",
                                labels={"predicted_score": "Predicted (%)"})
        fig_hist.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                               font=dict(color="white"))
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_scatter = px.scatter(df_work, x="study_hours", y="predicted_score", size="attendance",
                                 hover_data=["student_id", "student_name"], title="Study Hours vs Predicted %",
                                 labels={"study_hours": "Study Hours", "predicted_score": "Predicted (%)"})
        fig_scatter.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                                  font=dict(color="white"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        fig_box = px.box(df_work, y="predicted_score", points="all", title="Predicted Score Boxplot")
        fig_box.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                              font=dict(color="white"))
        st.plotly_chart(fig_box, use_container_width=True)
    with c4:
        # average metrics bar
        avg_df = pd.DataFrame({
            "Metric": ["Study Hours", "Attendance", "Assignment Score"],
            "Value": [df_work["study_hours"].mean(), df_work["attendance"].mean(), df_work["assignment_score"].mean()]
        })
        fig_avg = px.bar(avg_df, x="Metric", y="Value", text="Value", title="Average Metrics")
        fig_avg.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                              font=dict(color="white"))
        st.plotly_chart(fig_avg, use_container_width=True)

    # ---- Top #1 Student Detailed Summary + Visuals ----
    st.markdown("### 🏆 Top Student Detailed Summary")
    top = top_student.copy()
    st.markdown(f"**Top Student:** `{top['student_id']}` — **{top['student_name']}**")
    st.markdown(f"- Predicted Score: **{top['predicted_score']:.2f}%**")
    st.markdown(f"- Grade: **{top['grade']}**")
    st.markdown(f"- Status: **{top['status']}**")
    # Detailed cards with red-black theme
    tcol1, tcol2, tcol3 = st.columns([1,1,1])
    def mini_card(title, value):
        card_html = f"""
        <div style="background: linear-gradient(135deg,#0b0b0b,#1a1a1a); padding:12px; border-radius:12px; text-align:center; box-shadow:0 6px 18px rgba(255,0,0,0.12); border:1px solid rgba(255,0,0,0.2);">
            <div style="color:#ff6666; font-weight:700; font-size:14px;">{title}</div>
            <div style="color:white; font-size:20px; font-weight:700; margin-top:6px;">{value}</div>
        </div>
        """
        return card_html
    tcol1.markdown(mini_card("Study Hours", f"{top['study_hours']:.2f}"), unsafe_allow_html=True)
    tcol2.markdown(mini_card("Attendance", f"{top['attendance']:.2f}%"), unsafe_allow_html=True)
    tcol3.markdown(mini_card("Assignment Score", f"{top['assignment_score']:.2f}"), unsafe_allow_html=True)

    # Two charts about top student (2 per row)
    tc1, tc2 = st.columns(2)
    with tc1:
        skills_df = pd.DataFrame({
            "Skill": ["Writing", "Reading", "Computer"],
            "Score": [top["writing_skills"], top["reading_skills"], top["computer_skills"]]
        })
        fig_skills = px.bar(skills_df, x="Skill", y="Score", text="Score", title="Top Student: Skills")
        fig_skills.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                                 font=dict(color="white"), yaxis=dict(range=[0,10]))
        st.plotly_chart(fig_skills, use_container_width=True)
    with tc2:
        # mini radial chart using pie to show passing gap to 100
        fig_pie = px.pie(values=[top["predicted_score"], 100 - top["predicted_score"]],
                         names=["Achieved", "Remaining"], title="Progress to 100%")
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                              font=dict(color="white"))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---- Download batch results as CSV (with predictions) ----
    csv_buf = df_work.copy()
    csv_buf["predicted_score"] = csv_buf["predicted_score"].apply(lambda x: f"{x:.2f}")
    csv_bytes = csv_buf.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Batch Predictions (CSV)", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")

    st.success("Batch processing complete ✅")

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

# ====== Session State for Authentication ======
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "admins" not in st.session_state:
    # default admin
    st.session_state.admins = {"admin": "admin123"}  

# ===== Sidebar Login =====
st.sidebar.title("Login / User")

if not st.session_state.logged_in:
    user_type = st.sidebar.radio("Select User Type", ["Guest", "Admin"])
    
    if user_type == "Guest":
        if st.sidebar.button("Continue as Guest"):
            st.session_state.logged_in = True
            st.session_state.user_type = "Guest"
            st.sidebar.success("Logged in as Guest")

    elif user_type == "Admin":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username in st.session_state.admins and st.session_state.admins[username] == password:
                st.session_state.logged_in = True
                st.session_state.user_type = "Admin"
                st.session_state.username = username
                st.sidebar.success(f"Logged in as Admin ({username})")
            else:
                st.sidebar.error("Invalid Credentials")

else:
    st.sidebar.success(f"Logged in as {st.session_state.user_type}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.experimental_rerun()

# ===== Admin Only Section =====
if st.session_state.logged_in and st.session_state.user_type == "Admin":
    st.subheader("🔧 Admin Panel")
    with st.expander("Retrain Model"):
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
            st.success("✅ Model retrained and saved successfully!")

    with st.expander("Manage Admins"):
        new_username = st.text_input("New Admin Username")
        new_password = st.text_input("New Admin Password", type="password")
        if st.button("Add Admin"):
            if new_username in st.session_state.admins:
                st.warning("⚠️ This admin already exists!")
            else:
                st.session_state.admins[new_username] = new_password
                st.success(f"✅ New Admin '{new_username}' added successfully!")
