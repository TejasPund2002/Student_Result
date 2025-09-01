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

        # ===== Chart: Week vs Study Hours =====
        st.subheader("📊 Weekly Study Hours Progress")
        fig_study = px.bar(
            df_weekly, x="Week", y="Study Hours",
            color="Study Hours", text="Study Hours",
            color_continuous_scale="Tealgrn"
        )
        fig_study.update_traces(textposition="outside")
        fig_study.update_layout(
            yaxis=dict(title="Study Hours"),
            xaxis=dict(title="Weeks"),
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_study, use_container_width=True)

        # ===== Stylish Cards for Other Attributes =====
        st.subheader("🎯 Skills & Attributes (Need to Improve)")
        col1, col2, col3, col4, col5 = st.columns(5)

        card_style = """
            <div style="
                background: linear-gradient(135deg, #4A90E2, #50E3C2);
                padding: 15px;
                border-radius: 15px;
                text-align: center;
                color: white;
                font-family: 'Segoe UI', sans-serif;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            ">
                <h4 style="margin:0;">{}</h4>
                <p style="font-size:22px; font-weight:bold; margin:0;">{}</p>
            </div>
        """

        col1.markdown(card_style.format("Attendance", f"{df_weekly['Attendance'].iloc[-1]}%"), unsafe_allow_html=True)
        col2.markdown(card_style.format("Assignments", df_weekly["Assignment Score"].iloc[-1]), unsafe_allow_html=True)
        col3.markdown(card_style.format("Writing Skill", df_weekly["Writing"].iloc[-1]), unsafe_allow_html=True)
        col4.markdown(card_style.format("Reading Skill", df_weekly["Reading"].iloc[-1]), unsafe_allow_html=True)
        col5.markdown(card_style.format("Computer Skill", df_weekly["Computer"].iloc[-1]), unsafe_allow_html=True)
