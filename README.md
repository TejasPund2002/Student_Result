# 🎓 Student Performance Prediction System

This project predicts student performance using ML models (XGBoost, RandomForest, LightGBM) and provides study guidance.

---

## 🚀 Features
- **Single Record Prediction** – Enter study details → get predicted score, grade & pass/fail.
- **Batch Prediction** – Upload CSV & download results.
- **Visualizations** – Predicted vs Actual, Residuals, Feature Importance, Dynamic graphs, Gauge meter.
- **Guidance Planner** – Weekly study/skill plan auto-generated.
- **Model Comparison** – RF vs XGBoost vs LightGBM (CV scores).
- **User Accounts** – Track history of predictions & progress.
- **Reports Export** – Download results as PDF, CSV, or ICS calendar.
- **Language Support** – Marathi & English UI toggle.
- **Admin Tools** – Auto retrain when new labeled data uploaded.

---

## 🛠 Tech Stack
- **Backend**: Python, scikit-learn, XGBoost, LightGBM
- **Frontend**: Streamlit, Plotly, Matplotlib, Seaborn
- **Other**: Joblib (model saving), FPDF (report export), ICS (calendar)

---

## 📂 Project Structure
student-performance/
│── artifacts/ # Saved models
│── data/ # Sample datasets
│── app.py # Streamlit app
│── train_model.py # Training pipeline
│── requirements.txt # Dependencies
│── README.md # Documentation

