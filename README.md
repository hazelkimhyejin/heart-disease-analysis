# 🫀 Heart Disease Risk Predictor

A clinical ML web app built with Python and Streamlit, trained on the UCI Cleveland Heart Disease dataset (1,025 patients).

## Live Demo
🔗 [heart-disease-predictor.streamlit.app](https://heart-disease-predictor.streamlit.app) *(deploy link goes here)*

## Model Performance
| Metric | Score |
|---|---|
| **Recall** | **87.4%** ← key metric |
| Precision | 75.6% |
| Accuracy | 79.5% |

> **Why recall?** In medical screening, missing a sick patient (false negative) is far more dangerous than a false alarm. The model uses a 0.40 classification threshold instead of the default 0.50 to maximise sensitivity.

## Key Findings from EDA
- **Chest pain type** and **max heart rate** are the strongest predictors — not cholesterol
- Patients *without* heart disease had paradoxically **higher cholesterol** on average (251 vs 241 mg/dL)
- Typical angina appeared **more often in healthy patients** — explaining why heart disease is frequently misdiagnosed
- Exercise-induced angina appeared in **44% of disease patients** vs 20% of healthy patients

## Tech Stack
- **Python** · pandas · NumPy
- **scikit-learn** — Logistic Regression, StandardScaler, threshold tuning
- **Streamlit** — web app deployment
- **Matplotlib / Seaborn / SciPy** — EDA and visualisation

## Project Structure
```
heart-disease-app/
├── app.py              # Streamlit app
├── model.pkl           # Trained logistic regression model
├── scaler.pkl          # StandardScaler for input normalisation
├── requirements.txt    # Python dependencies
└── README.md
```

## Run Locally
```bash
git clone https://github.com/hazelkimhyejin/heart-disease-analysis
cd heart-disease-analysis
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app.py`
5. Click **Deploy** — live in ~2 minutes

## Next Steps
- [ ] Add Random Forest / XGBoost ensemble → target recall >95%
- [ ] SHAP values for deeper explainability
- [ ] Threshold tuning UI slider for clinicians

## Dataset
UCI Machine Learning Repository — Cleveland Heart Disease Dataset  
Source: [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)  
1,025 patients · 14 variables · 51% disease rate

---

Built by **[Hazel I.](https://linkedin.com/in/hazel-ip-jl)** · Applied AI & ML Portfolio  
Singapore / Seoul · English–Korean Bilingual
