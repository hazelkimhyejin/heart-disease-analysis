# Heart Disease Analysis 🫀

## Overview
Analysis of the UCI Cleveland Heart Disease dataset 
consisting of 1025 patients and 14 variables, exploring 
risk factors associated with heart disease.

## Tools Used
- Python
- Pandas
- Matplotlib & Seaborn
- Scipy
- Jupyter Notebook

## Key Findings
1. FIndings about age
Average age was 54, but disease peaked visually around 58, suggesting middle-to-late age carries higher risk.

2. Findings about cholesterol
Average cholesterol was 246 mg/dL, which is borderline high (normal is under 200), yet paradoxically patients WITHOUT disease had higher cholesterol, suggesting cholesterol alone is a poor predictor.

3. Findings about chest pain
Chest pain is positively correlated with heart disease. More chest pain = higher likelihood of having heart disease.

4. Findings about thalach and cp correlation
thalach (max heart rate) is positively correlated with heart disease.


## Surprising Findings
1. Patients with no heart disease have higher cholesterol. Cholesterol alone doesn't tell entire story. This means heart disease is caused by a combination of factors, age, blood pressure, cholesterol and lifestyle, not simply cholesterol alone. Thus, it is important to multivariate analysis (look at many variables together) instead of just one at a time.

2. Typical angina patients had LESS disease than expected.

## Limitations
- Data set dates from 1988 and consists of four databases from Cleveland, Hungary, Switzerland, and Long Beach V. This means that Southeast Asians are not included in the dataset, limiting the dataset.

## How to Run
1. Clone this repo
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy`
3. Open `heart_disease_analysis.ipynb` in Jupyter or VS Code
4. Run all cells

## Dataset
UCI Heart Disease Dataset (Cleveland) via Kaggle