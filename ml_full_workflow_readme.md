# Full ML Workflow: From 0 to Deployment

This is a comprehensive guide for using the **Reusable Supervised ML Template** to build and deploy a supervised learning model.

---

## 1️⃣ Project Setup

**Folder Structure:**
```
ml-supervised-template/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── regression_models/
│   └── classification_models/
├── models/
│   ├── trained_models/
│   └── scalers_encoders/
├── reports/
│   ├── figures/
│   └── model_comparison.csv
├── src/
└── README.md
```

**Step 1:** Place your dataset in `data/raw/`.

---

## 2️⃣ Data Cleaning
**Notebook:** `01_data_cleaning.ipynb`

**Tasks:**
- Handle missing values
- Remove duplicates
- Fix data types
- Remove outliers
- Save cleaned data to `data/interim/cleaned.csv`

**Usage:**
```python
import pandas as pd
df = pd.read_csv('data/interim/cleaned.csv')
df.head()
```

---

## 3️⃣ Exploratory Data Analysis (EDA)
**Notebook:** `02_eda.ipynb`

**Tasks:**
- Target distribution analysis
- Numerical and categorical feature analysis
- Correlation heatmap
- Document insights for model selection
- Save plots to `reports/figures/`

**Usage:**
```python
# Run notebook cells
# Review and save key insights
```

---

## 4️⃣ Feature Engineering
**Notebook:** `03_feature_engineering.ipynb`

**Tasks:**
- Separate features and target
- Encode categorical features
- Perform feature selection
- Save processed data to `data/processed/features.csv`

**Usage:**
```python
processed_df = pd.read_csv('data/processed/features.csv')
processed_df.head()
```

---

## 5️⃣ Preprocessing
**Notebook:** `04_preprocessing.ipynb`

**Tasks:**
- Train-test split
- Scale numeric features
- Save scalers to `models/scalers_encoders/`

**Usage:**
```python
import joblib
scaler = joblib.load('models/scalers_encoders/scaler.pkl')
X_test_scaled = scaler.transform(X_test)
```

---

## 6️⃣ Model Training
**Notebooks:**
- Regression: `notebooks/regression_models/`
- Classification: `notebooks/classification_models/`

**Workflow:**
1. Load preprocessed datasets
2. Train baseline model first
3. Train advanced models
4. Evaluate metrics (RMSE, R², Accuracy, F1, ROC-AUC)
5. Save best model to `models/trained_models/`

**Example:**
```python
from src.utils import save_model
save_model(model, 'models/trained_models/best_model.pkl')
```

---

## 7️⃣ Model Evaluation & Comparison
**Tasks:**
- Record metrics
- Save model comparison CSV
- Save evaluation plots

**Example:**
```python
results_df.to_csv('reports/model_comparison.csv', index=False)
```

**Load comparison:**
```python
import pandas as pd
comparison_df = pd.read_csv('reports/model_comparison.csv')
print(comparison_df)
```

---

## 8️⃣ Deployment (Optional)
**Steps:**
1. Load trained model and scaler
2. Prepare API using Flask or FastAPI
3. Use endpoints to predict new data

**Example:**
```python
import joblib
from fastapi import FastAPI

model = joblib.load('models/trained_models/best_model.pkl')
scaler = joblib.load('models/scalers_encoders/scaler.pkl')

app = FastAPI()

@app.post('/predict')
def predict(data: list):
    X = scaler.transform([data])
    y_pred = model.predict(X)
    return {'prediction': y_pred.tolist()}
```

**Run API:**
```bash
uvicorn main:app --reload
```

---

## 9️⃣ General Tips
- Run notebooks in order: 01 → 02 → 03 → 04 → model notebooks
- Keep `src/` modular and import functions in notebooks
- Document all observations and save artifacts (models, scalers, plots)
- Keep raw data untouched
- Maintain reproducibility with saved scalers and encoders
- Use `reports/model_comparison.csv` to track experiments

---

This workflow ensures a **professional, reproducible, and GitHub-ready ML project** from scratch to deployment.

