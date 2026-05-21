# Notebooks Directory

## Phase 1: Dataset Loading & EDA ✅
**File**: `01_Dataset_Loading_EDA.py`

This script performs comprehensive exploratory data analysis:
- Loads UCI Hypothyroid Dataset (3,772 records)
- Analyzes data structure and types
- Examines missing values
- Creates statistical summaries
- Generates visualizations

**Run**: `python 01_Dataset_Loading_EDA.py`

**Outputs**:
- `data/hypothyroid_raw.csv` - Raw dataset for next phase
- `reports/01_target_distribution.png` - Class balance visualization
- `reports/02_age_distribution.png` - Age histogram
- `reports/03_gender_distribution.png` - Gender distribution
- `reports/04_tsh_distribution.png` - TSH levels
- `reports/05_boolean_features.png` - Medical history features

## Phase 2: Data Preprocessing
**File**: `02_Data_Preprocessing.py` (Coming Soon)

Will handle:
- Missing value imputation (mean/median/mode)
- Categorical variable encoding
- Feature scaling (StandardScaler)
- Data validation

## Phase 3: Model Building
**File**: `03_Model_Building.py` (Coming Soon)

Will train:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

## Phase 4: Model Evaluation
**File**: `04_Model_Evaluation.py` (Coming Soon)

Will generate:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion matrices
- Model comparison report

## Phase 5: SHAP Explainability
**File**: `05_SHAP_Analysis.py` (Coming Soon)

Will create:
- Feature importance plots
- SHAP force plots
- Decision plots
- Model interpretability analysis

---

**How to Run**:

```bash
# Run Phase 1 (First Time)
python 01_Dataset_Loading_EDA.py

# Monitor outputs in reports/ directory
ls reports/
```

**Note**: Each script is standalone but builds on outputs from previous phases.
