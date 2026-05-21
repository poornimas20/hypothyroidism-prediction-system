# Phase 1 Project Structure

## Phase 1: Data Processing & Model Training

This directory contains the foundational machine learning pipeline for the Hypothyroidism Prediction System.

### ✅ Phase 1 Deliverables

#### 1. Data Processing (`src/data_processor.py`)
- Load UCI hypothyroid dataset
- Handle missing values (>50% threshold rule)
- Remove duplicate records
- Encode categorical features
- Handle outliers using IQR method
- Normalize numerical features
- Export cleaned dataset

#### 2. Model Training (`src/model_trainer.py`)
- Implement 3 ML algorithms:
  - Logistic Regression
  - Random Forest (100 estimators)
  - Gradient Boosting (100 estimators)
- Train/test split (80/20)
- Stratified sampling for class balance
- Cross-validation support
- Comprehensive metrics calculation

#### 3. Configuration (`src/config.py`)
- Centralized settings management
- File path configuration
- Model hyperparameters
- Performance thresholds
- Class definitions

#### 4. Utilities (`src/utils.py`)
- Logging setup
- Directory creation
- Event logging
- Output formatting

#### 5. Scripts
- `scripts/train_model.py`: Main pipeline orchestrator
- `scripts/predict.py`: Prediction engine for inference
- `data/download_dataset.py`: Automated dataset download

### 📊 Model Evaluation Metrics

Each model is evaluated on:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate among positive predictions
- **Recall**: Sensitivity - detection of actual positives
- **F1-Score**: Harmonic mean balancing precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

### 🎯 Expected Performance Targets

- Minimum Accuracy: 85%
- Minimum Precision: 80%
- Minimum Recall: 80%
- Minimum F1-Score: 80%

### 📁 Generated Files

After running `scripts/train_model.py`:
- `data/hypothyroid_cleaned.csv` - Processed dataset
- `models/best_model.pkl` - Trained model binary
- `app.log` - Training logs and events

### 🔄 Phase 1 Pipeline Workflow

```
1. Download Dataset
   └─> data/hypothyroid.data

2. Data Processing
   ├─> Load raw data
   ├─> Assign column names
   ├─> Handle missing values
   ├─> Remove duplicates
   ├─> Encode categorical features
   ├─> Handle outliers
   ├─> Normalize features
   └─> data/hypothyroid_cleaned.csv

3. Train/Test Split
   ├─> 80% training set
   └─> 20% testing set

4. Model Training
   ├─> Logistic Regression
   ├─> Random Forest
   └─> Gradient Boosting

5. Model Evaluation
   ├─> Calculate metrics
   ├─> Compare performance
   └─> Select best model

6. Save Best Model
   └─> models/best_model.pkl
```

### 🚀 How to Run Phase 1

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
cd data
python download_dataset.py
cd ..

# 3. Run complete pipeline
python scripts/train_model.py

# 4. Make predictions with trained model
python scripts/predict.py
```

### 📝 Code Organization

```
src/
├── __init__.py              # Package initialization
├── config.py                # Configuration and constants
├── data_processor.py         # DataProcessor class
├── model_trainer.py          # ModelTrainer class
└── utils.py                 # Utility functions

scripts/
├── train_model.py           # Training pipeline orchestrator
└── predict.py               # Prediction engine

data/
├── download_dataset.py      # Dataset download utility
├── README.md                # Data documentation
└── (generated files)
```

### 🔧 Key Classes

#### DataProcessor
```python
processor = DataProcessor()
processor.process_data(input_path, output_path)
```
Handles all data cleaning and preprocessing steps.

#### ModelTrainer
```python
trainer = ModelTrainer()
trainer.train_and_save(data_path, model_path)
```
Trains, evaluates, and saves the best model.

#### PredictionEngine
```python
engine = PredictionEngine(model_path)
prediction = engine.predict(features)
```
Makes predictions using trained model.

### 📚 Dataset Details

- **Total Records**: 3,772
- **Features**: 29 clinical attributes
- **Target Classes**: 4
  - Negative (Normal thyroid function)
  - Compensated hypothyroid
  - Primary hypothyroid
  - Secondary hypothyroid
- **Data Types**: Mixed (categorical, boolean, continuous)
- **Missing Values**: Marked as '?'

### 🔍 Feature Categories

**Demographic** (2):
- age, sex

**Medical History** (17):
- on_thyroxine, pregnant, thyroid_surgery, etc.

**Lab Results** (7):
- TSH, T3, T4, T3U, FTI, TBG, TPO

**Other** (3):
- referral_source, query_type, class

### 💾 Model Persistence

The best trained model is saved as a pickle file:
- Format: scikit-learn model binary
- Size: ~2-5 MB depending on model type
- Used for: Inference and predictions
- Compatibility: Python 3.6+, scikit-learn 1.0+

### 🎓 Learning Outcomes

By completing Phase 1, you'll understand:
- Complete ML pipeline from raw data to trained model
- Data preprocessing and feature engineering techniques
- Model training, evaluation, and selection
- Performance metrics interpretation
- Model persistence and loading
- Python project structure and organization

### ⚠️ Important Notes

1. **Data Privacy**: Keep model and data secure
2. **Version Control**: Always commit configuration changes
3. **Environment**: Use virtual environment (venv)
4. **Dependencies**: Lock versions in requirements.txt
5. **Logging**: Check app.log for debugging

### 📈 Performance Expectations

Model performance varies by:
- Dataset quality and size
- Feature selection and engineering
- Model hyperparameters
- Data preprocessing techniques
- Train/test split ratio

### 🔮 Preparing for Phase 2

Phase 1 outputs used in Phase 2:
- Trained model (`best_model.pkl`)
- Feature names and preprocessing info
- Class mappings
- Configuration settings

---

**Phase 1 Status**: ✅ Complete  
**Next Phase**: Phase 2 - Flask Web Application
