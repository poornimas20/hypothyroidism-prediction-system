# Dataset Directory

## UCI Hypothyroid Dataset

### Download Instructions

Run the download script to automatically fetch the dataset:

```bash
python download_dataset.py
```

Or manually download from:
- **Data file**: https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data
- **Documentation**: https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.names

### Files

- `hypothyroid.data` - Raw dataset file (3,772 records)
- `hypothyroid.names` - Feature documentation
- `hypothyroid_cleaned.csv` - Processed dataset (generated after preprocessing)

### Dataset Overview

**Records**: 3,772 patient records  
**Features**: 29 attributes  
**Classes**: 
- Negative (Normal)
- Compensated hypothyroid
- Primary hypothyroid
- Secondary hypothyroid
- Other thyroid conditions

### Key Features

**Demographic**:
- age
- sex (M/F)

**Medical History (Boolean)**:
- on_thyroxine
- on_antithyroid_medication
- pregnant
- thyroid_surgery
- I131_treatment
- query_hypothyroid
- query_hyperthyroid
- lithium
- goitre
- tumor
- hypopituitary
- psych
- sick

**Lab Results (Continuous)**:
- TSH (Thyroid Stimulating Hormone) - VERY IMPORTANT
- T3 (Triiodothyronine) - VERY IMPORTANT
- TT4 (Total Thyroxine) - VERY IMPORTANT
- T4U (Thyroxine Uptake)
- FTI (Free Thyroxine Index)
- TBG (Thyroxine-binding globulin)

**Other**:
- referral_source
- class (target variable)

### Data Quality Notes

- Missing values are marked as "?"
- Some patients may not have all lab results measured
- Mixed data types: categorical, boolean, continuous
- Class imbalance present in multiclass classification
