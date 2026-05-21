"""
PHASE 1: Dataset Loading & Exploratory Data Analysis
=====================================================

This notebook covers:
1. Loading the UCI Hypothyroid Dataset
2. Understanding the structure and features
3. Exploratory Data Analysis (EDA)
4. Missing value analysis
5. Statistical summary
6. Visualization of key insights

Author: Poornima S
MCA Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# STEP 1: LOAD AND UNDERSTAND THE DATASET
# ============================================================================

def load_hypothyroid_dataset(data_path="data/hypothyroid.data"):
    """
    Load the UCI Hypothyroid Dataset
    
    Parameters:
    -----------
    data_path : str
        Path to the hypothyroid.data file
    
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataset
    """
    
    # Feature names (from UCI documentation)
    feature_names = [
        'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
        'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
        'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
        'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
        'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
        'referral_source', 'class'
    ]
    
    # Load data
    df = pd.read_csv(data_path, header=None, names=feature_names, na_values='?')
    
    return df


def display_dataset_info(df):
    """
    Display basic information about the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    """
    
    print("="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"\nShape: {df.shape}")
    print(f"Records: {df.shape[0]}")
    print(f"Features: {df.shape[1]}")
    
    print("\n" + "="*80)
    print("DATA TYPES")
    print("="*80)
    print(df.dtypes)
    
    print("\n" + "="*80)
    print("FIRST 5 ROWS")
    print("="*80)
    print(df.head())
    
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(df.describe())


def analyze_missing_values(df):
    """
    Analyze missing values in the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    """
    
    print("\n" + "="*80)
    print("MISSING VALUES ANALYSIS")
    print("="*80)
    
    missing_data = pd.DataFrame({
        'Feature': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_data[missing_data['Missing_Count'] > 0])
    
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")
    print(f"Percentage of missing data: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")
    
    return missing_data


def analyze_target_variable(df):
    """
    Analyze the target variable (class) distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    """
    
    print("\n" + "="*80)
    print("TARGET VARIABLE DISTRIBUTION")
    print("="*80)
    
    class_dist = df['class'].value_counts()
    print("\nClass Distribution:")
    print(class_dist)
    
    print("\nClass Distribution (%):")
    print((class_dist / len(df) * 100).round(2))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    class_dist.plot(kind='bar', color='steelblue')
    plt.title('Distribution of Target Variable (Class)', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('reports/01_target_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: reports/01_target_distribution.png")
    plt.close()


def analyze_numeric_features(df):
    """
    Analyze numeric features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    """
    
    print("\n" + "="*80)
    print("NUMERIC FEATURES ANALYSIS")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric Features: {len(numeric_cols)}")
    print(list(numeric_cols))
    
    # Important thyroid hormone levels
    print("\n" + "-"*80)
    print("KEY THYROID HORMONE LEVELS")
    print("-"*80)
    
    thyroid_features = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG', 'age']
    
    for feature in thyroid_features:
        if feature in df.columns:
            print(f"\n{feature}:")
            print(f"  Count: {df[feature].count()}")
            print(f"  Mean: {df[feature].mean():.4f}")
            print(f"  Std: {df[feature].std():.4f}")
            print(f"  Min: {df[feature].min():.4f}")
            print(f"  Max: {df[feature].max():.4f}")
            print(f"  Missing: {df[feature].isnull().sum()}")


def analyze_categorical_features(df):
    """
    Analyze categorical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    """
    
    print("\n" + "="*80)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical Features: {len(categorical_cols)}")
    print(list(categorical_cols))
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Value counts:\n{df[col].value_counts()}")
        print(f"  Missing: {df[col].isnull().sum()}")


def visualize_key_features(df):
    """
    Create visualizations for key features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    """
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Age Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['age'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Age Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/02_age_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/02_age_distribution.png")
    plt.close()
    
    # 2. Sex Distribution
    plt.figure(figsize=(8, 6))
    df['sex'].value_counts().plot(kind='bar', color=['pink', 'skyblue'])
    plt.title('Gender Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/03_gender_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/03_gender_distribution.png")
    plt.close()
    
    # 3. TSH Distribution (log scale due to wide range)
    plt.figure(figsize=(10, 6))
    tsh_data = df['TSH'].dropna()
    plt.hist(tsh_data[tsh_data > 0], bins=50, color='green', edgecolor='black')
    plt.title('TSH Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('TSH Level', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/04_tsh_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/04_tsh_distribution.png")
    plt.close()
    
    # 4. Boolean Features Overview
    boolean_cols = [col for col in df.columns if df[col].dtype == 'object' and 
                    set(df[col].dropna().unique()).issubset({'f', 't', 'F', 'T'})]
    
    print(f"\nBoolean Features Found: {boolean_cols}")
    
    if len(boolean_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for idx, col in enumerate(boolean_cols[:4]):
            ax = axes[idx // 2, idx % 2]
            df[col].value_counts().plot(kind='bar', ax=ax, color=['coral', 'lightblue'])
            ax.set_title(f'{col} Distribution', fontweight='bold')
            ax.set_xlabel('')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/05_boolean_features.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: reports/05_boolean_features.png")
        plt.close()


def generate_eda_report(df):
    """
    Generate a comprehensive EDA report
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    """
    
    print("\n" + "="*80)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS - COMPLETE")
    print("="*80)
    
    print("\n📊 KEY INSIGHTS:")
    print(f"  • Total Records: {df.shape[0]}")
    print(f"  • Total Features: {df.shape[1]}")
    print(f"  • Missing Data: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")
    print(f"  • Target Classes: {df['class'].nunique()}")
    print(f"  • Numeric Features: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  • Categorical Features: {len(df.select_dtypes(include=['object']).columns)}")
    
    print("\n✓ EDA Complete!")
    print("Next Step: Phase 2 - Data Preprocessing")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Create reports directory if it doesn't exist
    Path("reports").mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("HYPOTHYROIDISM PREDICTION SYSTEM")
    print("PHASE 1: DATASET LOADING & EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Step 1: Load Dataset
    print("\n[1/7] Loading dataset...")
    df = load_hypothyroid_dataset()
    print("✓ Dataset loaded successfully!")
    
    # Step 2: Display basic info
    print("\n[2/7] Displaying dataset information...")
    display_dataset_info(df)
    
    # Step 3: Analyze missing values
    print("\n[3/7] Analyzing missing values...")
    analyze_missing_values(df)
    
    # Step 4: Analyze target variable
    print("\n[4/7] Analyzing target variable...")
    analyze_target_variable(df)
    
    # Step 5: Analyze numeric features
    print("\n[5/7] Analyzing numeric features...")
    analyze_numeric_features(df)
    
    # Step 6: Analyze categorical features
    print("\n[6/7] Analyzing categorical features...")
    analyze_categorical_features(df)
    
    # Step 7: Create visualizations
    print("\n[7/7] Creating visualizations...")
    visualize_key_features(df)
    
    # Generate final report
    generate_eda_report(df)
    
    # Save cleaned dataset for next phase
    print("\nSaving processed dataset for next phase...")
    df.to_csv("data/hypothyroid_raw.csv", index=False)
    print("✓ Saved: data/hypothyroid_raw.csv")
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE ✓")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review the generated visualizations in 'reports/' directory")
    print("2. Proceed to Phase 2: Data Preprocessing")
    print("3. Run: python notebooks/02_Data_Preprocessing.py")
    print("="*80 + "\n")
