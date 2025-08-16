import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline



# Model persistence
import joblib

# Warning
import warnings
warnings.filterwarnings('ignore')











# Load Titanic dataset 
import seaborn as sns
titanic = sns.load_dataset('titanic')



#First look at our data
# print("Dataset shape:", titanic.shape)
# print("\nFirst 5 rows:")
# print(titanic.head())


# print("Dataset info:")
# print(titanic.info())
# print("\nMissing values:")
# print(titanic.isnull().sum())
# print("\nBasic statistics:")
# print(titanic.describe())


# print("Survival distribution:")
# print(titanic['survived'].value_counts())
# print("\nSurvival rate:")
# print(titanic['survived'].mean())

# plt.figure(figsize=(8,6))
# sns.countplot(data=titanic, x='survived')
# plt.title('Distribution of Survival')
# plt.show()


# 2.3 Categorical features analysis
# categorical_features = ['sex', 'embarked', 'class', 'who', 'deck']
# for feature in categorical_features:
#     if feature in titanic.columns:
#         print(f"\n{feature} distribution:")
#         print(titanic[feature].value_counts())

        
#         #Survival rate by category
#         survival_rate = titanic.groupby(feature)['survived'].mean()
#         print(f"Survival rate by {feature}:")
#         print(survival_rate)



# 2.4 Correlation Analysis
# Correlation matrix for numerical feature
# plt.figure(figsize=(10,8))
# correlation_matrix = titanic.corr(numeric_only=True)
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Feature Correlation Matrix')
# plt.show()


# 2.5 Data Quality Assessment
def assess_data_quality(df):
    """Comprehensive data quality assessment"""

    print("=== DATA QUALITY REPORT ===")

    # Missing data analysis
    print("\n1. Missing Data Analysis:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_table = pd.DataFrame({
        'Missing Count': missing_data,
        'Percentage': missing_percent
    })
    print(missing_table[missing_table['Missing Count'] > 0])

    # Duplicate rows
    print(f"\n2. Duplicate rows: {df.duplicated().sum()}")

    # Outlier detection for numerical columns
    print("\n3. Potential outliers (values beyond 3 standard deviations):")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} potential outliers")


    print("\n4. check datatypes of each column")
    df.dtypes
    df.info()


    print("\n5. shows min max for columns")
    df.describe(include='all')

    return missing_table

# Run data quality assessment
quality_report = assess_data_quality(titanic)

def hanlde_missing_data(df):
    """Handle missing data with domain knowledge"""
    df_clean = df.copy()

    # Age: Fill with median based on passenger class and sex
    df_clean['age'].fillna(
        df_clean.groupby(['pclass', 'sex'])['age'].transform('median'),
        inplace=True
    )

    # Embarked: Fill with most common value 
    df_clean['embarked'].fillna(df_clean['embarked'].mode()[0], inplace=True)

    # Deck: Create 'Unknown' category
    df_clean['deck'].fillna('Unknown', inplace=True)

    return df_clean

titanic_clean = hanlde_missing_data(titanic)
print("Missing values after cleaning:")
print(titanic_clean.isnull().sum())



