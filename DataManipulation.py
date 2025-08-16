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
# titanic = sns.load_dataset('titanic')

titanic = pd.read_csv('Datasets/titanic/train.csv')
titanic.columns = titanic.columns.str.lower()
titanic['deck'] = titanic['cabin'].str[0] 


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
#quality_report = assess_data_quality(titanic)


# 3.1 Handle Missing Data
def handle_missing_data(df):
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
    # df_clean['deck'] = df_clean['deck'].cat.add_categories(['Unknown'])
    df_clean['deck'].fillna('Unknown', inplace=True)

    df_clean['cabin'].fillna('Unknown', inplace=True)


    return df_clean

# print(titanic['deck'].dtype)
# print(type(titanic['deck'].dtype))
titanic_clean = handle_missing_data(titanic)
# print("Missing values after cleaning:")
# print(titanic_clean.isnull().sum())



# 3.2 Feature Engineering
def create_new_features(df):
    """Create new feaatures based on domain knowledge"""
    df_featured = df.copy()

    # Family size
    df_featured['family_size'] = df_featured['sibsp'] + df_featured['parch'] + 1

    # Is alone
    df_featured['is_alone'] = (df_featured['family_size'] == 1).astype(int)

    # Age groups
    df_featured['age_group'] = pd.cut(df_featured['age'],
                                      bins=[0, 12, 18, 60, 100],
                                      labels=['Child', 'Teen', 'Adult', 'Senior'])

    # Fare per person
    df_featured['fare_per_person'] = df_featured['fare'] / df_featured['family_size']

    # Tite extraction from name
    df_featured['title'] = df_featured['name'].str.extract('([A-Za-z]+)\.', expand=False)
    df_featured['title'] = df_featured['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_featured['title'] = df_featured['title'].replace('Mlle', 'Miss')
    df_featured['title'] = df_featured['title'].replace('Ms', 'Miss')
    df_featured['title'] = df_featured['title'].replace('Mme', 'Mrs')

    return df_featured

titanic_featured = create_new_features(titanic_clean)

# 3.3 Prepare Features for ML
def prepare_features_for_ml(df):
    """Prepare features for machine learning"""

    # Select features for modeling
    features_to_use = ['pclass', 'sex', 'fare', 'embarked', 'family_size', 'is_alone', 'title']

    # Create feature matrix
    X = df[features_to_use].copy()
    y = df['survived']

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['sex', 'embarked', 'title']

    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    return X, y, label_encoders

X, y, encoders = prepare_features_for_ml(titanic_featured)
print("Final feature matrix shape:", X.shape)
print("Features:", X.columns.tolist())



################# Part 4 ################################

# 4.1 Train-Validation-Test Split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, 
    stratify=y_temp
)

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)



# 4.2 Model Comparison
def compare_models(X_train, y_train, X_val, y_val):
    """Compare different ML algorithms"""

    models = {
        'Logistic Regression':
        LogisticRegression(random_state=42),
#         'Random Forest Tune 1': RandomForestClassifier(
#     n_estimators=20,       # Fewer trees
#     max_depth=5,           # Limit depth
#     min_samples_split=20,  # Need more samples to split (534/20 ≈ 27 per split)
#     min_samples_leaf=10,   # Need more samples in leaves
#     max_features='sqrt'    # Use fewer features per tree
# ),
#         'Random Forest Tune 2': RandomForestClassifier(
#     n_estimators=15,
#     max_depth=4,
#     min_samples_split=30,
#     min_samples_leaf=15,
#     random_state=42
# ),

    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Calculate metrics
        train_accuracy = (train_pred == y_train).mean()
        val_accuracy = (val_pred == y_val).mean()

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        results[name] = {
            'train_accuracy': train_accuracy, 
            'val_accuracy': val_accuracy, 
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }

        print(f"\n{name}:")
        print(f"    Training Accuracy: {train_accuracy:.4f}")
        print(f"    Validation Accuracy: {val_accuracy:.4f}")
        print(f"    CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return results
    
model_results = compare_models(X_train, y_train, X_val, y_val)


# 4.3 Hyperparameter Optimization
def optimize_random_forest(X_train, y_train):
    """Optimize Random Forest hyperparameters"""

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5 ,10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Grid search with cross-validation
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    return grid_search.best_estimator_

best_rf = optimize_random_forest(X_train, y_train)


################## Part 5 ########################



# 5.1 Comprehensive Model Evaluation
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Comprehensive model evaluation"""

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Probabilities for ROC AUC
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    print("=== MODEL EVALUATION ===")

    # Accuracy scores
    print(f"Training Accuracy: {(train_pred == y_train).mean():.4f}")
    print(f"Validation Accuracy: {(val_pred == y_val).mean():.4f}")
    print(f"Test Accuracy: {(test_pred == y_test).mean():.4f}")

    # ROC AUC scores
    print(f"Training ROC AUC: {roc_auc_score(y_train, train_pred_proba):.4f}")
    print(f"Validation ROC AUC: {roc_auc_score(y_val, val_pred_proba):.4f}")
    print(f"Test ROC AUC: {roc_auc_score(y_test, test_pred_proba):.4f}")

    # Detailed classification report
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, test_pred))

    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)

    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Feature Importance')
        plt.show()

evaluate_model(best_rf, X_train, y_train, X_val, y_val, X_test, y_test)



