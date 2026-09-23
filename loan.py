from unicodedata import numeric

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("F:\project\Loan prediction\credit_risk_dataset.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)

#check missing value 
print("Missing values in each column:")
print(df.isnull().sum())

#check target variable distribution
print("Target variable distribution:")
print(df["loan_status"].unique())
print(df['loan_status'].value_counts())

#visualize target variable distribution
sns.countplot(x='loan_status', data=df)
plt.show()

#Outlier check
numeric_columns = df.select_dtypes(include=[np.number]).columns
print("Numeric columns for outlier check:", numeric_columns)
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

print((df["person_age"]>100).sum())
print((df["person_emp_length"]>60).sum())

#Correlation check
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

#Data Validation & Outlier Handling
# Handle outliers for person_age and person_emp_length
df_copy=df.copy()
print(f"Original Rows: {df_copy.shape[0]}")

#Remove Duplicate rows
df_copy.drop_duplicates(inplace=True)
print(f"Rows after removing duplicates : {df_copy.shape[0]}")

# Handle outliers for person_age and person_emp_length
#Remove ages below 18 or below 100
df_copy=df_copy[(df_copy["person_age"]>=18) & (df_copy["person_age"]<=100)]
#Remove employment length greater than age or extreme outliers (>60)
df_copy = df_copy[df_copy['person_emp_length'] <= df_copy['person_age']]
df_copy = df_copy[df_copy['person_emp_length'] <= 60]

#Remove the rows with zero or negative income and loan amount
df_copy = df_copy[df_copy['loan_amnt'] > 0]

print("Loan intreset rate range from (min: 5.42 and max: 23.22)")

#Features, Target & Train-Test Split
x=df_copy.drop(columns=['loan_status'])
y=df_copy['loan_status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                     'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

#handle imbalance in target variable
#0 -> no loan (most)
#1 -> yes loan

neg, pos = np.bincount(y_train)
scale_weigths = neg/pos
print(f"Scale weights for handling imbalance: {scale_weigths}")

#Data Preprocessing — Pipelines & Column Transformer
#Raw data is not ready for a model directly.
#We need to prepare numeric and text columns differently.
#We will build a Pipeline for each model.
#We will use ColumnTransformer to apply different steps to different columns.

#Preprocessing for Logistic Regression
#Fill missing numeric values.
#Scale numeric values, since this model is sensitive to scale.
#Convert category columns into numbers using one-hot encoding.

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor_lr = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

#pipeline for XGboost and random forest
preprocessor_xg = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ]), numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

print("Preprocessor for Logistic Regression:", preprocessor_lr)
print("Preprocessor for XGBoost and Random Forest:", preprocessor_xg)

#Stratified Cross-Validation
# One single train-test split may not be reliable.
# We split the training data into five folds instead.
# We check the model's score across all folds.

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score,classification_report
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from itertools import starmap
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = {"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}

#logistic regression cross-validation
lr_cv_pipeline = Pipeline(steps=[('preprocessor', preprocessor_lr),
                                 ('classifier', LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))
                                 ])
#random forest cross-validation
rf_cv_pipeline = Pipeline(steps=[('preprocessor', preprocessor_xg),
                                 ('classifier', RandomForestClassifier(class_weight="balanced", random_state=42,n_estimators=300,max_depth=5))
                                 ])
#xgboost cross-validation
xg_cv_pipeline = Pipeline(steps=[('preprocessor', preprocessor_xg),
                                 ('classifier', XGBClassifier(random_state=42,scale_pos_weight=scale_weigths,n_estimators=300,max_depth=5,learning_rate=0.1))
                                 ])

for cv_name,pipe in [("Logistic Regression", lr_cv_pipeline),
                     ("Random Forest", rf_cv_pipeline),
                     ("XGBoost", xg_cv_pipeline)]:
    cv_result = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring_metrics,n_jobs=-1)
    print(cv_name)
    print(f"ROC-AUC : {cv_result['test_roc_auc'].mean()}")
    print(f"Accuracy : {cv_result['test_accuracy'].mean()}")
    print(f"Precision : {cv_result['test_precision'].mean()}")
    print(f"Recall : {cv_result['test_recall'].mean()}")
    print(f"F1 : {cv_result['test_f1'].mean()}")
    print()

