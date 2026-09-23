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
