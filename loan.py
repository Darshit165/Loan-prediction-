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
