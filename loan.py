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

# Evaluation Helper Function
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve
def evaluate_model(model_name, model, X_test, y_test, threshold=None):
  y_pred_prob = model.predict_proba(X_test)[:,1]

  if threshold:
    y_pred = (y_pred_prob > threshold).astype(int)
  else:
    y_pred = model.predict(X_test)

  #Metrics
  accuracy     = accuracy_score(y_test, y_pred)    #Overall model's preformance
  precision    = precision_score(y_test, y_pred)   #When the model says 1, how oftenly is it actually 1.
  recall       = recall_score(y_test, y_pred)      #of the actual 1 how many did the model catches
  f1           = f1_score(y_test, y_pred)          #Harmonic mean of precision and recall

  conf_matrix   = confusion_matrix(y_test, y_pred)
  class_report  = classification_report(y_test, y_pred)


  print(f"{model_name} Metrics:")
  if threshold is not None: # Only print threshold if provided
    print(f"Used Threshold : {threshold:.2f}")
  print(f"Accuracy : {accuracy:.2f}")
  print(f"Precision : {precision:.2f}")
  print(f"Recall : {recall:.2f}")
  print(f"F1 : {f1:.2f}")
  print()
  print(f"{model_name} Classification Report:")
  print(class_report)


  #Plotting confusion matrix
  sns.heatmap(conf_matrix, annot=True, fmt="d")
  plt.title(f"{model_name} Confusion Matrix")
  plt.ylabel("Actual Values")
  plt.xlabel("Predicted Values")
  plt.show()


  #Plotting ROC-AUC curve
  precisions, recalls, threshold = precision_recall_curve(y_test, y_pred_prob)
  plt.plot(recalls, precisions)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title(f"{model_name} Precision-Recall Curve")
  plt.show()

  #Baseline Model — Logistic Regression
  from sklearn.pipeline import Pipeline
baseline_model=Pipeline([
    ("preprocessor",preprocessor_lr),
    ("classifier",LogisticRegression(class_weight="balanced",random_state=42))
])
baseline_model.fit(X_train,y_train)
evaluate_model("Baseline Logistic Regression", baseline_model, X_test, y_test)

#Baseline Model — Randomforest classification Regression
from sklearn.ensemble import RandomForestClassifier
randomforest=Pipeline([
    ("preprocessor",preprocessor_xg),
    ("classifier",RandomForestClassifier(random_state=42,class_weight="balanced"))
])
randomforest.fit(X_train,y_train)
evaluate_model("Baseline Random Forest", randomforest, X_test, y_test)
#random forest Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
param_grid_forest = {
    "classifier__n_estimators": [231],
    "classifier__max_depth" : [4],
    "classifier__min_samples_split" :[8],
    "classifier__min_samples_leaf" : [2],
    "classifier__max_features" : ["sqrt"],
    "classifier__max_leaf_nodes" : [32],
    "classifier__min_impurity_decrease" :[0.000301742493144741],
    "classifier__criterion":["entropy"],
    "classifier__bootstrap" : [True],
    "classifier__oob_score" : [False],
    "classifier__warm_start" : [True],
    "classifier__class_weight" : ["balanced"]
}

random_search_forest= RandomizedSearchCV(
    randomforest,
    param_grid_forest,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1,
    verbose=3
)
random_search_forest.fit(X_train,y_train)
evaluate_model("Random Forest after Hyperparameter Tuning", random_search_forest.best_estimator_, X_test, y_test)