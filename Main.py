import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)

# imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

import xgboost as xgb
import lightgbm as lgb

# Settings for plots
plt.style.use('seaborn-v0_8')

from Preprocessing import (
    analyze_target, univariate_analysis, bivariate_analysis,
    feature_engineering,
)

from Modelling import (run_credit_scoring_complete,save_best_model)

# ====================================
# 1. OVERVIEW OF THE DATA, HANDLING MISSING VALUES & OUTLIERS
# ====================================


df_train = pd.read_csv('GiveMeSomeCredit/cs-training.csv')
df_val = pd.read_csv('GiveMeSomeCredit/cs-test.csv')

#Create ID column
df_train.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
df_val.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

# First Overview
print(df_train.shape)
print(df_train.describe())
print("\nThere is some outliers in variables 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio'\n")

#Analyze outliers in 'RevolvingUtilizationOfUnsecuredLines' and 'DebtRatio'
print(df_train[(df_train['RevolvingUtilizationOfUnsecuredLines'] >3) ].shape)

 
df_train = df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] <= 3]

 
plt.hist(df_train[(df_train['DebtRatio'] > 30) ] )
plt.ylabel('Count')
plt.xlabel('DebtRatio')
plt.title('DebtRatio Distribution')
plt.show()
df_train[df_train['DebtRatio'] > 10000].head(10)


 
print(df_train.isnull().sum())


print("""\nWe can see that there is a lot of missing values for the variable 
 MonthlyIncome, so we can't just eject them because of the number of 
 observation that would exclude. So instead, we gonna impute the median 
 value of the MonthlyIncome. And for the missing value in NumberOfDependents, 
 we gonna eject them.\n""")

 
print(df_train[df_train['MonthlyIncome'].isnull()].describe())

#Impute the missing values in MonthlyIncome
median_income = df_train["MonthlyIncome"].median()
mask_nan_income = df_train["MonthlyIncome"].isna()

 
df_train.loc[mask_nan_income, "MonthlyIncome"] = median_income

 
median_debt = df_train["DebtRatio"].median()
df_train.loc[mask_nan_income, "DebtRatio"] = median_debt

 
print(df_train.describe())

#Eject the Missing Values in NumberOfDependents 
print(df_train[df_train['NumberOfDependents'].isnull()].describe())
print("We can see that the missing values in NumberOfDependents aren't connected")

 
df_train = df_train[df_train['NumberOfDependents'].notnull()]

 
print(df_train[df_train["DebtRatio"] > 3].shape)

 
df_train = df_train[df_train["DebtRatio"] <= 3]

df_train.describe()
print("\n The max value for MonthlyIncome is looking like a outlier")

print(df_train[df_train['MonthlyIncome'] > 30000].sort_values(by = 'MonthlyIncome', ascending=False).head(10))

# We can see that there is some people with age = 0
print("We can see that there is some people with age = 0")
df_train = df_train[df_train['age'] != 0]

# ====================================
# 2. PREPROCESSING
# ====================================


analyze_target(df=df_train)
univariate_analysis(df=df_train)
bivariate_analysis(df=df_train)

# ====================================
# 3. MODELLING
# ====================================


pipeline_results = run_credit_scoring_complete(df_train, df_val, target_col='SeriousDlqin2yrs',
                             handle_imbalance_method='smote')


save_best_model(pipeline_results)

