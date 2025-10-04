import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ====================================
# 1. OVERVIEW OF THE DATA, HANDLING MISSING VALUES & OUTLIERS
# ====================================


# df_train = pd.read_csv('GiveMeSomeCredit/cs-training.csv')
# df_test = pd.read_csv('GiveMeSomeCredit/cs-test.csv')

# #Create ID column
# df_train.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
# df_test.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

# # First Overview
# print(df_train.shape)
# print(df_train.describe())
# print("\nThere is some outliers in variables 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio'\n")

# #Analyze outliers in 'RevolvingUtilizationOfUnsecuredLines' and 'DebtRatio'
# print(df_train[(df_train['RevolvingUtilizationOfUnsecuredLines'] >3) ].shape)

 
# df_train = df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] <= 3]

 
# plt.hist(df_train[(df_train['DebtRatio'] > 30) ] )
# plt.ylabel('Count')
# plt.xlabel('DebtRatio')
# plt.title('DebtRatio Distribution')
# plt.show()
# df_train[df_train['DebtRatio'] > 10000].head(10)


 
# print(df_train.isnull().sum())


# print("""\nWe can see that there is a lot of missing values for the variable 
#  MonthlyIncome, so we can't just eject them because of the number of 
#  observation that would exclude. So instead, we gonna impute the median 
#  value of the MonthlyIncome. And for the missing value in NumberOfDependents, 
#  we gonna eject them.\n""")

 
# print(df_train[df_train['MonthlyIncome'].isnull()].describe())

# #Impute the missing values in MonthlyIncome
# median_income = df_train["MonthlyIncome"].median()
# mask_nan_income = df_train["MonthlyIncome"].isna()

 
# df_train.loc[mask_nan_income, "MonthlyIncome"] = median_income

 
# median_debt = df_train["DebtRatio"].median()
# df_train.loc[mask_nan_income, "DebtRatio"] = median_debt

 
# print(df_train.describe())

# #Eject the Missing Values in NumberOfDependents 
# print(df_train[df_train['NumberOfDependents'].isnull()].describe())
# print("We can see that the missing values in NumberOfDependents aren't connected")

 
# df_train = df_train[df_train['NumberOfDependents'].notnull()]

 
# print(df_train[df_train["DebtRatio"] > 3].shape)

 
# df_train = df_train[df_train["DebtRatio"] <= 3]

# df_train.describe()
# print("\n The max value for MonthlyIncome is looking like a outlier")

# print(df_train[df_train['MonthlyIncome'] > 30000].sort_values(by = 'MonthlyIncome', ascending=False).head(10))

# # We can see that there is some people with age = 0
# print("We can see that there is some people with age = 0")
# df_train = df_train[df_train['age'] != 0]



# ====================================
# 2. ANALYSE OF THE TARGET VARIABLE
# ====================================

def analyze_target(df, target_col='SeriousDlqin2yrs'):
    """Analyse of the target variable"""
    print("=== ANALYSE OF THE TARGET VARIABLE ===\n"*6)
    
    # Distribution
    target_counts = df[target_col].value_counts()
    target_props = df[target_col].value_counts(normalize=True)
    
    # Graphics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Barplot
    target_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
    ax1.set_title('Distribution of the Target Variable')
    ax1.set_xlabel('Serious Delinquency in 2 years')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    # Pie chart
    ax2.pie(target_counts.values, labels=['No Default', 'Default'], 
            autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    ax2.set_title('Classes Repartition')
    
    plt.tight_layout()
    plt.show()
    
    

# ====================================
# 3. UNIVARIATE ANALYSIS
# ====================================

def univariate_analysis(df):

    print("=== UNIVARIATE ANALYSIS ===\n"*6)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SeriousDlqin2yrs' in numeric_cols:
        numeric_cols.remove('SeriousDlqin2yrs')

    if 'ID' in numeric_cols:
        numeric_cols.remove('ID')
      
      
    # Analysis of numerical variables
    print(f"Numerical variables analyzed: {len(numeric_cols)} \n"
        "We actually analyze all the variables.")
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
        
        for i, col in enumerate(numeric_cols[:len(axes)]):
            if i < len(axes):
                # Histogramme
                axes[i].hist(df[col], bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution de {col}')
                axes[i].set_ylabel('Densité')
            
                # Add mean and median lines
                mean_val = df[col].mean()
                median_val = df[col].median()
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                axes[i].legend()
    
        # Remove empty axes
        for i in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    


# ====================================
# 4. BIVARIATE ANALYSIS (vs target)
# ====================================

def bivariate_analysis(df): 
    print("=== BIVARIATE ANALYSIS (vs Target) ===\n"*6)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SeriousDlqin2yrs' in numeric_cols:
        numeric_cols.remove('SeriousDlqin2yrs')
    if 'ID' in numeric_cols:
        numeric_cols.remove('ID')
    
    # Correlations
    print("Correlations with the target variable:")
    correlations = df[numeric_cols + ['SeriousDlqin2yrs']].corr()['SeriousDlqin2yrs'].sort_values(key=abs, ascending=False)
    print(correlations[1:])  # Exclude self-correlation

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_cols + ['SeriousDlqin2yrs']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Distribution by class
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
    
    for i, col in enumerate(numeric_cols[:len(axes)]):
        if i < len(axes):
            # Boxplot by class
            df.boxplot(column=col, by='SeriousDlqin2yrs', ax=axes[i])
            axes[i].set_title(f'{col} by Class')
            axes[i].set_xlabel('Default Status')

    # Remove empty axes
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# ====================================
# 5. FEATURE ENGINEERING
# ====================================

def feature_engineering(df):
    print("=== FEATURE ENGINEERING ===\n"*6)
    
    df_processed = df.copy()
    
    print("Création de nouvelles features...")
    
    # Creation of new features 'TotalDebt'
    df_processed['TotalDebt'] = df_processed['MonthlyIncome'] * df_processed['DebtRatio']
    
    # Age groups
    df_processed['AgeGroup'] = pd.cut(df_processed['age'], 
                                         bins=[0, 30, 50, 65, 100], 
                                         labels=['Young', 'Middle', 'Senior', 'Elderly'])
    # Utilisation of the ratio
    df_processed['HighUtilization'] = (df_processed['RevolvingUtilizationOfUnsecuredLines'] > 0.8).astype(int)
    
    # Log transforms for highly skewed variables
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    skewed_features = []
    
    for col in numeric_cols:
        if (col != 'SeriousDlqin2yrs') & (col != 'ID'):  # Skip target and ID
            skew_val = df_processed[col].skew()
            if abs(skew_val) > 2:  # Highly skewed
                skewed_features.append((col, skew_val))
    
    if skewed_features:
        print("Features avec skew élevé (|skew| > 2):")
        for feat, skew_val in skewed_features:
            print(f"  {feat}: {skew_val:.2f}")
            if (df_processed[feat] > 0).all():  
                df_processed[f'{feat}_log'] = np.log1p(df_processed[feat])


    print(f"Original database: {df.shape[1]} colonnes")
    print(f"News features: {df_processed.shape[1] - df.shape[1]}")
   
    return df_processed


# ====================================
# 6. FONCTION PRINCIPALE
# ====================================

    
"""
print("🚀 DÉBUT DU PIPELINE EDA & PREPROCESSING")
print("="*50)

analyze_target(df=df_train)
df_processed = feature_engineering(df=df_train)
univariate_analysis(df=df_processed)
bivariate_analysis(df=df_processed)


print("="*50)
print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
print(f"Dataset final prêt pour modélisation: {df_processed.shape}")
    


"""