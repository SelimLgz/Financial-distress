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

# ====================================
# 1. LOAD DATA
# ====================================

def load_data_correct(df, df_val, target_col='SeriousDlqin2yrs'):
    print("=== LOAD DATA ===")
    
 
    
    # Split of the training set into train (80%) and test (20%)
    X_train_full = df.drop(columns=[target_col])
    y_train_full = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_full
    )
    
    # Final validation set (from df_val)
    X_validation = df_val.drop(columns=[target_col])
    y_validation = df_val[target_col]
    
    print(f"\nAfter split:")
    print(f"Train (80%): {X_train.shape[0]} - Defaults: {y_train.sum()}/{len(y_train)} ({y_train.mean():.1%})")
    print(f"Test (20%): {X_test.shape[0]} - Defaults: {y_test.sum()}/{len(y_test)} ({y_test.mean():.1%})")
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation


# ====================================
# 2. IMBALANCE HANDLING
# ====================================

def handle_imbalance(X_train, y_train, method='smote'):
    print(f"\n=== IMBALANCE HANDLING ({method.upper()}) ===")
    
    # Initial analysis
    counts = pd.Series(y_train).value_counts()
    ratio = counts[0] / counts[1]
    counts = pd.Series(y_train).value_counts()
    counts_dict = {int(k): int(v) for k, v in counts.items()}
    print(f"Initial distribution: {counts_dict}")
    print(f"Imbalance ratio: {ratio:.1f}:1")
    
    # If not too imbalanced, keep as is
    if ratio < 3:
        print("Dataset not too imbalanced, no resampling")
        return X_train, y_train

    # Resampling application
    if method == 'smote':
        resampler = SMOTE(random_state=42)
    elif method == 'smote_tomek':
        resampler = SMOTETomek(random_state=42)
    else:
        print("Method not recognized, no resampling")
        return X_train, y_train
    
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

    # Results
    new_counts = pd.Series(y_resampled).value_counts()
    new_counts_dict = {int(k): int(v) for k, v in new_counts.items()}
    print(f"Distribution after: {new_counts_dict}")
    print(f"New data: {X_train.shape[0]} → {X_resampled.shape[0]}")
    
    return X_resampled, y_resampled

# ====================================
# 3. NORMALIZATION
# ====================================

def normalize_data(X_train, X_test, X_validation, method='robust'):
    """Normalize the data"""
    print("\n=== NORMALIZATION ===")
    
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # Fit only on train, transform on all
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_validation_scaled = scaler.transform(X_validation)

    print("✅ Normalization complete (fit on train only)")
    
    return X_train_scaled, X_test_scaled, X_validation_scaled, scaler
# ====================================
# 4. MODELS WITH MULTIPLE PARAMETERS
# ====================================

def create_models_variants():
    print("\n=== CREATING MODELS (MULTIPLE VARIANTS) ===")
    
    models = {
        # Logistic Regression - 2 variantes
        'Logistic_Simple': LogisticRegression(random_state=42, max_iter=1000),
        'Logistic_L1': LogisticRegression(random_state=42, penalty='l1', solver='liblinear', C=0.1),
        
        # Random Forest - 3 variantes
        'RandomForest_100': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'RandomForest_200': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15),
        'RandomForest_500': RandomForestClassifier(n_estimators=500, random_state=42, max_depth=20),
        
        # XGBoost - 3 variantes
        'XGBoost_Conservative': xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, 
            random_state=42, eval_metric='logloss'
        ),
        'XGBoost_Aggressive': xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.2, max_depth=6,
            random_state=42, eval_metric='logloss'
        ),
        'XGBoost_Balanced': xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=42, eval_metric='logloss'
        ),
        
        # LightGBM - 2 variantes
        'LightGBM_Fast': lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=42, verbose=-1
        ),
        'LightGBM_Deep': lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=42, verbose=-1
        )
    }
    
    print(f"Created models: {len(models)} variants")
    for name in models.keys():
        print(f"  - {name}")
    
    return models

# ====================================
# 5. MODELS TRAINING 
# ====================================

def train_models_variants(models, X_train, y_train, X_test, y_test):
    print(f"\n=== TRAINING OF {len(models)} MODELS ===")
    print("(Training on TRAIN, evaluating on TEST)")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        try:
            # 1. Training on train
            model.fit(X_train, y_train)

            # 2. Predictions on test (not validation!)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # 3. Calculate metrics on test
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # 4. Stockage
            results[name] = {
                'model': model,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred
            }
            
            print(f"  ✅ AUC: {auc:.3f} | F1: {f1:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    print(f"\n✅ {len(results)} trained models with results")
    return results

# ====================================
# 5. MODEL COMPARISON
# ====================================

def compare_models(results):
    print("\n=== MODEL COMPARISON ===")
    
    # Comparison DataFrame
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'AUC': f"{metrics['auc']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('AUC', ascending=False)

    print("\n📊 TABLE OF PERFORMANCE:")
    print(comparison_df.to_string(index=False))

    # Best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_auc = comparison_df.iloc[0]['AUC']

    print(f"\n🏆 BEST MODEL: {best_model_name} (AUC = {best_auc})")

    return best_model_name, comparison_df

# ====================================
# 6. VISUALIZATIONS
# ====================================

def plot_model_comparison(y_val, results):
    print("\n=== VISUALIZATIONS ===")

    # 1. ROC Curves and AUC Barplot
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: ROC
    plt.subplot(1, 2, 1)
    colors = ['blue', 'orange', 'green', 'red', 'black', 'brown']
    
    for i, (name, metrics) in enumerate(results.items()):
        y_pred_proba = metrics['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc = metrics['auc']
        
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', 
                color=colors[i], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbes ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: AUC Barplot
    plt.subplot(1, 2, 2)
    model_names = list(results.keys())
    auc_scores = [results[name]['auc'] for name in model_names]
    
    bars = plt.bar(model_names, auc_scores, color=colors[:len(model_names)])
    plt.ylabel('AUC Score')
    plt.title('Comparaison AUC')
    plt.ylim(0.5, 1.0)
    
    # Add scores on top of bars
    for bar, score in zip(bars, auc_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ====================================
# 7. ANALYZE BEST MODEL
# ====================================

def analyze_best_model(best_model_name, results, y_val):
    print(f"\n=== DETAILED ANALYSIS - {best_model_name} ===")
    
    best_result = results[best_model_name]
    model = best_result['model']
    y_pred = best_result['y_pred']
    
    # 1. Report of classification
    print("\n📋 REPORT OF CLASSIFICATION:")
    print(classification_report(y_val, y_pred))
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    plt.figure(figsize=(10, 4))
    
    # Confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Default', 'DDefault'],
               yticklabels=['No Default', 'DDefault'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Reality')
    plt.xlabel('Prediction')
    
    # Distribution of scores
    plt.subplot(1, 2, 2)
    y_pred_proba = best_result['y_pred_proba']

    # Histograms of probabilities by class
    plt.hist(y_pred_proba[y_val == 0], bins=30, alpha=0.7, label='No Default', color='blue')
    plt.hist(y_pred_proba[y_val == 1], bins=30, alpha=0.7, label='DDefault', color='red')
    plt.xlabel('Probability of Default')
    plt.ylabel('Number of Clients')
    plt.title('Distribution of Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # 3. Analyze business
    tn, fp, fn, tp = cm.ravel()

    print(f"\n💼 ANALYZE BUSINESS:")
    print(f"  True Negatives (TN): {tn} - Correctly identified solvent clients")
    print(f"  False Positives (FP): {fp} - Rejected solvent clients (business loss)")
    print(f"  False Negatives (FN): {fn} - Accepted risky clients (losses)")
    print(f"  True Positives (TP): {tp} - Correctly identified risky clients")

    return model

# ====================================
# 8. FEATURE IMPORTANCE
# ====================================

def show_feature_importance(model, feature_names, model_name, top_n=10):
    print(f"\n=== FEATURE IMPORTANCE - {model_name} ===")
    
    # Extraction of importances
    if hasattr(model, 'feature_importances_'):
        # Pour Random Forest, XGBoost, LightGBM
        importances = model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(model, 'coef_'):
        # For Logistic Regression
        importances = np.abs(model.coef_[0])
        importance_type = "Coefficient (absolute value)"
    else:
        print("❌ This model does not support importance analysis")
        return
    
    # Creation of DataFrame
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    # Display
    print(f"\n📊 TOP {top_n} IMPORTANT FEATURES:")
    for i, row in feature_imp_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Graphique
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_imp_df)), feature_imp_df['Importance'])
    plt.yticks(range(len(feature_imp_df)), feature_imp_df['Feature'])
    plt.xlabel(importance_type)
    plt.title(f'Top {top_n} Features - {model_name}')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ====================================
# 9. SAVE BEST MODEL
# ====================================




def save_predict_validation(best_model, X_validation, df_val):
    print("\n=== SAVING PREDICTIONS ON VALIDATION SET ===")

    y_pred = best_model.predict(X_validation)
    # Creation of the submission DataFrame
    submission = pd.DataFrame({
        'ID': df_val['ID'],
        'SeriousDlqin2yrs': y_pred
    })

    # Save to CSV format (without index)
    submission.to_csv('submission.csv', index=False)
    print("✅ Submission file created: submission.csv")


def save_best_model(pipeline_results, filename='best_credit_model.pkl'):
    """Sauvegarde le meilleur modèle"""
    import pickle
    
    model_package = {
        'model': pipeline_results['best_model'],
        'scaler': pipeline_results['scaler'],
        'model_name': pipeline_results['best_model_name'],
        'feature_names': pipeline_results['feature_names']
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"💾 Modèle sauvegardé: {filename}")


def run_credit_scoring_complete(train_file, validation_file, target_col='SeriousDlqin2yrs',
                               handle_imbalance_method='smote'):
    
    print("🚀 PIPELINE CREDIT SCORING COMPLET")
    print("="*60)
    
    # 1. Load data
    X_train, X_test, X_validation, y_train, y_test, y_validation = load_data_correct(
        train_file, validation_file, target_col
    )
    
    # 2. Normalization
    X_train_scaled, X_test_scaled, X_validation_scaled, scaler = normalize_data(
        X_train, X_test, X_validation
    )

    # 3. Handling imbalance 
    if handle_imbalance_method:
        X_train_balanced, y_train_balanced = handle_imbalance(
            X_train_scaled, y_train, handle_imbalance_method
        )
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        print("\n⚠️  No imbalance handling applied")

    # 4. Creation of models with variants
    models = create_models_variants()

    # 5. Training (train) and evaluation 
    results =  train_models_variants(models, X_train_balanced, y_train_balanced, X_test_scaled, y_test)

    # 6. Comparison and selection of the best
    best_model_name, comparison_df = compare_models(results)

    # 7. Visualisations
    top_results = dict(sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)[:6])
    plot_model_comparison(y_test, top_results)

    # 8. Detailed analysis of the best
    best_model = analyze_best_model(best_model_name, results, y_test)
    
    # 9. Feature importance
    show_feature_importance(best_model, X_train.columns, best_model_name)
    
    # 10. Save the prediction
    save_predict_validation(best_model, X_validation_scaled, validation_file)
    
    print("="*60)
    print("✅ PIPELINE COMPLETE DONE")
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'all_results': results,
        'top_results': top_results,
        'scaler': scaler,
        'comparison': comparison_df,
        'feature_names': list(X_train.columns),
        'imbalance_method': handle_imbalance_method
    }
