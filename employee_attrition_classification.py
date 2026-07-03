
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, roc_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ─────────────────────────────────────────────────────────────────
# SECTION 1: SETUP
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  EMPLOYEE ATTRITION PREDICTION")
print("=" * 65)

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Dataset path — place the CSV in the same folder as this script
data_path = os.path.join(current_dir, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"\n❌ Dataset not found at:\n   {data_path}\n"
        "   Download from Kaggle and place it in the same folder as this script."
    )

df = pd.read_csv(data_path)
print(f"\n✅ Loaded dataset: {df.shape[0]} employees, {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 2: EXPLORATORY DATA ANALYSIS")
print("─" * 65)

print("\n📋 Dataset Info:")
print(df.dtypes.value_counts())

print("\n🔍 Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  No missing values found!")

print("\n📊 Target Distribution (Attrition):")
print(df['Attrition'].value_counts())
attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
print(f"\n  → Attrition rate: {attrition_rate:.1f}%")
print(f"  → This is an IMBALANCED dataset — we'll handle it with class_weight='balanced'")

# ── EDA Plot 1: Attrition by Department & Job Role ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Attrition Analysis by Department & Job Role', fontsize=14, fontweight='bold')

dept_attrition = df.groupby('Department')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index(name='AttritionRate')
sns.barplot(data=dept_attrition, x='Department', y='AttritionRate',
            palette='Reds_d', ax=axes[0])
axes[0].set_title('Attrition Rate by Department (%)')
axes[0].set_ylabel('Attrition Rate (%)')
axes[0].set_xlabel('')
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.1f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10)

role_attrition = df.groupby('JobRole')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index(name='AttritionRate').sort_values('AttritionRate', ascending=True)
sns.barplot(data=role_attrition, y='JobRole', x='AttritionRate',
            palette='Oranges_d', ax=axes[1])
axes[1].set_title('Attrition Rate by Job Role (%)')
axes[1].set_xlabel('Attrition Rate (%)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '01_attrition_by_dept_role.png'), dpi=150)
plt.show()
print("  📸 Saved: 01_attrition_by_dept_role.png")

# ── EDA Plot 2: Key Numeric Distributions ───────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Numeric Feature Distributions by Attrition', fontsize=14, fontweight='bold')

numeric_cols_eda = ['Age', 'MonthlyIncome', 'YearsAtCompany',
                    'DistanceFromHome', 'TotalWorkingYears', 'JobSatisfaction']

for i, col in enumerate(numeric_cols_eda):
    ax = axes[i // 3][i % 3]
    df[df['Attrition'] == 'No'][col].hist(bins=20, alpha=0.6, label='Stay', color='steelblue', ax=ax)
    df[df['Attrition'] == 'Yes'][col].hist(bins=20, alpha=0.6, label='Leave', color='tomato', ax=ax)
    ax.set_title(col)
    ax.legend(fontsize=8)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '02_numeric_distributions.png'), dpi=150)
plt.show()
print("  📸 Saved: 02_numeric_distributions.png")

# ── EDA Plot 3: OverTime & Work-Life Balance impact ──────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('OverTime & Work-Life Balance vs Attrition', fontsize=14, fontweight='bold')

ot_counts = df.groupby(['OverTime', 'Attrition']).size().unstack()
ot_counts.plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'], edgecolor='white')
axes[0].set_title('Attrition Count by OverTime')
axes[0].set_xlabel('OverTime')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(['Stay', 'Leave'])

wlb_attrition = df.groupby('WorkLifeBalance')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index(name='AttritionRate')
sns.barplot(data=wlb_attrition, x='WorkLifeBalance', y='AttritionRate',
            palette='coolwarm', ax=axes[1])
axes[1].set_title('Attrition Rate by Work-Life Balance (1=Bad, 4=Best)')
axes[1].set_xlabel('Work-Life Balance Score')
axes[1].set_ylabel('Attrition Rate (%)')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '03_overtime_worklife.png'), dpi=150)
plt.show()
print("  📸 Saved: 03_overtime_worklife.png")

# ─────────────────────────────────────────────────────────────────
# SECTION 3: DATA CLEANING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 3: DATA CLEANING & FEATURE ENGINEERING")
print("─" * 65)

df_model = df.copy()

# Drop columns with no variance (IBM dataset includes these)
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df_model.drop(columns=[c for c in cols_to_drop if c in df_model.columns], inplace=True)
print(f"\n  Dropped constant/ID columns: {cols_to_drop}")

# Encode target variable
df_model['Attrition'] = (df_model['Attrition'] == 'Yes').astype(int)

# Feature Engineering — create new meaningful features
df_model['IncomePerYear']         = df_model['MonthlyIncome'] / (df_model['YearsAtCompany'] + 1)
df_model['SatisfactionScore']     = (df_model['JobSatisfaction'] +
                                      df_model['EnvironmentSatisfaction'] +
                                      df_model['RelationshipSatisfaction']) / 3
df_model['CareerGrowthRate']      = df_model['YearsAtCompany'] / (df_model['TotalWorkingYears'] + 1)
df_model['PromotionLag']          = df_model['YearsAtCompany'] - df_model['YearsSinceLastPromotion']
df_model['OverTimeBinary']        = (df_model['OverTime'] == 'Yes').astype(int)

print("  ✅ Engineered new features:")
print("     • IncomePerYear         (monthly income relative to tenure)")
print("     • SatisfactionScore     (average of job/env/relationship satisfaction)")
print("     • CareerGrowthRate      (years at company vs total experience)")
print("     • PromotionLag          (years since last promotion vs tenure)")
print("     • OverTimeBinary        (0/1 encoding of OverTime)")

# Identify categorical columns (excluding OverTime — already encoded above)
categorical_cols = df_model.select_dtypes(include='object').columns.tolist()
categorical_cols = [c for c in categorical_cols if c != 'OverTime']
print(f"\n  Encoding categorical columns: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Drop original OverTime (already binary encoded)
if 'OverTime' in df_model.columns:
    df_model.drop(columns=['OverTime'], inplace=True)

print(f"\n  Final dataset shape: {df_model.shape}")

# ─────────────────────────────────────────────────────────────────
# SECTION 4: MODEL BUILDING
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 4: MODEL BUILDING")
print("─" * 65)

X = df_model.drop(columns=['Attrition'])
y = df_model['Attrition']

print(f"\n  Features: {X.shape[1]} columns")
print(f"  Target class balance — Stay: {(y==0).sum()}, Leave: {(y==1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ── Compare 3 models quickly ────────────────────────────────────
print("\n  🔄 Comparing base models (5-fold CV AUC)...")
models_compare = {
    'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
compare_results = {}
for name, model in models_compare.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    compare_results[name] = scores.mean()
    print(f"     {name:<25} AUC = {scores.mean():.4f} ± {scores.std():.4f}")

best_model_name = max(compare_results, key=compare_results.get)
print(f"\n  🏆 Best base model: {best_model_name} (AUC = {compare_results[best_model_name]:.4f})")

# ── Hyperparameter Tuning on Random Forest ───────────────────────
print("\n  ⚙️  Tuning Random Forest with GridSearchCV...")
param_grid = {
    'n_estimators':     [100, 200],
    'max_depth':        [10, 20, None],
    'min_samples_split':[2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight':     ['balanced']   # important for imbalanced data
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"  ✅ Best parameters: {grid_search.best_params_}")
print(f"  ✅ Best CV AUC:     {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ─────────────────────────────────────────────────────────────────
# SECTION 5: MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 5: MODEL EVALUATION")
print("─" * 65)

y_pred       = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_pred_proba)

print(f"""
  ┌──────────────────────────────────────┐
  │         FINAL MODEL METRICS          │
  ├──────────────────────────────────────┤
  │  Accuracy   : {accuracy:.4f}  ({accuracy*100:.2f}%)        │
  │  Precision  : {precision:.4f}                    │
  │  Recall     : {recall:.4f}                    │
  │  F1 Score   : {f1:.4f}                    │
  │  AUC-ROC    : {auc:.4f}                    │
  └──────────────────────────────────────┘
""")

print("  📋 Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

# ── Plot: Confusion Matrix + ROC Curve ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stay', 'Leave'])
disp.plot(cmap='Blues', values_format='d', ax=axes[0])
axes[0].set_title('Confusion Matrix')

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='darkorange', linewidth=2, label=f'Random Forest (AUC = {auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.08, color='darkorange')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '04_confusion_matrix_roc.png'), dpi=150)
plt.show()
print("  📸 Saved: 04_confusion_matrix_roc.png")

# ─────────────────────────────────────────────────────────────────
# SECTION 6: FEATURE IMPORTANCE & BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 6: FEATURE IMPORTANCE & BUSINESS INSIGHTS")
print("─" * 65)

feature_importance = pd.DataFrame({
    'Feature':    X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\n  🔝 Top 15 Features Driving Attrition:")
print(feature_importance.head(15).to_string(index=False))

# Plot top 15 features
top15 = feature_importance.head(15)
plt.figure(figsize=(10, 7))
colors = ['#d73027' if i < 5 else '#fc8d59' if i < 10 else '#fee090'
          for i in range(len(top15))]
bars = plt.barh(top15['Feature'][::-1], top15['Importance'][::-1], color=colors[::-1], edgecolor='white')
plt.xlabel('Feature Importance Score')
plt.title('Top 15 Features Driving Employee Attrition', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '05_feature_importance.png'), dpi=150)
plt.show()
print("  📸 Saved: 05_feature_importance.png")

# Business Insights
print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                    BUSINESS INSIGHTS                        ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  Top attrition driver: {feature_importance.iloc[0]['Feature']:<36} ║
  ║                                                              ║
  ║  📊 Model Performance Summary:                               ║
  ║     • Correctly identifies {recall*100:.1f}% of employees who leave  ║
  ║     • When it flags "will leave", correct {precision*100:.1f}% of time ║
  ║     • Overall AUC: {auc:.3f} — {"Excellent ✅" if auc>0.8 else "Good 👍" if auc>0.7 else "Needs work ⚠️":<30} ║
  ║                                                              ║
  ║  💡 HR Recommendations:                                      ║
  ║     1. Focus retention efforts on high-risk employees         ║
  ║     2. Review overtime policies — strong attrition signal     ║
  ║     3. Monthly income & satisfaction are key levers           ║
  ║     4. Monitor employees with long promotion gaps             ║
  ╚══════════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────────────────────────
# SECTION 7: PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────
def predict_attrition(employee_data: dict) -> dict:
    """
    Predict whether a single employee will leave.

    Parameters:
        employee_data (dict): A dictionary of feature values matching
                              the training columns.

    Returns:
        dict: {'prediction': 'Leave'/'Stay', 'probability': float}

    Example usage:
        result = predict_attrition({
            'Age': 32,
            'MonthlyIncome': 4500,
            'OverTimeBinary': 1,
            'YearsAtCompany': 2,
            'JobSatisfaction': 2,
            ...
        })
    """
    input_df = pd.DataFrame([employee_data])

    # Add any missing engineered features if raw inputs provided
    if 'IncomePerYear' not in input_df.columns and 'MonthlyIncome' in input_df.columns:
        input_df['IncomePerYear'] = input_df['MonthlyIncome'] / (input_df.get('YearsAtCompany', 1) + 1)
    if 'SatisfactionScore' not in input_df.columns:
        sat_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']
        if all(c in input_df.columns for c in sat_cols):
            input_df['SatisfactionScore'] = input_df[sat_cols].mean(axis=1)

    # Align columns to training set
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0   # default for missing features

    input_df = input_df[X.columns]

    proba = best_model.predict_proba(input_df)[0, 1]
    prediction = 'Leave' if proba >= 0.5 else 'Stay'

    return {'prediction': prediction, 'probability': round(proba, 4)}


# Demo prediction
print("  🔮 Example Prediction:")
example = {col: X_test.iloc[0][col] for col in X.columns}
result = predict_attrition(example)
actual = 'Leave' if y_test.iloc[0] == 1 else 'Stay'
print(f"     Actual:    {actual}")
print(f"     Predicted: {result['prediction']} (probability of leaving: {result['probability']:.2%})")

# ─────────────────────────────────────────────────────────────────
# SECTION 8: SAVE MODEL
# ─────────────────────────────────────────────────────────────────
model_path = os.path.join(current_dir, 'attrition_model.pkl')
joblib.dump({
    'model':           best_model,
    'encoders':        label_encoders,
    'feature_columns': list(X.columns),
    'metrics': {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'auc':       auc
    }
}, model_path)

print(f"\n  💾 Model saved to: {model_path}")
print("\n" + "=" * 65)
print("  ✅ PROJECT COMPLETE — All outputs saved to 'results/' folder")
print("=" * 65)