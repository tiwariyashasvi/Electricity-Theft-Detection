import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv("data.csv")

print("Shape:", df.shape)
print("\nClass Distribution:")
print(df['FLAG'].value_counts())
print("\nTotal Missing Values:", df.isnull().sum().sum())

missing_percentage = df.isnull().mean(axis=1) * 100
df_clean = df[missing_percentage <= 50].copy()

print("\nOriginal Shape:", df.shape)
print("Cleaned Shape:", df_clean.shape)

consumption_cols = df_clean.columns[2:]
normal = df_clean[df_clean['FLAG'] == 0]
theft = df_clean[df_clean['FLAG'] == 1]

fig, axes = plt.subplots(2, 1, figsize=(14, 6))
axes[0].plot(normal.iloc[0][consumption_cols].values, color='steelblue')
axes[0].set_title("Normal Consumer - Daily Consumption")
axes[0].set_xlabel("Days")
axes[0].set_ylabel("kWh")
axes[1].plot(theft.iloc[0][consumption_cols].values, color='crimson')
axes[1].set_title("Theft Consumer - Daily Consumption")
axes[1].set_xlabel("Days")
axes[1].set_ylabel("kWh")
plt.tight_layout()
plt.show()

df_clean = df_clean.drop("CONS_NO", axis=1)
consumption = df_clean.drop("FLAG", axis=1)

features_df = pd.concat([
    consumption.mean(axis=1).rename("avg_consumption"),
    consumption.max(axis=1).rename("max_consumption"),
    consumption.min(axis=1).rename("min_consumption"),
    consumption.std(axis=1).rename("std_consumption"),
    consumption.median(axis=1).rename("median_consumption"),
    (consumption == 0).sum(axis=1).rename("zero_days"),
    consumption.isnull().sum(axis=1).rename("missing_days"),
    (consumption.max(axis=1) - consumption.min(axis=1)).rename("range_consumption"),
], axis=1)

features_df["peak_to_avg_ratio"] = features_df["max_consumption"] / (features_df["avg_consumption"] + 1e-5)

print("\nFeatures shape:", features_df.shape)
print(features_df.head())

X = features_df
y = df_clean["FLAG"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_sm).value_counts().to_dict())

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_sm, y_train_sm)
print("\nModel Trained Successfully!")

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Theft"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["Normal", "Theft"], yticklabels=["Normal", "Theft"], cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC AUC = {roc_auc_score(y_test, y_pred_prob):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Electricity Theft Detection")
plt.legend()
plt.tight_layout()
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance["Feature"], importance["Importance"], color='steelblue')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

joblib.dump(model, "electricity_theft_model.joblib")
print("\nModel saved as electricity_theft_model.joblib")