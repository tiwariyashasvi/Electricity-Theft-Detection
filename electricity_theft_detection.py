import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("All libraries loaded successfully!\n")

df = pd.read_csv(r"C:\Users\Lenovo\Documents\ElectricityTheftProject\data.csv")
print(f"Dataset loaded! Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

df.dropna(how="all", inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

label_col = "FLAG"
feature_cols = [c for c in df.columns if c not in [label_col, "CONS_NO"]]

X = df[feature_cols]
y = df[label_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing done!\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Electricity Theft Detection - Data Overview", fontsize=14, fontweight="bold")

counts = y.value_counts()
colors = ["#2ecc71", "#e74c3c"]
axes[0].bar(["Normal (0)", "Theft (1)"], counts.values, color=colors, edgecolor="black", width=0.5)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Number of Consumers")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

df["FLAG"] = y
daily_avg = df.groupby("FLAG")[feature_cols].mean()
axes[1].plot(daily_avg.loc[0].values[:48], color="#2ecc71", label="Normal", linewidth=2)
axes[1].plot(daily_avg.loc[1].values[:48], color="#e74c3c", label="Theft", linewidth=2, linestyle="--")
axes[1].set_title("Average Consumption Pattern (first 48 readings)")
axes[1].set_xlabel("Time Slot")
axes[1].set_ylabel("Avg. Consumption (kWh)")
axes[1].legend()

plt.tight_layout()
plt.savefig("visualization_overview.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 8))
corr_df = pd.DataFrame(X_scaled[:, :20], columns=feature_cols[:20])
sns.heatmap(corr_df.corr(), cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap (first 20 time slots)")
plt.tight_layout()
plt.savefig("visualization_heatmap.png", dpi=150)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

theft_ratio = round(y.mean(), 3)

model = IsolationForest(
    n_estimators=200,
    contamination=theft_ratio,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)
print("Model trained!\n")

raw_preds = model.predict(X_test)
y_pred = np.where(raw_preds == -1, 1, 0)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Theft"]))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Theft (1)"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix - Isolation Forest", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("visualization_confusion_matrix.png", dpi=150)
plt.show()

scores = model.decision_function(X_test)
plt.figure(figsize=(10, 5))
plt.hist(scores[y_test == 0], bins=50, alpha=0.6, color="#2ecc71", label="Normal")
plt.hist(scores[y_test == 1], bins=50, alpha=0.6, color="#e74c3c", label="Theft")
plt.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Decision boundary")
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score (lower = more suspicious)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("visualization_scores.png", dpi=150)
plt.show()

tn, fp, fn, tp = cm.ravel()
accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=" * 55)
print("  PROJECT RESULTS SUMMARY")
print("=" * 55)
print(f"  Algorithm      : Isolation Forest")
print(f"  Training size  : {len(X_train)} samples")
print(f"  Testing size   : {len(X_test)} samples")
print(f"  Contamination  : {theft_ratio}")
print("-" * 55)
print(f"  Accuracy       : {accuracy*100:.2f}%")
print(f"  Precision      : {precision*100:.2f}%")
print(f"  Recall         : {recall*100:.2f}%")
print(f"  F1-Score       : {f1*100:.2f}%")
print("-" * 55)
print(f"  True Positives  : {tp}")
print(f"  True Negatives  : {tn}")
print(f"  False Positives : {fp}")
print(f"  False Negatives : {fn}")
print("=" * 55)
print("\nProject complete! All charts saved to your folder.\n")
