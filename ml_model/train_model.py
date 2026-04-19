"""
=============================================================
  Random Forest Model for Fake Social Media Profile Detection
=============================================================
Trains a Random Forest classifier on Instagram account features
to predict whether a profile is fake (1) or genuine (0).

Features used:
  - profile pic, nums/length username, fullname words,
    nums/length fullname, name==username, description length,
    external URL, private, #posts, #followers, #follows
  - Engineered: followers/following ratio, posts per follower

Outputs:
  - fake_profile_model.pkl  (trained model)
  - scaler.pkl              (fitted StandardScaler)
  - Console: accuracy, confusion matrix, classification report,
             feature importances
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import joblib

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR, "instagram-fake-spammer-genuine-accounts", "train.csv"
)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "fake_profile_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ── 1. Load Data ──────────────────────────────────────────────
print("=" * 60)
print("  FAKE PROFILE DETECTION – MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n📂 Loaded dataset: {DATA_PATH}")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Rename columns for cleaner access
df.columns = [
    "profile_pic", "ratio_num_username", "fullname_words",
    "ratio_num_fullname", "name_eq_username", "desc_length",
    "external_url", "private", "num_posts", "num_followers",
    "num_following", "fake",
]

print(f"\n📊 Class distribution:")
print(df["fake"].value_counts().to_string())
print(f"   Fake ratio: {df['fake'].mean():.2%}")

# ── 2. Feature Engineering ────────────────────────────────────
print("\n⚙️  Engineering additional features ...")

# Followers / Following ratio (avoid division by zero)
df["followers_following_ratio"] = df["num_followers"] / (
    df["num_following"] + 1
)

# Posts per follower
df["posts_per_follower"] = df["num_posts"] / (df["num_followers"] + 1)

# Has bio (description length > 0)
df["has_bio"] = (df["desc_length"] > 0).astype(int)

# High digit ratio in username (suspicious if > 0.3)
df["high_digit_username"] = (df["ratio_num_username"] > 0.3).astype(int)

print("   ✅ Added: followers_following_ratio, posts_per_follower, has_bio, high_digit_username")

# ── 3. Prepare Features & Target ─────────────────────────────
FEATURE_COLS = [
    "profile_pic", "ratio_num_username", "fullname_words",
    "ratio_num_fullname", "name_eq_username", "desc_length",
    "external_url", "private", "num_posts", "num_followers",
    "num_following", "followers_following_ratio",
    "posts_per_follower", "has_bio", "high_digit_username",
]

X = df[FEATURE_COLS].values
y = df["fake"].values

print(f"\n📐 Feature matrix shape: {X.shape}")
print(f"   Features: {FEATURE_COLS}")

# ── 4. Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🔀 Train/Test split:")
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test:  {X_test.shape[0]} samples")

# ── 5. Scale Features ────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 6. Train Random Forest ───────────────────────────────────
print("\n🌲 Training Random Forest (n_estimators=100) ...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train_scaled, y_train)

# ── 7. Evaluate ──────────────────────────────────────────────
y_pred = rf_model.predict(X_test_scaled)
y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"\n  ✅ Accuracy:  {accuracy:.4f}  ({accuracy:.2%})")
print(f"  📈 ROC-AUC:   {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ── 8. Feature Importances ───────────────────────────────────
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": importances,
}).sort_values("importance", ascending=False)

print("  🔑 Feature Importances (Top 10):")
for _, row in feat_imp.head(10).iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"    {row['feature']:30s} {row['importance']:.4f}  {bar}")

# Save feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x="importance", y="feature", palette="viridis")
plt.title("Feature Importances – Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "feature_importances.png")
plt.savefig(plot_path, dpi=150)
print(f"\n  📊 Feature importance plot saved to: {plot_path}")

# Save confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
print(f"  📊 Confusion matrix plot saved to: {cm_path}")

# ── 9. Save Model & Scaler ───────────────────────────────────
joblib.dump(rf_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# Also save the feature columns list for the backend to use
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_columns.pkl"))

print(f"\n  💾 Model saved to:  {MODEL_PATH}")
print(f"  💾 Scaler saved to: {SCALER_PATH}")
print(f"  💾 Feature columns saved to: {os.path.join(MODEL_DIR, 'feature_columns.pkl')}")

print("\n" + "=" * 60)
print("  ✅ TRAINING COMPLETE!")
print("=" * 60)
