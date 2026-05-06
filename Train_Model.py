import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ===============================
# PATHS
# ===============================
VEG_PATH = "Data/Veg_dataset.csv"
FASTING_PATH = "Data/Fasting_dataset.csv"
NONVEG_PATH = "Data/NonVeg_dataset.csv"
MODEL_PATH = "Model/vedaahar_model.pkl"

os.makedirs("Model", exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
veg_df = pd.read_csv(VEG_PATH)
fasting_df = pd.read_csv(FASTING_PATH)
nonveg_df = pd.read_csv(NONVEG_PATH)

def normalize(df):
    for col in ["prakriti", "ritu", "goal", "meal_slot", "meal_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df

veg_df = normalize(veg_df)
fasting_df = normalize(fasting_df)
nonveg_df = normalize(nonveg_df)

df = pd.concat([veg_df, fasting_df, nonveg_df], ignore_index=True)

print("✅ Dataset loaded")
print("Rows:", len(df))
print("Unique meals:", df["meal_id"].nunique())

# ===============================
# FEATURES & TARGET
# ===============================
X = df[["prakriti", "ritu", "goal", "meal_slot"]]
y = df["meal_id"]

# ✅ OneHot Encoding (FIXED)
X = pd.get_dummies(X)

# Encode target
meal_encoder = LabelEncoder()
y_enc = meal_encoder.fit_transform(y)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# ===============================
# MODEL
# ===============================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

print("✅ Model trained")

# ===============================
# EVALUATION (FIXED)
# ===============================
accuracy = rf.score(X_test, y_test)

probs = rf.predict_proba(X_test)

# ✅ Keep only valid classes
valid_mask = [y in rf.classes_ for y in y_test]

y_test_filtered = y_test[valid_mask]
probs_filtered = probs[valid_mask]

top5_acc = top_k_accuracy_score(
    y_test_filtered,
    probs_filtered,
    k=30,
    labels=rf.classes_
)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Top-30 Accuracy:", round(top5_acc * 100, 2), "%")

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(
    {
        "model": rf,
        "columns": X.columns,   # 🔥 VERY IMPORTANT
        "meal_encoder": meal_encoder
    },
    MODEL_PATH
)

print(f"✅ Model saved at: {MODEL_PATH}")