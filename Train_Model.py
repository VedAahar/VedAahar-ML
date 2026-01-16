import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
# FEATURE ENCODING
# ===============================
X = df[["prakriti", "ritu", "goal", "meal_slot"]]
y = df["meal_id"]

label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

meal_encoder = LabelEncoder()
y_enc = meal_encoder.fit_transform(y)

# ===============================
# TRAIN MODEL
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=70,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

print("✅ Model trained")
print("Accuracy:", round(rf.score(X_test, y_test) * 100, 2), "%")

# ===============================
# SAVE EVERYTHING
# ===============================
joblib.dump(
    {
        "model": rf,
        "label_encoders": label_encoders,
        "meal_encoder": meal_encoder
    },
    MODEL_PATH
)

print(f"✅ Model saved at: {MODEL_PATH}")
