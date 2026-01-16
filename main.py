import joblib
import pandas as pd
import numpy as np
import os
from datetime import date
from flask import Flask, request, jsonify
import warnings
import gdown

warnings.filterwarnings("ignore")
app = Flask(__name__)

# ===============================
# PATHS
# ===============================
MODEL_PATH = "Model/vedaahar_model.pkl"
DRIVE_FILE_ID = "10M3eqFw19ISbJZDYDZNCq4CkN61x2eRt"

VEG_PATH = "Data/Veg_dataset.csv"
FASTING_PATH = "Data/Fasting_dataset.csv"
NONVEG_PATH = "Data/NonVeg_dataset.csv"
HISTORY_PATH = "Data/meal_history.csv"

# ===============================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ===============================


def download_model_from_drive(file_id, destination):
    if os.path.exists(destination):
        print("✅ Model already exists, skipping download")
        return

    print("⬇️ Downloading model from Google Drive using gdown...")

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

    size_mb = os.path.getsize(destination) / (1024 * 1024)
    print(f"📦 Downloaded model size: {size_mb:.2f} MB")

    if size_mb < 50:
        raise RuntimeError("❌ Download failed: model file too small")




# ===============================
# LOAD MODEL (INFERENCE ONLY)
# ===============================
download_model_from_drive(DRIVE_FILE_ID, MODEL_PATH)

bundle = joblib.load(MODEL_PATH)
rf = bundle["model"]
label_encoders = bundle["label_encoders"]
meal_encoder = bundle["meal_encoder"]

print("✅ Model loaded from pkl")

# ===============================
# LOAD DATASETS
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

# 🔥 KEEP ONLY REQUIRED COLUMNS (VERY IMPORTANT)
veg_df = veg_df[["meal_id", "meal_slot"]]
fasting_df = fasting_df[["meal_id", "meal_slot"]]
nonveg_df = nonveg_df[["meal_id", "meal_slot"]]


# ===============================
# UTILS
# ===============================
def cleanup_old_history(days_to_keep=2):
    if not os.path.exists(HISTORY_PATH):
        return

    df = pd.read_csv(HISTORY_PATH)
    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_to_keep)
    df[df["date"] >= cutoff].to_csv(HISTORY_PATH, index=False)

def get_top_k_meals(encoded_row, k=30):
    probs = rf.predict_proba(encoded_row)[0]
    ranked = sorted(zip(rf.classes_, probs), key=lambda x: x[1], reverse=True)
    return [meal_encoder.inverse_transform([cls])[0] for cls, _ in ranked[:k]]

def choose_dataset(day_type):
    if day_type == "fasting":
        return fasting_df
    elif day_type == "nonveg":
        return nonveg_df
    return veg_df

def load_history(user):
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        return df[df["user"] == user] if not df.empty else df
    return pd.DataFrame(columns=["date","prakriti","ritu","goal","meal_slot","meal_id","user"])

def save_history(prakriti, ritu, goal, plan, user):
    today = str(date.today())
    rows = []

    for day, meals in plan.items():
        for slot, meal in meals.items():
            rows.append({
                "date": today,
                "prakriti": prakriti,
                "ritu": ritu,
                "goal": goal,
                "meal_slot": slot,
                "meal_id": meal,
                "user": user
            })

    pd.DataFrame(rows).to_csv(
        HISTORY_PATH,
        mode="a",
        header=not os.path.exists(HISTORY_PATH),
        index=False
    )

# ===============================
# MEAL GENERATOR
# ===============================
def generate_meal_plan(prakriti, ritu, goal, day_type, days, user):
    cleanup_old_history()
    used_meals = set(load_history(user)["meal_id"].tolist())
    plan = {}
    active_df = choose_dataset(day_type)

    for d in range(1, days + 1):
        plan[f"Day {d}"] = {}

        for slot in ["breakfast", "lunch", "dinner"]:
            row = pd.DataFrame([{
                "prakriti": prakriti,
                "ritu": ritu,
                "goal": goal,
                "meal_slot": slot
            }])

            for col in row.columns:
                row[col] = label_encoders[col].transform(row[col])

            ranked = get_top_k_meals(row)
            valid = active_df[active_df["meal_slot"] == slot]["meal_id"].tolist()

            chosen = next((m for m in ranked if m in valid and m not in used_meals), None)
            plan[f"Day {d}"][slot] = chosen or np.random.choice(valid)
            used_meals.add(plan[f"Day {d}"][slot])

    return plan

# ===============================
# API
# ===============================
@app.route("/generate-plan", methods=["POST"])
def generate():
    data = request.get_json()

    plan = generate_meal_plan(
        data["prakriti"].lower(),
        data["ritu"].lower(),
        data["goal"].lower(),
        data.get("day_type", "veg").lower(),
        int(data.get("days", 1)),
        data["username"].lower()
    )

    save_history(data["prakriti"], data["ritu"], data["goal"], plan, data["username"].lower())

    return jsonify({"status": "success", "meal_plan": plan})

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run()
