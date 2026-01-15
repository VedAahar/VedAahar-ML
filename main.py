import joblib
import pandas as pd
import numpy as np
import os
from datetime import date
from flask import Flask, request, jsonify
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

# ===============================
# PATHS
# ===============================
MODEL_PATH = "models/vedaahar_model.pkl"
VEG_PATH = "Data/Veg_dataset.csv"
FASTING_PATH = "Data/Fasting_dataset.csv"
NONVEG_PATH = "Data/NonVeg_dataset.csv"
HISTORY_PATH = "Data/meal_history.csv"

# ===============================
# LOAD MODEL
# ===============================
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
