import pandas as pd
import numpy as np
import os
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')
app = Flask(__name__)


def cleanup_old_history(days_to_keep=5):
    """
    Removes meal history records older than `days_to_keep` days
    for each user.
    """

    if not os.path.exists(HISTORY_PATH):
        return

    df = pd.read_csv(HISTORY_PATH)

    if df.empty:
        return

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_to_keep)

    # Keep only recent records
    df_cleaned = df[df["date"] >= cutoff_date]

    # Save back to CSV
    df_cleaned.to_csv(HISTORY_PATH, index=False)

    print(f"🧹 Cleaned meal history (kept last {days_to_keep} days)")


# =====================================================
# API  1
# =====================================================

@app.route('/generate-plan', methods=['POST'])
def handle_meal_plan():

    print("\n--- 🍽️ VedAahar Meal Planner ---")

    data = request.get_json()

    prakriti = data.get('prakriti', '').lower()
    ritu = data.get('ritu', '').lower()
    goal = data.get('goal', '').lower()
    day_type = data.get('day_type', '').lower()
    days = int(data.get('days', 1))
    user = data.get('username', '').lower()

    if not user:
        return jsonify({"error": "username is required"}), 400

    meal_plan = generate_meal_plan(
        prakriti, ritu, goal, day_type, days, user
    )

    print("\n🍱 Final Meal Plan\n")
    for day, meals in meal_plan.items():
        print(day)
        for slot, meal in meals.items():
            print(f"  {slot.capitalize()} → {meal}")

    save_history(prakriti, ritu, goal, meal_plan, user)
    print("\n📝 Meal history saved")

    return jsonify({
        "status": "success",
        "username": user,
        "meal_plan": meal_plan
    }), 200


# =====================================================
# DATA PATHS
# =====================================================

VEG_PATH = "Data/Veg_dataset.csv"
FASTING_PATH = "Data/Fasting_dataset.csv"
NONVEG_PATH = "Data/NonVeg_dataset.csv"
HISTORY_PATH = "Data/meal_history.csv"

os.makedirs("Data", exist_ok=True)

# =====================================================
# LOAD DATASETS
# =====================================================

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

print("✅ Dataset Loaded")
print("Total rows:", len(df))
print("Unique meals:", df["meal_id"].nunique())

# =====================================================
# FEATURE ENCODING
# =====================================================

X = df[["prakriti", "ritu", "goal", "meal_slot"]]
y = df["meal_id"]

label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

meal_encoder = LabelEncoder()
y_enc = meal_encoder.fit_transform(y)

# =====================================================
# MODEL TRAINING
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

print("✅ Model trained")
print("Accuracy:", round(rf.score(X_test, y_test) * 100, 2), "%")

# =====================================================
# UTIL FUNCTIONS
# =====================================================

def get_top_k_meals(encoded_row, k=30):
    probs = rf.predict_proba(encoded_row)[0]
    classes = rf.classes_

    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    return [
        meal_encoder.inverse_transform([cls])[0]
        for cls, _ in ranked[:k]
    ]

def choose_dataset(day_type):
    if day_type == "fasting":
        return fasting_df
    elif day_type == "nonveg":
        return nonveg_df
    else:
        return veg_df

# =====================================================
# HISTORY (USER ISOLATED)
# =====================================================

def load_history(user):
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        if df.empty:
            return df
        return df[df["user"] == user]
    else:
        return pd.DataFrame(columns=[
            "date","prakriti","ritu","goal","meal_slot","meal_id","user"
        ])

def save_history(prakriti, ritu, goal, meal_plan, user):
    today = str(date.today())
    rows = []

    for day, meals in meal_plan.items():
        for slot, meal in meals.items():
            if meal != "no_valid_meal":
                rows.append({
                    "date": today,
                    "prakriti": prakriti,
                    "ritu": ritu,
                    "goal": goal,
                    "meal_slot": slot,
                    "meal_id": meal,
                    "user": user
                })

    if not rows:
        return

    df_new = pd.DataFrame(rows)

    if os.path.exists(HISTORY_PATH):
        df_new.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(HISTORY_PATH, index=False)

# =====================================================
# MEAL PLANNER (USER-AWARE)
# =====================================================

def generate_meal_plan(prakriti, ritu, goal, day_type, days, user):
    """ Clean old history of user"""
    cleanup_old_history(2)
    history_df = load_history(user)
    used_meals = set(history_df["meal_id"].tolist())

    plan = {}
    active_df = choose_dataset(day_type)

    for day in range(1, days + 1):
        plan[f"Day {day}"] = {}

        for slot in ["breakfast", "lunch", "dinner"]:
            row = {
                "prakriti": prakriti,
                "ritu": ritu,
                "goal": goal,
                "meal_slot": slot
            }

            enc = pd.DataFrame([row])
            for col in enc.columns:
                enc[col] = label_encoders[col].transform(enc[col])

            ranked_meals = get_top_k_meals(enc, k=30)

            valid_meals = active_df[
                active_df["meal_slot"] == slot
            ]["meal_id"].tolist()

            chosen = None
            for meal in ranked_meals:
                if meal in valid_meals and meal not in used_meals:
                    chosen = meal
                    used_meals.add(meal)
                    break

            # fallback (important)
            if chosen is None and valid_meals:
                chosen = np.random.choice(valid_meals)

            plan[f"Day {day}"][slot] = chosen or "no_valid_meal"

    return plan


# =====================================================
# RUN
# =====================================================

if __name__ == '__main__':
    app.run(debug=True)
