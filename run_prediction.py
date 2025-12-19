import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from utils import load_dataset, FEATURES, build_future_match
from model_result import train_match_result_model
from model_over_under import train_over_under_model


print("Running prediction pipeline...\n")

team_name = "Wolves"
opponent_name = "Brentford"

with tqdm(total=5, desc="Pipeline progress") as pbar:

    # =========================
    # 1. LOAD DATA
    # =========================
    df = load_dataset("dataset/dataset_master_football.xlsx")
    pbar.update(1)

    # =========================
    # 2. TRAIN MODELS
    # =========================
    result_pack = train_match_result_model(df)
    ou_models = {
        2.5: train_over_under_model(df, 2.5),
        3.5: train_over_under_model(df, 3.5),
        4.5: train_over_under_model(df, 4.5),
    }
    pbar.update(1)

    # unpack
    model_res = result_pack["model"]
    scaler_res = result_pack["scaler"]
    le = result_pack["label_encoder"]
    X_test = result_pack["X_test"]
    y_test = result_pack["y_test"]
    coef_df = result_pack["coef_df"]

    # =========================
    # 3. EVALUATION
    # =========================
    y_pred = model_res.predict(X_test)
    pbar.update(1)

    # =========================
    # 4. FUTURE MATCH
    # =========================
    future_match = build_future_match(df, team_name, opponent_name)
    future_scaled = scaler_res.transform(future_match)
    future_pred = model_res.predict(future_scaled)
    future_prob = model_res.predict_proba(future_scaled)
    pbar.update(1)

    # =========================
    # 5.OVER / UNDER PROBABILITY
    # =========================
    ou_results = {}

    for th, pack in ou_models.items():
        X_scaled = pack["scaler"].transform(future_match)
        prob_under, prob_over = pack["model"].predict_proba(X_scaled)[0]
        ou_results[th] = (prob_under, prob_over)

    pbar.update(1)


# =========================
# OUTPUT (SETELAH PROGRESS SELESAI)
# =========================

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== FEATURE WEIGHTS ===")
print(coef_df)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))


print("\n=== FUTURE MATCH PREDICTION ===")
print(f"Predicted Team: {team_name} VS {opponent_name}")
print("Predicted Result:", le.inverse_transform(future_pred)[0])
print("Probabilities:")
for cls, prob in zip(le.classes_, future_prob[0]):
    print(f"  {cls}: {prob:.2f}")

print("\n=== PROBABILITIES ===")
for cls, prob in zip(le.classes_, future_prob[0]):
    print(f"  {cls}: {prob:.2f}")

print("\n=== OVER / UNDER PROBABILITIES ===")
for th, (u, o) in ou_results.items():
    print(f"Over {th}: {o:.3f} | Under {th}: {u:.3f}")