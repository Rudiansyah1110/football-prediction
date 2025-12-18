import pandas as pd

# =========================
# 1. LOAD MASTER DATA
# =========================
master_path = "dataset/dataset_master_football.xlsx"
df = pd.read_excel(master_path)

# =========================
# 2. SORT KRONOLOGIS
# =========================
df["match_date"] = pd.to_datetime(df["match_date"])
df = df.sort_values("match_date").reset_index(drop=True)

# =========================
# 3. HITUNG ROLLING TEAM (5 MATCH SEBELUMNYA)
# =========================
df["team_avg_goals_for_5"] = (
    df.groupby("team")["team_goals"]
    .shift(1)
    .rolling(5)
    .mean()
)

df["team_avg_goals_against_5"] = (
    df.groupby("team")["opp_goals"]
    .shift(1)
    .rolling(5)
    .mean()
)

# =========================
# 4. HITUNG ROLLING OPPONENT
# =========================
df["opp_avg_goals_for_5"] = (
    df.groupby("opponent")["opp_goals"]
    .shift(1)
    .rolling(5)
    .mean()
)

df["opp_avg_goals_against_5"] = (
    df.groupby("opponent")["team_goals"]
    .shift(1)
    .rolling(5)
    .mean()
)

# =========================
# 5. DROP BARIS TANPA ROLLING (WAJIB)
# =========================
df = df.dropna(subset=[
    "team_avg_goals_for_5",
    "team_avg_goals_against_5",
    "opp_avg_goals_for_5",
    "opp_avg_goals_against_5"
]).reset_index(drop=True)

# =========================
# 6. PASTIKAN URUTAN KOLOM TETAP
# =========================
final_columns = [
    "match_date",
    "team",
    "opponent",
    "is_home",
    "team_goals",
    "opp_goals",
    "team_avg_goals_for_5",
    "team_avg_goals_against_5",
    "opp_avg_goals_for_5",
    "opp_avg_goals_against_5",
    "team_shots",
    "team_sot",
    "opp_shots",
    "opp_sot",
    "result",
    "over_3_5"
]

df = df[final_columns]

# =========================
# 7. OVERWRITE MASTER DATASET
# =========================
df.to_excel(master_path, index=False)

print("ROLLING FEATURE SELESAI")
print("Jumlah data final:", len(df))
print("Tanggal terakhir:", df["match_date"].max())
