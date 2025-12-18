import pandas as pd

# =========================
# 1. LOAD MASTER DATASET
# =========================
master_path = "dataset/dataset_master_football.xlsx"
df_master = pd.read_excel(master_path)

# =========================
# 2. LOAD DATA BARU
# =========================
df_premier = pd.read_excel("dataset/2025_premier_liga.xlsx")
df_laliga = pd.read_excel("dataset/2025_la_liga.xlsx")

df_new = pd.concat([df_premier, df_laliga], ignore_index=True)

# =========================
# 3. DROP BARIS KOSONG
# =========================
df_new = df_new.dropna(subset=[
    "match_date", "team", "opponent",
    "team_goals", "opp_goals"
]).reset_index(drop=True)

# =========================
# 4. TAMBAHKAN KOLOM ROLLING (KOSONG DULU)
# =========================
rolling_cols = [
    "team_avg_goals_for_5",
    "team_avg_goals_against_5",
    "opp_avg_goals_for_5",
    "opp_avg_goals_against_5"
]

for col in rolling_cols:
    df_new[col] = pd.NA

# =========================
# 5. SAMAKAN URUTAN KOLOM
# =========================
standard_columns = [
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

df_new = df_new[standard_columns]
df_master = df_master[standard_columns]

# =========================
# 6. KONVERSI TIPE DATA
# =========================
df_new["match_date"] = pd.to_datetime(df_new["match_date"])
df_master["match_date"] = pd.to_datetime(df_master["match_date"])

numeric_cols = [
    "is_home", "team_goals", "opp_goals",
    "team_shots", "team_sot",
    "opp_shots", "opp_sot",
    "over_3_5"
]

df_new[numeric_cols] = df_new[numeric_cols].astype(float)
df_master[numeric_cols] = df_master[numeric_cols].astype(float)

# =========================
# 7. APPEND KE MASTER
# =========================
df_master_updated = pd.concat(
    [df_master, df_new],
    ignore_index=True
)

# =========================
# 8. SORT BERDASARKAN TANGGAL
# =========================
df_master_updated = df_master_updated.sort_values(
    "match_date"
).reset_index(drop=True)

# =========================
# 9. OVERWRITE FILE MASTER (INI KUNCI)
# =========================
df_master_updated.to_excel(
    master_path,
    index=False
)

print("UPDATE SELESAI")
print("Total baris:", len(df_master_updated))
print("Tanggal terakhir:", df_master_updated["match_date"].max())
