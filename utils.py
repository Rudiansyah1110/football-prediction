# utils.py
import pandas as pd

FEATURES = [
    "is_home",
    "team_avg_goals_for_5",
    "team_avg_goals_against_5",
    "opp_avg_goals_for_5",
    "opp_avg_goals_against_5",
    "team_shots",
    "team_sot",
    "opp_shots",
    "opp_sot"
]

def build_future_match(df, team, opponent, is_home=1):
    """
    Build ONE ROW future match feature vector
    based on last 5 historical matches
    """

    team_df = df[df["team"] == team].sort_values("match_date").tail(5)
    opp_df  = df[df["team"] == opponent].sort_values("match_date").tail(5)

    if len(team_df) < 5 or len(opp_df) < 5:
        raise ValueError("Not enough historical data for team or opponent")

    return pd.DataFrame([{
        "is_home": is_home,  # asumsi, bisa dibuat parameter
        "team_avg_goals_for_5": team_df["team_goals"].mean(),
        "team_avg_goals_against_5": team_df["opp_goals"].mean(),
        "opp_avg_goals_for_5": opp_df["team_goals"].mean(),
        "opp_avg_goals_against_5": opp_df["opp_goals"].mean(),
        "team_shots": team_df["team_shots"].mean(),
        "team_sot": team_df["team_sot"].mean(),
        "opp_shots": opp_df["team_shots"].mean(),
        "opp_sot": opp_df["team_sot"].mean()
    }])

# def build_future_match_ou(df, team, opponent):
#     row = df[
#         (df["team"] == team) &
#         (df["opponent"] == opponent)
#     ].sort_values("match_date").iloc[-1]

#     return row[FEATURES].to_frame().T

def load_dataset(path):
    df = pd.read_excel(path)

    # basic sanity check
    required_cols = FEATURES + [
        "team_goals", "opp_goals", "result"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df

