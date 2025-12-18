# model_over_under.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils import FEATURES

def train_over_under_model(df):
    df = df.copy()
    df["over_under_3_5"] = (
        df["team_goals"] + df["opp_goals"] > 3.5
    ).astype(int)

    X = df[FEATURES]
    y = df["over_under_3_5"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return {
        "model": model,
        "scaler": scaler
    }
