# model_over_under.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils import FEATURES

def train_over_under_model(df, threshold):
    df = df.copy()

    target = f"over_{threshold}"
    df[target] = (
        df["team_goals"] + df["opp_goals"] > threshold
    ).astype(int)

    X = df[FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold
    }