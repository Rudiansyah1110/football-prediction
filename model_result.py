# model_result.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from utils import FEATURES

def train_match_result_model(df):
    X = df[FEATURES]
    y = df["result"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    coef_df = pd.DataFrame(
        model.coef_.T,
        index=FEATURES,
        columns=[f"class_{c}" for c in le.classes_]
    )

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "coef_df": coef_df
    }
