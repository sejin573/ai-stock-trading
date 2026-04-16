from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from models.train import FEATURE_COLUMNS


def predict_next_day_direction(
    model: RandomForestClassifier,
    feature_df: pd.DataFrame,
) -> dict[str, float | str]:
    if feature_df.empty:
        raise ValueError("Feature frame is empty.")

    latest_row = feature_df.sort_values("date").iloc[-1]
    prediction_input = latest_row[FEATURE_COLUMNS].to_frame().T.fillna(0)
    probability = float(model.predict_proba(prediction_input)[0][1])
    label = "Up" if probability >= 0.5 else "Down"

    return {
        "label": label,
        "up_probability": probability,
    }
