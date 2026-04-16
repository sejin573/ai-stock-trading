from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLUMNS = [
    "avg_sentiment",
    "article_count",
    "positive_event_count",
    "negative_event_count",
    "daily_return",
    "volatility_5d",
    "volume_change",
]


def build_training_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    training_df = feature_df.copy().sort_values("date").reset_index(drop=True)
    training_df["next_close"] = training_df["close"].shift(-1)
    training_df["target_next_up"] = (training_df["next_close"] > training_df["close"]).astype(int)
    training_df = training_df.dropna(subset=FEATURE_COLUMNS + ["next_close"])
    return training_df.drop(columns=["next_close"])


def train_direction_model(feature_df: pd.DataFrame) -> RandomForestClassifier:
    training_df = build_training_frame(feature_df)
    if training_df.empty:
        raise ValueError("Not enough data to train a next-day direction model.")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )
    model.fit(training_df[FEATURE_COLUMNS], training_df["target_next_up"])
    return model
