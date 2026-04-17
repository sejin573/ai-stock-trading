from __future__ import annotations

import math
from typing import Any

import pandas as pd

from utils.helpers import compute_recent_return


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def calculate_price_forecast(
    stock_df: pd.DataFrame,
    signal: dict[str, Any],
    current_price: float,
) -> dict[str, float | str]:
    recent_return_5d = compute_recent_return(stock_df["close"], periods=5)
    recent_return_20d = compute_recent_return(stock_df["close"], periods=20)
    recent_volatility = float(signal.get("recent_volatility", 0.0) or 0.0)
    average_sentiment = float(signal.get("average_sentiment", 0.0) or 0.0)
    pattern_bias = float(signal.get("pattern_bias", 0.0) or 0.0)
    positive_events = int(signal.get("positive_events", 0) or 0)
    negative_events = int(signal.get("negative_events", 0) or 0)
    event_balance = (positive_events - negative_events) / max(positive_events + negative_events, 1)

    raw_probability = (
        average_sentiment * 2.2
        + event_balance * 1.1
        + pattern_bias * 2.4
        + recent_return_5d * 10
        + recent_return_20d * 4
        - recent_volatility * 3
    )
    up_probability = _sigmoid(raw_probability)

    expected_return_pct = (
        average_sentiment * 0.018
        + event_balance * 0.012
        + pattern_bias * 0.018
        + recent_return_5d * 0.35
        + recent_return_20d * 0.15
    )
    move_cap = max(0.01, min(0.08, recent_volatility * 1.75 if recent_volatility > 0 else 0.025))
    expected_return_pct = max(-move_cap, min(move_cap, expected_return_pct))

    if expected_return_pct >= 0.002:
        direction = "상승 예상"
    elif expected_return_pct <= -0.002:
        direction = "하락 예상"
    else:
        direction = "보합권"

    predicted_price = current_price * (1 + expected_return_pct)

    return {
        "direction": direction,
        "up_probability": up_probability,
        "down_probability": 1 - up_probability,
        "expected_return_pct": expected_return_pct,
        "predicted_price": predicted_price,
        "recent_return_5d": recent_return_5d,
        "recent_return_20d": recent_return_20d,
        "pattern_bias": pattern_bias,
    }
