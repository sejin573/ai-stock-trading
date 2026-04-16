from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEARNING_STATE_PATH = DATA_DIR / "learning_state.json"

DEFAULT_WEIGHTS = {
    "model_score": 0.80,
    "up_probability": 0.70,
    "expected_return": 0.90,
    "sentiment": 0.50,
    "volatility": 0.40,
    "entry_timing": 0.40,
}


def _default_learning_state() -> dict[str, Any]:
    return {
        "weights": DEFAULT_WEIGHTS.copy(),
        "bias": 0.0,
        "sample_count": 0,
        "processed_trade_ids": [],
        "last_updated": "",
    }


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_learning_state() -> dict[str, Any]:
    _ensure_data_dir()
    if not LEARNING_STATE_PATH.exists():
        return _default_learning_state()

    try:
        payload = json.loads(LEARNING_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _default_learning_state()

    state = _default_learning_state()
    if isinstance(payload, dict):
        weights = payload.get("weights", {})
        if isinstance(weights, dict):
            for key, value in DEFAULT_WEIGHTS.items():
                try:
                    state["weights"][key] = float(weights.get(key, value))
                except (TypeError, ValueError):
                    state["weights"][key] = value
        try:
            state["bias"] = float(payload.get("bias", 0.0))
        except (TypeError, ValueError):
            state["bias"] = 0.0
        try:
            state["sample_count"] = int(payload.get("sample_count", 0))
        except (TypeError, ValueError):
            state["sample_count"] = 0
        processed_trade_ids = payload.get("processed_trade_ids", [])
        if isinstance(processed_trade_ids, list):
            state["processed_trade_ids"] = [str(value) for value in processed_trade_ids if str(value).strip()]
        state["last_updated"] = str(payload.get("last_updated", ""))

    return state


def save_learning_state(state: dict[str, Any]) -> None:
    _ensure_data_dir()
    LEARNING_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tanh(value: float) -> float:
    return math.tanh(value)


def extract_learning_features(row: dict[str, Any]) -> dict[str, float]:
    final_score = _safe_float(row.get("final_score", row.get("opportunity_score", 50.0)), 50.0)
    up_probability_pct = _safe_float(row.get("up_probability_pct", row.get("up_probability", 50.0)), 50.0)
    expected_return_pct = _safe_float(row.get("expected_return_pct", 0.0))
    average_sentiment = _safe_float(row.get("average_sentiment", 0.0))
    recent_volatility_pct = _safe_float(row.get("recent_volatility_pct", 0.0))
    realtime_change_rate = _safe_float(row.get("realtime_change_rate", 0.0))

    return {
        "model_score": _clip((final_score - 50.0) / 50.0, -1.0, 1.0),
        "up_probability": _clip((up_probability_pct - 50.0) / 50.0, -1.0, 1.0),
        "expected_return": _clip(expected_return_pct / 8.0, -1.0, 1.0),
        "sentiment": _clip(average_sentiment, -1.0, 1.0),
        "volatility": _clip((recent_volatility_pct - 5.0) / 10.0, -1.0, 1.0),
        "entry_timing": _clip((-realtime_change_rate) / 10.0, -1.0, 1.0),
    }


def predict_learning_signal(row: dict[str, Any], state: dict[str, Any] | None = None) -> float:
    state = state or load_learning_state()
    weights = state.get("weights", DEFAULT_WEIGHTS)
    bias = _safe_float(state.get("bias", 0.0))
    features = extract_learning_features(row)

    raw_score = bias
    for key, value in features.items():
        raw_score += _safe_float(weights.get(key, DEFAULT_WEIGHTS[key])) * value

    return _tanh(raw_score / max(len(features), 1))


def apply_learning_to_row(
    row: dict[str, Any],
    *,
    score_key: str,
    state: dict[str, Any] | None = None,
    adjustment_scale: float = 12.0,
) -> dict[str, Any]:
    copied = row.copy()
    learning_signal = predict_learning_signal(copied, state=state)
    learning_adjustment = learning_signal * adjustment_scale
    copied["learning_signal"] = round(learning_signal, 4)
    copied["learning_adjustment"] = round(learning_adjustment, 2)
    copied[score_key] = round(_safe_float(copied.get(score_key, 0.0)) + learning_adjustment, 2)
    return copied


def _build_trade_id(row: dict[str, Any]) -> str:
    return f"{row.get('id', '')}:{row.get('closed_at', '')}:{row.get('symbol', '')}"


def _build_buy_lookup(strategy_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    order_rows = strategy_payload.get("orders", [])
    if not isinstance(order_rows, list):
        return lookup

    for row in order_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("type", "")).lower() != "buy":
            continue
        key = (str(row.get("symbol", "")), str(row.get("ordered_at", "")))
        lookup[key] = row
    return lookup


def _find_matching_buy(
    closed_position: dict[str, Any],
    buy_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    symbol = str(closed_position.get("symbol", ""))
    created_at = str(closed_position.get("created_at", ""))
    direct_match = buy_lookup.get((symbol, created_at))
    if direct_match:
        return direct_match

    fallback_rows = [row for (candidate_symbol, _), row in buy_lookup.items() if candidate_symbol == symbol]
    if fallback_rows:
        fallback_rows = sorted(fallback_rows, key=lambda row: str(row.get("ordered_at", "")), reverse=True)
        return fallback_rows[0]

    return {}


def update_learning_from_history(
    strategy_payload: dict[str, Any],
    trade_state_payload: dict[str, Any],
    learning_rate: float = 0.08,
) -> dict[str, Any]:
    state = load_learning_state()
    processed_ids = set(state.get("processed_trade_ids", []))
    weights = state.get("weights", DEFAULT_WEIGHTS.copy())
    bias = _safe_float(state.get("bias", 0.0))
    sample_count = int(state.get("sample_count", 0))
    buy_lookup = _build_buy_lookup(strategy_payload)

    closed_rows = trade_state_payload.get("history", [])
    if not isinstance(closed_rows, list):
        return state

    updated = False
    for closed_position in closed_rows:
        if not isinstance(closed_position, dict):
            continue

        trade_id = _build_trade_id(closed_position)
        if trade_id in processed_ids:
            continue

        entry_price = _safe_float(closed_position.get("entry_price", 0.0))
        sell_price = _safe_float(closed_position.get("sell_price", 0.0))
        if entry_price <= 0 or sell_price <= 0:
            processed_ids.add(trade_id)
            continue

        realized_return_pct = ((sell_price / entry_price) - 1.0) * 100.0
        reward = _clip(realized_return_pct / 10.0, -1.0, 1.0)

        buy_row = _find_matching_buy(closed_position, buy_lookup)
        learning_source = {}
        learning_source.update(closed_position)
        learning_source.update(buy_row)
        features = extract_learning_features(learning_source)

        predicted = _tanh((bias + sum(_safe_float(weights.get(key, DEFAULT_WEIGHTS[key])) * value for key, value in features.items())) / max(len(features), 1))
        error = reward - predicted

        for key, feature_value in features.items():
            current_weight = _safe_float(weights.get(key, DEFAULT_WEIGHTS[key]), DEFAULT_WEIGHTS[key])
            current_weight += learning_rate * error * feature_value
            weights[key] = round(_clip(current_weight, -3.0, 3.0), 6)

        bias = round(_clip(bias + learning_rate * error * 0.3, -2.0, 2.0), 6)
        sample_count += 1
        processed_ids.add(trade_id)
        updated = True

    if updated:
        state["weights"] = weights
        state["bias"] = bias
        state["sample_count"] = sample_count
        state["processed_trade_ids"] = sorted(processed_ids)[-2000:]
        state["last_updated"] = datetime.now().isoformat(timespec="seconds")
        save_learning_state(state)

    return state
