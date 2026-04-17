from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_float(value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _score_direction(bias_value: float) -> str:
    if bias_value >= 0.45:
        return "상승 우세"
    if bias_value >= 0.15:
        return "약상승"
    if bias_value <= -0.45:
        return "하락 우세"
    if bias_value <= -0.15:
        return "약하락"
    return "중립"


def calculate_impact_signal(
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> dict[str, Any]:
    average_sentiment = _safe_float(news_df["sentiment_score"].mean()) if not news_df.empty else 0.0
    recent_article_count = int(len(news_df))

    article_count_series = feature_df["article_count"] if "article_count" in feature_df else pd.Series(dtype=float)
    baseline_volume = _safe_float(article_count_series.mean())
    volume_ratio = recent_article_count / max(baseline_volume, 1.0)
    volume_score = min(volume_ratio / 2.0, 1.0)

    positive_events = int(feature_df["positive_event_count"].sum()) if "positive_event_count" in feature_df else 0
    negative_events = int(feature_df["negative_event_count"].sum()) if "negative_event_count" in feature_df else 0
    event_balance = (positive_events - negative_events) / max(positive_events + negative_events, 1)
    event_score = max(0.0, min(1.0, (event_balance + 1) / 2))

    volatility = _safe_float(stock_df["daily_return"].tail(5).std())
    stability_score = max(0.0, 1.0 - min(volatility / 0.06, 1.0))

    sentiment_score = max(0.0, min(1.0, (average_sentiment + 1) / 2))
    impact_score = round(
        sentiment_score * 40
        + volume_score * 20
        + event_score * 20
        + stability_score * 20
    )

    bias_value = average_sentiment + (positive_events * 0.08) - (negative_events * 0.12)
    direction = _score_direction(bias_value)

    return {
        "impact_score": int(max(0, min(100, impact_score))),
        "direction": direction,
        "average_sentiment": average_sentiment,
        "article_count": recent_article_count,
        "positive_events": positive_events,
        "negative_events": negative_events,
        "recent_volatility": volatility,
        "volume_ratio": volume_ratio,
    }


def calculate_issue_bias(news_df: pd.DataFrame) -> dict[str, Any]:
    if news_df.empty:
        return {
            "policy_bias_score": 0.0,
            "policy_sentiment": 0.0,
            "policy_article_count": 0,
            "policy_positive_count": 0,
            "policy_negative_count": 0,
        }

    average_sentiment = _safe_float(news_df["sentiment_score"].mean()) if "sentiment_score" in news_df else 0.0
    positive_count = int((news_df["sentiment_score"] >= 0.2).sum()) if "sentiment_score" in news_df else 0
    negative_count = int((news_df["sentiment_score"] <= -0.2).sum()) if "sentiment_score" in news_df else 0

    positive_event_count = 0
    negative_event_count = 0
    if "event_tag_list" in news_df.columns:
        positive_event_count = int(
            news_df["event_tag_list"].map(
                lambda tags: int(any(tag in {"policy_support", "partnership", "earnings"} for tag in (tags or [])))
            ).sum()
        )
        negative_event_count = int(
            news_df["event_tag_list"].map(
                lambda tags: int(any(tag in {"politics_risk", "regulation", "lawsuit"} for tag in (tags or [])))
            ).sum()
        )

    raw_bias_score = (
        average_sentiment * 11.0
        + (positive_count - negative_count) * 0.9
        + (positive_event_count - negative_event_count) * 1.4
    )
    policy_bias_score = max(-12.0, min(12.0, raw_bias_score))

    return {
        "policy_bias_score": round(policy_bias_score, 2),
        "policy_sentiment": round(average_sentiment, 3),
        "policy_article_count": int(len(news_df)),
        "policy_positive_count": int(positive_count + positive_event_count),
        "policy_negative_count": int(negative_count + negative_event_count),
    }


def format_signal_summary(signal: dict[str, Any]) -> str:
    score = int(signal["impact_score"])
    direction = str(signal["direction"])
    avg_sentiment = float(signal["average_sentiment"])
    article_count = int(signal["article_count"])
    positive_events = int(signal["positive_events"])
    negative_events = int(signal["negative_events"])
    recent_volatility = float(signal["recent_volatility"])

    return (
        f"영향 점수는 {score}/100이며 방향성은 {direction}입니다. "
        f"최근 기사 {article_count}건의 평균 감성 점수는 {avg_sentiment:.2f}입니다. "
        f"이벤트 태그는 긍정 {positive_events}건, 부정 {negative_events}건이 감지되었습니다. "
        f"최근 5일 수익률 변동성은 {recent_volatility:.2%}입니다."
    )
