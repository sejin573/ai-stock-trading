from __future__ import annotations

import pandas as pd


def build_daily_feature_frame(stock_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = stock_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"]).dt.normalize()

    if news_df.empty:
        feature_df["article_count"] = 0
        feature_df["avg_sentiment"] = 0.0
        feature_df["negative_event_count"] = 0
        feature_df["positive_event_count"] = 0
        return feature_df

    news_copy = news_df.copy()
    news_copy["date"] = pd.to_datetime(news_copy["published_at"]).dt.normalize()
    news_copy["has_negative_event"] = news_copy["event_tag_list"].map(
        lambda tags: int(any(tag in {"lawsuit", "regulation", "politics_risk"} for tag in tags))
    )
    news_copy["has_positive_event"] = news_copy["event_tag_list"].map(
        lambda tags: int(any(tag in {"earnings", "product_launch", "partnership", "policy_support"} for tag in tags))
    )

    daily_news = (
        news_copy.groupby("date", as_index=False)
        .agg(
            article_count=("title", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            negative_event_count=("has_negative_event", "sum"),
            positive_event_count=("has_positive_event", "sum"),
        )
        .sort_values("date")
    )

    merged_df = feature_df.merge(daily_news, on="date", how="left")
    fill_map = {
        "article_count": 0,
        "avg_sentiment": 0.0,
        "negative_event_count": 0,
        "positive_event_count": 0,
    }
    return merged_df.fillna(fill_map)
