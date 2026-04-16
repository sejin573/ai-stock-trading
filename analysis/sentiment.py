from __future__ import annotations

import re

import pandas as pd


POSITIVE_KEYWORDS = {
    "beat",
    "strong",
    "growth",
    "surge",
    "gain",
    "record",
    "optimistic",
    "partnership",
    "upgrade",
    "profit",
    "expands",
    "launch",
}

NEGATIVE_KEYWORDS = {
    "miss",
    "weak",
    "decline",
    "drop",
    "loss",
    "lawsuit",
    "probe",
    "downgrade",
    "delay",
    "cuts",
    "cut",
    "recall",
    "risk",
}

POSITIVE_KOREAN_KEYWORDS = {
    "호실적",
    "실적개선",
    "실적 개선",
    "성장",
    "증가",
    "상승",
    "급등",
    "반등",
    "수주",
    "계약",
    "확대",
    "신제품",
    "흑자",
    "호재",
    "강세",
    "개선",
}

NEGATIVE_KOREAN_KEYWORDS = {
    "부진",
    "실적악화",
    "실적 악화",
    "감소",
    "하락",
    "급락",
    "적자",
    "소송",
    "규제",
    "지연",
    "리콜",
    "악재",
    "우려",
    "약세",
    "충격",
    "불확실성",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def score_sentiment(text: str) -> float:
    tokens = _tokenize(text)
    normalized_text = text.lower()
    if not tokens:
        positive_korean_hits = sum(keyword in normalized_text for keyword in POSITIVE_KOREAN_KEYWORDS)
        negative_korean_hits = sum(keyword in normalized_text for keyword in NEGATIVE_KOREAN_KEYWORDS)
        raw_score = positive_korean_hits - negative_korean_hits
        return max(-1.0, min(1.0, raw_score / 3)) if raw_score else 0.0

    positive_hits = sum(token in POSITIVE_KEYWORDS for token in tokens)
    negative_hits = sum(token in NEGATIVE_KEYWORDS for token in tokens)
    positive_korean_hits = sum(keyword in normalized_text for keyword in POSITIVE_KOREAN_KEYWORDS)
    negative_korean_hits = sum(keyword in normalized_text for keyword in NEGATIVE_KOREAN_KEYWORDS)
    raw_score = (positive_hits + positive_korean_hits) - (negative_hits + negative_korean_hits)
    normalized_score = raw_score / max(len(tokens) ** 0.5, 1)
    return max(-1.0, min(1.0, normalized_score))


def sentiment_label(score: float) -> str:
    if score >= 0.2:
        return "긍정"
    if score <= -0.2:
        return "부정"
    return "중립"


def add_sentiment_columns(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        enriched_df = news_df.copy()
        enriched_df["sentiment_score"] = pd.Series(dtype=float)
        enriched_df["sentiment_label"] = pd.Series(dtype=str)
        return enriched_df

    enriched_df = news_df.copy()
    enriched_df["sentiment_score"] = enriched_df["article_text"].map(score_sentiment)
    enriched_df["sentiment_label"] = enriched_df["sentiment_score"].map(sentiment_label)
    return enriched_df
