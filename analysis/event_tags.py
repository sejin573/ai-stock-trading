from __future__ import annotations

import re

import pandas as pd


EVENT_RULES = {
    "earnings": {"earnings", "revenue", "guidance", "forecast"},
    "product_launch": {"launch", "release", "device", "product"},
    "lawsuit": {"lawsuit", "legal", "court", "sue", "settlement"},
    "merger_acquisition": {"acquire", "acquisition", "merge", "takeover"},
    "partnership": {"partnership", "collaboration", "joint", "alliance"},
    "regulation": {"regulation", "probe", "antitrust", "compliance"},
    "policy_support": {"subsidy", "stimulus", "support", "budget", "approval", "easing"},
    "politics_risk": {"political", "election", "tariff", "sanction", "strike", "conflict"},
}

KOREAN_EVENT_RULES = {
    "earnings": {"실적", "매출", "영업이익", "가이던스", "전망"},
    "product_launch": {"출시", "신제품", "공개", "라인업"},
    "lawsuit": {"소송", "법적", "법원", "고소", "합의"},
    "merger_acquisition": {"인수", "합병", "M&A", "매각"},
    "partnership": {"제휴", "협력", "동맹", "파트너십", "공동개발"},
    "regulation": {"규제", "조사", "반독점", "당국", "제재"},
    "policy_support": {"지원", "보조금", "예산", "추경", "완화", "승인", "육성", "진흥"},
    "politics_risk": {"정치", "선거", "탄핵", "관세", "제재", "갈등", "파업", "불확실성"},
}

EVENT_LABELS = {
    "earnings": "실적",
    "product_launch": "제품 출시",
    "lawsuit": "소송",
    "merger_acquisition": "인수합병",
    "partnership": "제휴",
    "regulation": "규제",
    "policy_support": "정책 수혜",
    "politics_risk": "정치/정책 리스크",
    "general": "일반",
}


def extract_event_tags(text: str) -> list[str]:
    lower_text = text.lower()
    tokens = set(re.findall(r"[a-zA-Z']+", lower_text))
    matched_tags = [tag for tag, keywords in EVENT_RULES.items() if tokens.intersection(keywords)]

    for tag, keywords in KOREAN_EVENT_RULES.items():
        if any(keyword.lower() in lower_text for keyword in keywords) and tag not in matched_tags:
            matched_tags.append(tag)

    return matched_tags or ["general"]


def add_event_tags(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        enriched_df = news_df.copy()
        enriched_df["event_tag_list"] = pd.Series(dtype=object)
        enriched_df["event_tags"] = pd.Series(dtype=str)
        return enriched_df

    enriched_df = news_df.copy()
    enriched_df["event_tag_list"] = enriched_df["article_text"].map(extract_event_tags)
    enriched_df["event_tags"] = enriched_df["event_tag_list"].map(
        lambda tags: ", ".join(EVENT_LABELS.get(tag, tag) for tag in tags)
    )
    return enriched_df
