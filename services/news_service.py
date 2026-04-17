from __future__ import annotations

from email.utils import parsedate_to_datetime
import html
import re
from typing import Any

import pandas as pd
import requests


NAVER_NEWS_API_URL = "https://openapi.naver.com/v1/search/news.json"

BASE_MARKET_ISSUE_QUERIES = [
    "한국 증시 정책 금리 환율",
    "정부 정치 증시 정책",
]

MARKET_THEME_QUERY_MAP = {
    "semiconductor": "반도체 정책 규제 수출",
    "battery": "2차전지 배터리 보조금 정책",
    "auto": "자동차 관세 전기차 보조금",
    "bio": "바이오 제약 의료 정책 규제",
    "finance": "금융 정책 은행 규제 금리",
    "energy": "에너지 원전 전력 정책",
    "defense": "방산 국방 예산 수출 정책",
    "platform": "플랫폼 AI 데이터 규제 정책",
    "steel": "철강 관세 건설 경기 정책",
    "retail": "소비 내수 유통 정책 물가",
}

COMPANY_THEME_HINTS = {
    "semiconductor": [
        "반도체", "하이닉스", "한미반도체", "db하이텍", "원익", "주성", "isc", "psk",
        "이오테크닉스", "하나마이크론", "lx세미콘", "테스", "칩스", "삼성전자",
    ],
    "battery": [
        "배터리", "lg화학", "삼성sdi", "에코프로", "포스코퓨처엠", "엘앤에프", "sk이노베이션",
        "금양", "더블유씨피", "나노신소재",
    ],
    "auto": [
        "현대차", "기아", "모비스", "자동차", "만도", "한온", "에스엘", "현대위아",
    ],
    "bio": [
        "바이오", "셀트리온", "유한", "삼성바이오", "sk바이오", "한미약품", "제약", "메디",
    ],
    "finance": [
        "금융", "은행", "증권", "카카오뱅크", "kb", "신한", "하나금융", "우리금융", "메리츠",
    ],
    "energy": [
        "에너지", "원전", "두산에너빌리티", "한국전력", "한전", "가스", "정유", "s-oil", "gs",
    ],
    "defense": [
        "방산", "한화에어로", "한국항공우주", "lig넥스원", "현대로템", "풍산",
    ],
    "platform": [
        "naver", "네이버", "카카오", "플랫폼", "엔씨", "크래프톤", "ai", "데이터",
    ],
    "steel": [
        "철강", "포스코", "동국", "현대제철", "세아", "건설",
    ],
    "retail": [
        "유통", "소비", "호텔", "면세", "화장품", "아모레", "f&f", "이마트", "롯데",
    ],
}


def _strip_html_tags(text: str) -> str:
    clean_text = re.sub(r"<[^>]+>", "", text or "")
    return html.unescape(clean_text).strip()


def fetch_company_news(
    query: str,
    client_id: str,
    client_secret: str,
    page_size: int = 20,
    sort_by: str = "date",
    start: int = 1,
) -> pd.DataFrame:
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {
        "query": query,
        "display": min(page_size, 100),
        "start": start,
        "sort": sort_by,
    }

    response = requests.get(NAVER_NEWS_API_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    rows: list[dict[str, Any]] = []
    for article in payload.get("items", []):
        title = _strip_html_tags(article.get("title", ""))
        description = _strip_html_tags(article.get("description", ""))
        raw_published_at = article.get("pubDate", "")
        published_at = parsedate_to_datetime(raw_published_at) if raw_published_at else None
        article_url = article.get("originallink") or article.get("link") or ""

        rows.append(
            {
                "source": "네이버 뉴스 검색",
                "author": None,
                "title": title,
                "description": description,
                "content": description,
                "url": article_url,
                "published_at": published_at,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "author",
                "title",
                "description",
                "content",
                "url",
                "published_at",
            ]
        )

    news_df = pd.DataFrame(rows)
    news_df["published_at"] = pd.to_datetime(news_df["published_at"], errors="coerce").dt.tz_localize(None)
    news_df["article_text"] = (
        news_df["title"].fillna("")
        + ". "
        + news_df["description"].fillna("")
        + ". "
        + news_df["content"].fillna("")
    ).str.strip()
    news_df = news_df.drop_duplicates(subset=["url", "title"]).sort_values("published_at")
    return news_df.reset_index(drop=True)


def infer_market_issue_theme(company_name: str) -> str | None:
    normalized_name = (company_name or "").strip().lower()
    if not normalized_name:
        return None

    for theme, hints in COMPANY_THEME_HINTS.items():
        if any(hint.lower() in normalized_name for hint in hints):
            return theme
    return None


def build_market_issue_queries(company_name: str) -> list[str]:
    queries = list(BASE_MARKET_ISSUE_QUERIES)
    inferred_theme = infer_market_issue_theme(company_name)
    if inferred_theme:
        theme_query = MARKET_THEME_QUERY_MAP.get(inferred_theme)
        if theme_query:
            queries.append(theme_query)

    deduped_queries: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized_query = query.strip()
        if not normalized_query or normalized_query in seen:
            continue
        deduped_queries.append(normalized_query)
        seen.add(normalized_query)
    return deduped_queries


def fetch_news_by_queries(
    queries: list[str],
    client_id: str,
    client_secret: str,
    page_size: int = 6,
    sort_by: str = "date",
) -> pd.DataFrame:
    if not queries:
        return pd.DataFrame(
            columns=[
                "source",
                "author",
                "title",
                "description",
                "content",
                "url",
                "published_at",
                "query",
            ]
        )

    frames: list[pd.DataFrame] = []
    for query in queries:
        query_df = fetch_company_news(
            query=query,
            client_id=client_id,
            client_secret=client_secret,
            page_size=page_size,
            sort_by=sort_by,
        )
        if query_df.empty:
            continue
        enriched_df = query_df.copy()
        enriched_df["query"] = query
        frames.append(enriched_df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "source",
                "author",
                "title",
                "description",
                "content",
                "url",
                "published_at",
                "query",
            ]
        )

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["url", "title"]).sort_values("published_at")
    return merged_df.reset_index(drop=True)
