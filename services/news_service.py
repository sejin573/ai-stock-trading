from __future__ import annotations

from email.utils import parsedate_to_datetime
import html
import re
from typing import Any

import pandas as pd
import requests


NAVER_NEWS_API_URL = "https://openapi.naver.com/v1/search/news.json"


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
