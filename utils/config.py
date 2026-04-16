from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    naver_client_id: str
    naver_client_secret: str
    kis_app_key: str
    kis_app_secret: str
    kis_mock_app_key: str
    kis_mock_app_secret: str
    kis_account_no: str
    kis_account_product_code: str


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        naver_client_id=os.getenv("NAVER_CLIENT_ID", "").strip(),
        naver_client_secret=os.getenv("NAVER_CLIENT_SECRET", "").strip(),
        kis_app_key=os.getenv("KIS_APP_KEY", "").strip(),
        kis_app_secret=os.getenv("KIS_APP_SECRET", "").strip(),
        kis_mock_app_key=os.getenv("KIS_MOCK_APP_KEY", "").strip(),
        kis_mock_app_secret=os.getenv("KIS_MOCK_APP_SECRET", "").strip(),
        kis_account_no=os.getenv("KIS_ACCOUNT_NO", "").strip(),
        kis_account_product_code=os.getenv("KIS_ACCOUNT_PRODUCT_CODE", "").strip(),
    )
