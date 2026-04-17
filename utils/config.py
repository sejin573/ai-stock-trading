from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip() or str(default))
    except ValueError:
        return default


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
    portfolio_sync_enabled: bool
    portfolio_sync_github_token: str
    portfolio_sync_github_repo: str
    portfolio_sync_github_branch: str
    portfolio_sync_github_path: str
    portfolio_sync_min_interval_seconds: int


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
        portfolio_sync_enabled=_env_bool("PORTFOLIO_SYNC_ENABLED", False),
        portfolio_sync_github_token=os.getenv("PORTFOLIO_SYNC_GITHUB_TOKEN", "").strip(),
        portfolio_sync_github_repo=os.getenv("PORTFOLIO_SYNC_GITHUB_REPO", "").strip(),
        portfolio_sync_github_branch=os.getenv("PORTFOLIO_SYNC_GITHUB_BRANCH", "main").strip() or "main",
        portfolio_sync_github_path=os.getenv("PORTFOLIO_SYNC_GITHUB_PATH", "public_data/portfolio_snapshot.json").strip() or "public_data/portfolio_snapshot.json",
        portfolio_sync_min_interval_seconds=max(10, _env_int("PORTFOLIO_SYNC_MIN_INTERVAL_SECONDS", 15)),
    )
