from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from utils.config import Settings


SYNC_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolio_sync_state.json"


def _ensure_state_dir() -> None:
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_sync_state() -> dict[str, Any]:
    _ensure_state_dir()
    if not SYNC_STATE_PATH.exists():
        return {}

    try:
        payload = json.loads(SYNC_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_sync_state(state: dict[str, Any]) -> None:
    _ensure_state_dir()
    SYNC_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def should_sync_portfolio_snapshot(settings: Settings) -> bool:
    return bool(
        settings.portfolio_sync_enabled
        and settings.portfolio_sync_github_token
        and settings.portfolio_sync_github_repo
        and settings.portfolio_sync_github_path
    )


def sync_portfolio_snapshot_to_github(
    settings: Settings,
    *,
    snapshot_content: str,
    source_label: str = "local auto trading",
) -> tuple[bool, str]:
    if not should_sync_portfolio_snapshot(settings):
        return False, "portfolio sync is disabled"

    state = _load_sync_state()
    now_ts = int(time.time())
    min_interval_seconds = max(10, int(settings.portfolio_sync_min_interval_seconds))
    content_hash = _build_content_hash(snapshot_content)

    if state.get("last_content_hash") == content_hash:
        return False, "snapshot content has not changed"

    last_synced_ts = int(state.get("last_synced_ts", 0) or 0)
    if last_synced_ts and (now_ts - last_synced_ts) < min_interval_seconds:
        return False, "snapshot sync is waiting for the minimum interval"

    owner_repo = settings.portfolio_sync_github_repo.strip().strip("/")
    branch = settings.portfolio_sync_github_branch.strip() or "main"
    target_path = settings.portfolio_sync_github_path.strip().lstrip("/")

    api_url = f"https://api.github.com/repos/{owner_repo}/contents/{target_path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {settings.portfolio_sync_github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    remote_sha = str(state.get("remote_sha", "")).strip()
    if not remote_sha:
        response = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=20)
        if response.status_code == 200:
            payload = response.json()
            remote_sha = str(payload.get("sha", "")).strip()
        elif response.status_code != 404:
            response.raise_for_status()

    body: dict[str, Any] = {
        "message": f"Update portfolio snapshot from {source_label}",
        "content": base64.b64encode(snapshot_content.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if remote_sha:
        body["sha"] = remote_sha

    response = requests.put(api_url, headers=headers, json=body, timeout=30)
    response.raise_for_status()
    payload = response.json()
    content_info = payload.get("content", {}) if isinstance(payload, dict) else {}

    state.update(
        {
            "last_content_hash": content_hash,
            "last_synced_ts": now_ts,
            "last_synced_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now_ts)),
            "remote_sha": str(content_info.get("sha", "")).strip() or remote_sha,
            "repo": owner_repo,
            "branch": branch,
            "path": target_path,
        }
    )
    _save_sync_state(state)
    return True, f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{target_path}"
