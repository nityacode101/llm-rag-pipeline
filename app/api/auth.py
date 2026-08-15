"""Optional API-token gate so Gemini calls cannot be used anonymously."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.config import Settings, get_settings


def require_app_token(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = (settings.app_api_token or "").strip()
    if not expected:
        return
    provided = (x_api_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")
