# usage/service.py
from fastapi import HTTPException
from datetime import date
from db import crud

DAILY_QUERY_LIMIT = 30
DAILY_TOKEN_LIMIT = 30000


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def reset_daily_if_needed(usage):
    today = date.today()
    if usage.last_request is None or usage.last_request.date() != today:
        usage.daily_queries = 0
        usage.daily_tokens = 0


def enforce_limits(db, user_id: str):
    usage = crud.get_usage(db, user_id)
    if not usage:
        return None

    reset_daily_if_needed(usage)

    if usage.daily_queries >= DAILY_QUERY_LIMIT:
        raise HTTPException(429, "Daily query limit reached. Try again tomorrow.")

    if usage.daily_tokens >= DAILY_TOKEN_LIMIT:
        raise HTTPException(429, "Daily token limit reached. Try again tomorrow.")

    return usage
