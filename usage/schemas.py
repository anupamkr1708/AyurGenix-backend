from pydantic import BaseModel
from datetime import datetime


class UsageOut(BaseModel):
    total_queries: int
    total_tokens: int
    daily_queries: int
    daily_tokens: int
    last_request: datetime | None

    class Config:
        from_attributes = True
