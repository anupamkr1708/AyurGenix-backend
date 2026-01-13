import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "AyurGenix AI"
    FREE_TIER_DAILY_LIMIT = 20
    MAX_TOKENS_PER_USER = 100000


settings = Settings()
