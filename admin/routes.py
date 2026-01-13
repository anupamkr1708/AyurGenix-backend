from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db.session import get_db
from db import models

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/stats")
def platform_stats(db: Session = Depends(get_db)):
    users = db.query(models.User).count()
    conversations = db.query(models.Conversation).count()
    messages = db.query(models.Message).count()
    usages = db.query(models.Usage).all()

    return {
        "users": users,
        "conversations": conversations,
        "messages": messages,
        "total_queries": sum(u.total_queries for u in usages),
        "total_tokens": sum(u.total_tokens for u in usages),
        "avg_queries_per_user": round(sum(u.total_queries for u in usages) / max(users, 1), 2)
    }
