# usage/routes.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.session import get_db
from auth.security import get_current_user
from db import crud

router = APIRouter(prefix="/usage", tags=["Usage"])


@router.get("/me")
def my_usage(db: Session = Depends(get_db), user=Depends(get_current_user)):
    usage = crud.get_usage(db, user.id)

    return {
        "email": user.email,
        "total_queries": usage.total_queries,
        "total_tokens": usage.total_tokens,
        "daily_queries": usage.daily_queries,
        "daily_tokens": usage.daily_tokens,
        "last_request": usage.last_request,
    }
