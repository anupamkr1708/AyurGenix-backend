from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from db.session import get_db
from auth.security import get_current_user
from usage.service import enforce_limits

def enforce_usage_limits(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    usage = enforce_limits(db, user.id)
    if not usage:
        raise HTTPException(500, "Usage record not found")
    return usage
