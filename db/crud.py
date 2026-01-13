from sqlalchemy.orm import Session
from datetime import datetime, date, timezone

from db import models


# ---------------- USERS ----------------

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_id(db: Session, user_id: str):
    return db.query(models.User).filter(models.User.id == user_id).first()


def create_user(db: Session, email: str, hashed_password: str):
    user = models.User(email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)

    usage = models.Usage(user_id=user.id)
    db.add(usage)
    db.commit()

    return user


# ---------------- CONVERSATIONS ----------------

def create_conversation(db: Session, user_id: str, title: str = "New Chat"):
    convo = models.Conversation(user_id=user_id, title=title)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo


def get_conversation(db: Session, conversation_id: str, user_id: str):
    return (
        db.query(models.Conversation)
        .filter(
            models.Conversation.id == conversation_id,
            models.Conversation.user_id == user_id
        )
        .first()
    )


def get_user_conversations(db: Session, user_id: str):
    return (
        db.query(models.Conversation)
        .filter(models.Conversation.user_id == user_id)
        .order_by(models.Conversation.created_at.desc())
        .all()
    )


# ---------------- MESSAGES ----------------

def create_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    confidence=None,
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
    model_name=None,
):
    msg = models.Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        confidence=str(confidence) if confidence is not None else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model_name=model_name,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_conversation_messages(db: Session, conversation_id: str):
    return (
        db.query(models.Message)
        .filter(models.Message.conversation_id == conversation_id)
        .order_by(models.Message.created_at.asc())
        .all()
    )


# ---------------- USAGE ----------------

def get_usage(db: Session, user_id: str):
    return db.query(models.Usage).filter(models.Usage.user_id == user_id).first()


def _reset_daily_usage_if_needed(usage: models.Usage):
    today_utc = datetime.now(timezone.utc).date()

    if not usage.last_request:
        return

    last_day = usage.last_request.date()

    if last_day != today_utc:
        usage.daily_queries = 0
        usage.daily_tokens = 0


def increment_usage(db: Session, user_id: str, tokens: int = 0):

    usage = get_usage(db, user_id)

    if not usage:
        usage = models.Usage(user_id=user_id)
        db.add(usage)
        db.commit()
        db.refresh(usage)

    _reset_daily_usage_if_needed(usage)

    usage.total_queries += 1
    usage.total_tokens += tokens

    usage.daily_queries += 1
    usage.daily_tokens += tokens

    usage.last_request = datetime.now(timezone.utc)

    db.commit()
    db.refresh(usage)

    return usage
