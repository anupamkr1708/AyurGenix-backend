from db.session import SessionLocal
from db import crud

db = SessionLocal()

print("Users:", crud.get_user_by_email(db, "test@test.com"))
