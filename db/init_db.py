from db.session import engine
from db.base import Base
from db import models

print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("Done.")
