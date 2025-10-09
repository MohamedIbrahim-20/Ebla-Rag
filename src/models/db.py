from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    # default to local sqlite file in src directory
    db_path = os.path.join(os.path.dirname(__file__), "chat_history.sqlite")
    return f"sqlite:///{db_path}"


engine = create_engine(get_database_url(), echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db():
    # Import models to register metadata
    from .schemas import Session, Message, Retrieval, Summary  # noqa: F401
    Base.metadata.create_all(bind=engine)


