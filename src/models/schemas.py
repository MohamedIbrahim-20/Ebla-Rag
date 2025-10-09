from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Float
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime
import uuid

from .db import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata: Mapped[dict | None] = mapped_column(SQLITE_JSON, nullable=True)

    messages: Mapped[list["Message"]] = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    summaries: Mapped[list["Summary"]] = relationship("Summary", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user|assistant|system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parent_message_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("messages.id"), nullable=True)

    session: Mapped[Session] = relationship("Session", back_populates="messages")
    retrievals: Mapped[list["Retrieval"]] = relationship("Retrieval", back_populates="message", cascade="all, delete-orphan")


class Retrieval(Base):
    __tablename__ = "retrievals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)  # faiss|llamaindex|other
    source_id: Mapped[str] = mapped_column(String(512), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunk: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[dict | None] = mapped_column(SQLITE_JSON, nullable=True)

    message: Mapped[Message] = relationship("Message", back_populates="retrievals")


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    up_to_message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id"), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    tokens_saved: Mapped[int | None] = mapped_column(Integer, nullable=True)

    session: Mapped[Session] = relationship("Session", back_populates="summaries")


