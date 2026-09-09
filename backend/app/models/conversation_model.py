from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


def utc_now_naive() -> datetime:
    """
    Return the current UTC time without timezone information.

    The PostgreSQL schema uses TIMESTAMP WITHOUT TIME ZONE,
    so application timestamps must also be timezone-naive.
    """
    return datetime.utcnow()


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="users.id",
        index=True,
        nullable=False,
    )

    title: str = Field(
        default="New Conversation",
        max_length=255,
    )

    created_at: datetime = Field(
        default_factory=utc_now_naive,
        nullable=False,
    )

    updated_at: datetime = Field(
        default_factory=utc_now_naive,
        nullable=False,
    )