from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


def utc_now_naive() -> datetime:
    """
    Return current UTC time without timezone information.

    PostgreSQL stores this column as TIMESTAMP WITHOUT TIME ZONE,
    therefore the Python value must also be timezone-naive.
    """
    return datetime.utcnow()


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    conversation_id: int = Field(
        foreign_key="conversations.id",
        index=True,
        nullable=False,
    )

    role: str = Field(
        nullable=False,
        max_length=50,
    )

    content: str = Field(
        nullable=False,
    )

    created_at: datetime = Field(
        default_factory=utc_now_naive,
        nullable=False,
    )