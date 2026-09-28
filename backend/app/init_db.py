from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, Session, create_engine, select

from app.core.config import settings
from app.core.security import get_password_hash
from app.models.conversation_model import Conversation
from app.models.document_model import Document
from app.models.message_model import Message
from app.models.user_model import User


async def init_db() -> None:
    engine = create_async_engine(settings.ASYNC_DATABASE_URI, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(SQLModel.metadata.create_all)
    await engine.dispose()


def create_super_user() -> None:
    engine = create_engine(settings.SYNC_DATABASE_URI, pool_pre_ping=True)
    with Session(engine) as session:
        user = session.exec(
            select(User).where(User.email == settings.FIRST_SUPERUSER.lower())
        ).first()
        if user:
            return

        session.add(
            User(
                email=settings.FIRST_SUPERUSER.lower(),
                hashed_password=get_password_hash(settings.FIRST_SUPERUSER_PASSWORD),
                is_active=True,
                is_superuser=True,
            )
        )
        session.commit()
    engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_db())
    create_super_user()
