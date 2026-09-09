from __future__ import annotations

from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message_model import Message


class MessageCRUD:
    """
    Async CRUD operations for conversation messages.

    Message writes intentionally use flush() rather than commit()
    so ChatService can persist the user and assistant messages
    atomically in one transaction.
    """

    @staticmethod
    async def get_history(
        session: AsyncSession,
        conversation_id: int,
        limit: Optional[int] = None,
    ) -> list[Message]:
        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id
            )
            .order_by(
                Message.created_at.asc()
            )
        )

        if limit is not None:
            statement = statement.limit(limit)

        result = await session.execute(statement)

        return list(
            result.scalars().all()
        )

    @staticmethod
    async def get(
        session: AsyncSession,
        message_id: int,
    ) -> Optional[Message]:

        result = await session.execute(
            select(Message).where(
                Message.id == message_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def add(
        session: AsyncSession,
        conversation_id: int,
        role: str,
        content: str,
    ) -> Message:
        """
        Add a message without committing.

        The caller owns the transaction.
        """

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

        session.add(message)

        await session.flush()
        await session.refresh(message)

        return message

    @staticmethod
    async def create(
        session: AsyncSession,
        conversation_id: int,
        role: str,
        content: str,
    ) -> Message:
        """
        Backwards-compatible alias for add().
        """

        return await MessageCRUD.add(
            session=session,
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

    @staticmethod
    async def delete_from(
        session: AsyncSession,
        conversation_id: int,
        from_message_id: int,
    ) -> int:
        """
        Delete a message and everything sent after it in the same
        conversation (by id, which is monotonically increasing with
        created_at). Used when a user edits an earlier message: the
        edited message and its old reply (and anything after) are
        removed so the edited text can be resent as a clean new turn.

        Commits immediately -- this is called as its own request, not
        as part of a larger atomic transaction.
        """

        result = await session.execute(
            delete(Message).where(
                Message.conversation_id == conversation_id,
                Message.id >= from_message_id,
            )
        )

        await session.commit()

        return int(
            result.rowcount or 0
        )

    @staticmethod
    async def delete_for_conversation(
        session: AsyncSession,
        conversation_id: int,
    ) -> int:
        """
        Delete all messages for a conversation.

        Does not commit. The caller owns the transaction.
        """

        result = await session.execute(
            delete(Message).where(
                Message.conversation_id == conversation_id
            )
        )

        return int(
            result.rowcount or 0
        )


message_crud = MessageCRUD()