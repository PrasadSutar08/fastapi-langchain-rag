from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.message_crud import MessageCRUD
from app.models.conversation_model import Conversation


class ConversationCRUD:

    @staticmethod
    async def get(
        session: AsyncSession,
        conversation_id: int,
    ) -> Conversation | None:

        result = await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_for_user(
        session: AsyncSession,
        conversation_id: int,
        user_id: int,
    ) -> Conversation | None:

        result = await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_user(
        session: AsyncSession,
        user_id: int,
    ) -> list[Conversation]:

        result = await session.execute(
            select(Conversation)
            .where(
                Conversation.user_id == user_id
            )
            .order_by(
                Conversation.updated_at.desc()
            )
        )

        return list(
            result.scalars().all()
        )

    @staticmethod
    async def create(
        session: AsyncSession,
        user_id: int,
        title: str = "New Conversation",
    ) -> Conversation:

        conversation = Conversation(
            user_id=user_id,
            title=(
                title or "New Conversation"
            ).strip()[:255],
        )

        session.add(conversation)

        await session.commit()
        await session.refresh(conversation)

        return conversation

    @staticmethod
    async def rename(
        session: AsyncSession,
        conversation: Conversation,
        title: str,
    ) -> Conversation:

        conversation.title = (title or "New Conversation").strip()[:255] or "New Conversation"

        session.add(conversation)

        await session.commit()
        await session.refresh(conversation)

        return conversation

    @staticmethod
    async def delete(
        session: AsyncSession,
        conversation: Conversation,
    ) -> None:

        try:

            if conversation.id is not None:

                await MessageCRUD.delete_for_conversation(
                    session=session,
                    conversation_id=conversation.id,
                )

            await session.delete(
                conversation
            )

            await session.commit()

        except Exception:

            await session.rollback()

            raise


conversation_crud = ConversationCRUD()