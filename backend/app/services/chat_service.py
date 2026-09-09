from __future__ import annotations

import time
from datetime import datetime

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import (
    logger,
    settings,
)

from app.crud.conversation_crud import (
    ConversationCRUD,
)

from app.crud.message_crud import (
    MessageCRUD,
)

from app.services.rag_service import (
    rag_service,
)


def utc_now_naive() -> datetime:
    """
    Return current UTC time as a timezone-naive datetime.

    The PostgreSQL columns are TIMESTAMP WITHOUT TIME ZONE.
    """
    return datetime.utcnow()


class ChatService:

    async def chat(
        self,
        session: AsyncSession,
        user_id: int,
        conversation_id: int,
        question: str,
    ) -> dict:

        started = time.perf_counter()

        question = (
            question or ""
        ).strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        logger.info(
            "CHAT REQUEST STARTED "
            f"| user_id={user_id} "
            f"| conversation_id={conversation_id} "
            f"| question={question!r}"
        )

        # --------------------------------------------------
        # Validate conversation ownership
        # --------------------------------------------------

        conversation = (
            await ConversationCRUD.get_for_user(
                session=session,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )

        if conversation is None:

            logger.warning(
                "CHAT CONVERSATION NOT FOUND "
                f"| user_id={user_id} "
                f"| conversation_id={conversation_id}"
            )

            raise ValueError(
                "Conversation not found."
            )

        # --------------------------------------------------
        # Load conversation history
        # --------------------------------------------------

        stored_messages = (
            await MessageCRUD.get_history(
                session=session,
                conversation_id=conversation_id,
                limit=settings.MAX_HISTORY_MESSAGES,
            )
        )

        history = []

        for message in stored_messages:

            if message.role == "user":

                history.append(
                    HumanMessage(
                        content=message.content
                    )
                )

            elif message.role == "assistant":

                history.append(
                    AIMessage(
                        content=message.content
                    )
                )

        logger.info(
            "CHAT HISTORY LOADED "
            f"| user_id={user_id} "
            f"| conversation_id={conversation_id} "
            f"| history_messages={len(stored_messages)}"
        )

        try:

            # --------------------------------------------------
            # RAG
            # --------------------------------------------------

            logger.info(
                "CHAT RAG STARTED "
                f"| user_id={user_id} "
                f"| conversation_id={conversation_id}"
            )

            result = await rag_service.ask(
                question=question,
                chat_history=history,
                user_id=user_id,
            )

            if not result:
                raise RuntimeError(
                    "RAG service returned an empty response."
                )

            answer = (
                result.get("answer")
                or ""
            ).strip()

            if not answer:
                raise RuntimeError(
                    "RAG service returned an empty answer."
                )

            sources = result.get(
                "sources",
                [],
            )

            logger.info(
                "CHAT RAG COMPLETED "
                f"| user_id={user_id} "
                f"| conversation_id={conversation_id} "
                f"| answer_chars={len(answer)} "
                f"| sources={len(sources)}"
            )

            # --------------------------------------------------
            # Persist user message
            # --------------------------------------------------

            user_message = await MessageCRUD.add(
                session=session,
                conversation_id=conversation_id,
                role="user",
                content=question,
            )

            # --------------------------------------------------
            # Persist assistant message
            # --------------------------------------------------

            assistant_message = await MessageCRUD.add(
                session=session,
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
            )

            # --------------------------------------------------
            # Update conversation timestamp
            #
            # IMPORTANT:
            # PostgreSQL uses TIMESTAMP WITHOUT TIME ZONE.
            # Therefore this MUST be timezone-naive UTC.
            # --------------------------------------------------

            conversation.updated_at = (
                utc_now_naive()
            )

            session.add(
                conversation
            )

            # --------------------------------------------------
            # Atomic commit
            # --------------------------------------------------

            await session.commit()

            logger.info(
                "CHAT PERSISTENCE COMPLETED "
                f"| user_id={user_id} "
                f"| conversation_id={conversation_id}"
            )

        except Exception as exc:

            await session.rollback()

            logger.exception(
                "CHAT REQUEST FAILED "
                f"| user_id={user_id} "
                f"| conversation_id={conversation_id} "
                f"| error_type={type(exc).__name__} "
                f"| error={exc}"
            )

            raise

        # --------------------------------------------------
        # Completion
        # --------------------------------------------------

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000

        logger.success(
            "CHAT COMPLETE "
            f"| user_id={user_id} "
            f"| conversation_id={conversation_id} "
            f"| sources={len(sources)} "
            f"| duration={elapsed:.2f} ms"
        )

        result["user_message_id"] = user_message.id
        result["assistant_message_id"] = assistant_message.id

        return result


chat_service = ChatService()