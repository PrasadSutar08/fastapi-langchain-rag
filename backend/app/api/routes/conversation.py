from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from pydantic import BaseModel, Field

from app.api.deps import (
    CurrentUser,
    SessionDep,
)
from app.core.config import logger, settings
from app.crud.conversation_crud import ConversationCRUD
from app.crud.message_crud import MessageCRUD
from app.services.llm_service import llm_service
from app.services.redis_service import redis_service


router = APIRouter()


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ConversationAutoTitleRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


DEFAULT_CONVERSATION_TITLE = "New Conversation"


def _fallback_title(message: str) -> str:
    """Deterministic short title used if LLM title generation fails."""
    clean = " ".join(message.split())
    return clean[:48].rstrip() + ("…" if len(clean) > 48 else "")


def auto_title_rate_limit(request: Request):
    return redis_service.rate_limit(
        request=request,
        route_name="conversation_auto_title",
        limit=30,
        window_seconds=60,
    )


# ==========================================================
# LIST CONVERSATIONS
# ==========================================================

@router.get("")
async def list_conversations(
    current_user: CurrentUser,
    session: SessionDep,
):
    """
    List all conversations belonging to the authenticated user.
    """

    try:
        conversations = (
            await ConversationCRUD.list_for_user(
                session=session,
                user_id=current_user.id,
            )
        )

        logger.info(
            "Conversations listed "
            f"| user_id={current_user.id} "
            f"| count={len(conversations)}"
        )

        return {
            "data": conversations
        }

    except Exception:
        logger.exception(
            "Failed to list conversations "
            f"| user_id={current_user.id}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations.",
        )


# ==========================================================
# GET CONVERSATION
# ==========================================================

@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    current_user: CurrentUser,
    session: SessionDep,
):
    """
    Retrieve a conversation belonging to the authenticated user.
    """

    conversation = (
        await ConversationCRUD.get_for_user(
            session=session,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    )

    if conversation is None:

        logger.warning(
            "Conversation not found "
            f"| conversation_id={conversation_id} "
            f"| user_id={current_user.id}"
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return {
        "data": conversation
    }



@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: int,
    current_user: CurrentUser,
    session: SessionDep,
):
    """Return bounded message history for an owned conversation."""
    conversation = await ConversationCRUD.get_for_user(
        session=session,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = await MessageCRUD.get_history(
        session=session,
        conversation_id=conversation_id,
        limit=settings.MAX_HISTORY_MESSAGES,
    )
    return {"data": messages}

# ==========================================================
# CREATE CONVERSATION
# ==========================================================

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    current_user: CurrentUser,
    session: SessionDep,
):
    """
    Create a new conversation for the authenticated user.
    """

    try:

        conversation = (
            await ConversationCRUD.create(
                session=session,
                user_id=current_user.id,
            )
        )

        logger.info(
            "Conversation API created "
            f"| conversation_id={conversation.id} "
            f"| user_id={current_user.id}"
        )

        return {
            "data": conversation
        }

    except Exception:

        logger.exception(
            "Conversation API creation failed "
            f"| user_id={current_user.id}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation.",
        )


# ==========================================================
# RENAME CONVERSATION
# ==========================================================

@router.patch("/{conversation_id}")
async def rename_conversation(
    conversation_id: int,
    payload: ConversationRenameRequest,
    current_user: CurrentUser,
    session: SessionDep,
):
    """
    Rename a conversation belonging to the authenticated user.

    Used by the frontend to set the conversation title from a summary
    of the first user message once a conversation moves past its
    default "New Conversation" title.
    """

    conversation = (
        await ConversationCRUD.get_for_user(
            session=session,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    )

    if conversation is None:

        logger.warning(
            "Conversation rename requested for missing "
            f"conversation | conversation_id={conversation_id} "
            f"| user_id={current_user.id}"
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    try:

        conversation = await ConversationCRUD.rename(
            session=session,
            conversation=conversation,
            title=payload.title,
        )

        logger.info(
            "Conversation renamed "
            f"| conversation_id={conversation_id} "
            f"| user_id={current_user.id}"
        )

        return {
            "data": conversation
        }

    except Exception:

        logger.exception(
            "Conversation rename failed "
            f"| conversation_id={conversation_id} "
            f"| user_id={current_user.id}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rename conversation.",
        )


# ==========================================================
# AUTO-TITLE CONVERSATION (LLM-generated short title)
# ==========================================================

@router.post(
    "/{conversation_id}/auto-title",
    dependencies=[Depends(auto_title_rate_limit)],
)
async def auto_title_conversation(
    conversation_id: int,
    payload: ConversationAutoTitleRequest,
    current_user: CurrentUser,
    session: SessionDep,
):
    """
    Generate a short (2-6 word) title from a message using the local
    LLM, similar to how Claude.ai names new conversations, and save it.

    Only renames conversations still on the default title, so this is
    safe to call after every message without clobbering a title the
    user (or an earlier call) already set.
    """

    conversation = (
        await ConversationCRUD.get_for_user(
            session=session,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    if conversation.title != DEFAULT_CONVERSATION_TITLE:
        # Already titled (by a previous call, or the user renamed it
        # manually) -- nothing to do, and avoids a wasted LLM call.
        return {"data": conversation}

    try:
        title = await llm_service.generate_title(payload.message)
        if not title:
            title = _fallback_title(payload.message)
    except Exception:
        logger.warning(
            "Falling back to truncated title "
            f"| conversation_id={conversation_id} "
            f"| user_id={current_user.id}"
        )
        title = _fallback_title(payload.message)

    conversation = await ConversationCRUD.rename(
        session=session,
        conversation=conversation,
        title=title,
    )

    return {"data": conversation}


# ==========================================================
# DELETE MESSAGES FROM A GIVEN MESSAGE ONWARDS (used for edit)
# ==========================================================

@router.delete(
    "/{conversation_id}/messages/from/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_messages_from(
    conversation_id: int,
    message_id: int,
    current_user: CurrentUser,
    session: SessionDep,
):
    """
    Delete a message and everything sent after it in the conversation.

    Used when the user edits an earlier message in the UI: the
    original message, its reply, and anything after are removed so
    the edited text can be resent as a clean new turn.
    """

    conversation = (
        await ConversationCRUD.get_for_user(
            session=session,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    await MessageCRUD.delete_from(
        session=session,
        conversation_id=conversation_id,
        from_message_id=message_id,
    )

    return None


# ==========================================================
# DELETE CONVERSATION
# ==========================================================

@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: int,
    current_user: CurrentUser,
    session: SessionDep,
):
    """
    Delete a conversation belonging to the authenticated user.

    Messages belonging to the conversation are deleted first.
    """

    conversation = (
        await ConversationCRUD.get_for_user(
            session=session,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    )

    if conversation is None:

        logger.warning(
            "Conversation deletion requested for missing "
            f"conversation | conversation_id={conversation_id} "
            f"| user_id={current_user.id}"
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    try:

        await ConversationCRUD.delete(
            session=session,
            conversation=conversation,
        )

        logger.info(
            "Conversation API deleted "
            f"| conversation_id={conversation_id} "
            f"| user_id={current_user.id}"
        )

    except Exception:

        logger.exception(
            "Conversation API deletion failed "
            f"| conversation_id={conversation_id} "
            f"| user_id={current_user.id}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation.",
        )

    return None