import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import CurrentUser, SessionDep
from app.core.config import logger
from app.schemas.chat_schema import ChatBody
from app.services.chat_service import chat_service
from app.services.redis_service import redis_service


router = APIRouter(
    prefix="/qa",
    tags=["QA"],
)


def qa_rate_limit(request: Request):
    return redis_service.rate_limit(
        request=request,
        route_name="qa_chat",
        limit=20,
        window_seconds=60,
    )


@router.post(
    "/chat",
    dependencies=[Depends(qa_rate_limit)],
)
async def chat_action(
    request: ChatBody,
    current_user: CurrentUser,
    session: SessionDep,
):
    """
    Ask a question against the authenticated user's documents.
    """

    start_time = time.perf_counter()

    # IMPORTANT:
    # Copy primitive values immediately.
    # Do not access current_user.id inside exception handlers.
    user_id = current_user.id
    conversation_id = request.conversation_id
    question = request.message

    logger.info(
        f"QA REQUEST STARTED "
        f"| user_id={user_id} "
        f"| conversation_id={conversation_id} "
        f"| question={question!r}"
    )

    try:
        result = await chat_service.chat(
            session=session,
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
        )

        answer = result.get("answer", "")
        documents = result.get("documents", [])

        sources = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata or {}

            source = {
                "document_id": metadata.get("document_id"),
                "filename": metadata.get("filename"),
                "page": metadata.get("page"),
                "source": metadata.get("source"),
                "chunk_index": metadata.get("chunk_index"),
            }

            sources.append(source)

            logger.info(
                f"QA SOURCE #{index} "
                f"| document_id={source['document_id']} "
                f"| filename={source['filename']} "
                f"| page={source['page']} "
                f"| chunk_index={source['chunk_index']}"
            )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.success(
            f"QA REQUEST COMPLETED "
            f"| user_id={user_id} "
            f"| conversation_id={conversation_id} "
            f"| documents={len(documents)} "
            f"| duration={elapsed_ms:.2f} ms"
        )

        return {
            "data": {
                "conversation_id": conversation_id,
                "answer": answer,
                "sources": sources,
                "user_message_id": result.get("user_message_id"),
                "assistant_message_id": result.get("assistant_message_id"),
            }
        }

    except ValueError as exc:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.warning(
            f"QA REQUEST VALIDATION ERROR "
            f"| user_id={user_id} "
            f"| conversation_id={conversation_id} "
            f"| duration={elapsed_ms:.2f} ms "
            f"| error={exc}"
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # IMPORTANT:
        # Never access current_user.id here.
        # user_id is already a primitive int.
        logger.exception(
            f"QA REQUEST ERROR "
            f"| user_id={user_id} "
            f"| conversation_id={conversation_id} "
            f"| duration={elapsed_ms:.2f} ms "
            f"| error_type={type(exc).__name__} "
            f"| error={exc}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="QA request failed. Check server logs for the underlying error.",
        ) from exc