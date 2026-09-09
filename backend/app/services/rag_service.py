from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger, settings
from app.services.llm_service import llm_service
from app.services.metrics_service import metrics_service
from app.services.search_service import search_service


class RAGService:

    def __init__(self) -> None:

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an enterprise document question-answering assistant.

Rules:

1. Answer ONLY from the supplied document context.
2. Never use outside knowledge.
3. Never invent facts.
4. Preserve exact numbers, dates, names,
   limits, exclusions and conditions.
5. If, and only if, NONE of the question can be
   answered from the context, say exactly:

I couldn't find that information in the uploaded documents.

   Do not add this sentence after an answer that
   already addresses the question, even partially --
   answer what the context supports and stop there.

6. Be concise and directly answer the question.
7. Cite supporting context using [Source N]
   when useful.

Document context:
{context}
""",
                ),
                (
                    "human",
                    "{question}",
                ),
            ]
        )

    @staticmethod
    def _clean(content: str) -> str:
        return " ".join(
            (content or "").split()
        )

    _FALLBACK_LINE = (
        "I couldn't find that information in the uploaded documents."
    )

    @classmethod
    def _strip_redundant_fallback(cls, answer: str) -> str:
        """
        The model is instructed to answer only from context and to say
        the fallback line when nothing relevant is found. For questions
        with several parts, it sometimes answers the parts it *can*
        support and then tacks the fallback line on as a trailing
        sentence for the parts it can't -- which reads as a contradiction
        ("here's a full answer... I couldn't find that information").

        If the fallback line appears as a trailing sentence *after*
        substantial other content, drop it: the answer already stands
        on its own, and the caveat adds confusion rather than clarity.
        """

        stripped = answer.strip()

        if not stripped.endswith(cls._FALLBACK_LINE):
            return stripped

        remainder = stripped[: -len(cls._FALLBACK_LINE)].strip()

        # Only the fallback line was produced -- keep it as-is.
        if len(remainder) < 40:
            return stripped

        return remainder

    @staticmethod
    def _preview(answer: str, limit: int = 160) -> str:
        flat = " ".join(answer.split())
        if len(flat) <= limit:
            return flat
        return flat[:limit].rstrip() + "…"

    @staticmethod
    def _build_retrieval_query(
        question: str,
        chat_history: list[BaseMessage] | None,
    ) -> str:

        question = question.strip()

        if not chat_history:
            return question

        recent_user_messages = [
            message.content
            for message in chat_history[-4:]
            if (
                isinstance(message, HumanMessage)
                and isinstance(
                    message.content,
                    str,
                )
            )
        ]

        if not recent_user_messages:
            return question

        previous = recent_user_messages[-1].strip()

        # Avoid duplicating the current question.
        if previous.lower() == question.lower():
            return question

        return (
            f"Previous user question: {previous}\n"
            f"Current question: {question}"
        )

    def _build_context(
        self,
        documents: list[Document],
    ) -> str:

        parts: list[str] = []

        total_chars = 0

        for index, document in enumerate(
            documents[:settings.MAX_CONTEXT_DOCUMENTS],
            start=1,
        ):

            content = self._clean(
                document.page_content
            )

            if not content:
                continue

            metadata = document.metadata or {}

            filename = metadata.get(
                "filename",
                "Unknown document",
            )

            page = metadata.get("page")

            label = (
                f"[Source {index}: {filename}"
            )

            if page is not None:
                label += f", page {page}"

            label += "]"

            block = (
                f"{label}\n"
                f"{content}"
            )

            remaining = (
                settings.MAX_CONTEXT_CHARS
                - total_chars
            )

            if remaining <= 0:
                break

            if len(block) > remaining:
                block = block[:remaining]

            parts.append(block)

            total_chars += len(block)

        return "\n\n".join(parts)

    @staticmethod
    def _build_sources(
        documents: list[Document],
    ) -> list[dict[str, Any]]:

        sources = []

        seen = set()

        for document in documents:

            metadata = document.metadata or {}

            key = (
                metadata.get(
                    "document_id"
                ),
                metadata.get("page"),
                metadata.get(
                    "chunk_index"
                ),
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "document_id": metadata.get(
                        "document_id"
                    ),
                    "filename": metadata.get(
                        "filename"
                    ),
                    "page": metadata.get(
                        "page"
                    ),
                    "source": metadata.get(
                        "source"
                    ),
                    "chunk_index": metadata.get(
                        "chunk_index"
                    ),
                }
            )

        return sources

    async def ask(
        self,
        question: str,
        chat_history: list[BaseMessage] | None,
        user_id: int,
        top_k: int | None = None,
        search_type: str | None = None,
    ) -> dict[str, Any]:

        started = time.perf_counter()

        question = (
            question or ""
        ).strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        retrieval_started = time.perf_counter()

        retrieval_query = (
            self._build_retrieval_query(
                question,
                chat_history,
            )
        )

        effective_top_k = (
            top_k
            if top_k is not None
            else settings.RETRIEVAL_TOP_K
        )

        effective_search_type = (
            search_type
            if search_type is not None
            else settings.RETRIEVAL_SEARCH_TYPE
        )

        logger.info(
            "RAG retrieval started "
            f"| user_id={user_id} "
            f"| top_k={effective_top_k} "
            f"| search_type={effective_search_type}"
        )

        results = search_service.search(
            query=retrieval_query,
            user_id=user_id,
            top_k=effective_top_k,
            search_type=effective_search_type,
        )

        metrics_service.increment("rag_retrieval_total")

        retrieval_ms = (            time.perf_counter()
            - retrieval_started
        ) * 1000

        documents = [
            Document(
                page_content=result["content"],
                metadata={
                    "document_id": result.get(
                        "document_id"
                    ),
                    "filename": result.get(
                        "filename"
                    ),
                    "page": result.get(
                        "page"
                    ),
                    "source": result.get(
                        "source"
                    ),
                    "chunk_index": result.get(
                        "chunk_index"
                    ),
                },
            )
            for result in results
        ]

        sources = self._build_sources(
            documents
        )

        context = self._build_context(
            documents
        )

        if not context:

            return {
                "answer": (
                    "I couldn't find that information "
                    "in the uploaded documents."
                ),
                "documents": [],
                "sources": [],
            }

        messages = self.prompt.format_messages(
            context=context,
            question=question,
        )

        logger.info(
            "RAG LLM invocation started "
            f"| user_id={user_id} "
            f"| sources={len(sources)} "
            f"| context_chars={len(context)} "
            f"| retrieval_ms={retrieval_ms:.2f}"
        )

        llm_started = time.perf_counter()

        response = await llm_service.ainvoke(
            messages
        )

        metrics_service.increment("rag_llm_requests_total")

        llm_ms = (            time.perf_counter()
            - llm_started
        ) * 1000

        answer = getattr(
            response,
            "content",
            str(response),
        ).strip()

        answer = self._strip_redundant_fallback(answer)

        metrics_service.increment("rag_requests_total")

        total_ms = (            time.perf_counter()
            - started
        ) * 1000

        logger.success(
            "RAG completed "
            f"| user_id={user_id} "
            f"| sources={len(sources)} "
            f"| answer_chars={len(answer)} "
            f"| retrieval_ms={retrieval_ms:.2f} "
            f"| llm_ms={llm_ms:.2f} "
            f"| total_ms={total_ms:.2f}"
        )

        logger.info(
            "RAG ANSWER PREVIEW "
            f"| user_id={user_id} "
            f"| preview={self._preview(answer)!r}"
        )

        return {
            "answer": answer,
            "documents": documents,
            "sources": sources,
        }


rag_service = RAGService()