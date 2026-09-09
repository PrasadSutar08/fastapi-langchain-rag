from __future__ import annotations

import re
import time

from app.core.config import logger, settings
from app.services.embedding_service import embedding_service
from app.services.vector_store_service import VectorStoreService
from app.services.metrics_service import metrics_service


class SearchService:
    """
    Production-oriented retrieval service.

    Pipeline:

        vector retrieval
             ↓
        MMR diversity
             ↓
        lexical reranking
             ↓
        duplicate suppression
             ↓
        top-k
    """

    def __init__(self) -> None:
        self._vector_store_service: VectorStoreService | None = None
        self._vector_store = None

    def _get_store(self):
        if self._vector_store is None:

            started = time.perf_counter()

            embeddings = (
                embedding_service.get_embeddings()
            )

            self._vector_store_service = (
                VectorStoreService(embeddings)
            )

            self._vector_store = (
                self._vector_store_service
                .get_vector_store()
            )

            elapsed = (
                time.perf_counter() - started
            ) * 1000

            logger.info(
                "Search store initialized "
                f"| duration={elapsed:.2f} ms"
            )

        return self._vector_store

    def warmup(self) -> None:
        """
        Initialize embeddings + PGVector before
        serving user traffic.
        """

        started = time.perf_counter()

        self._get_store()

        elapsed = (
            time.perf_counter() - started
        ) * 1000

        logger.success(
            "Search service warmed up "
            f"| duration={elapsed:.2f} ms"
        )

    @staticmethod
    def _clean_content(content: str) -> str:
        return " ".join(
            (content or "").split()
        )

    @staticmethod
    def _terms(query: str) -> list[str]:

        raw = re.findall(
            r"\b[\w-]+\b",
            query.lower(),
        )

        stop = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "to",
            "was",
            "what",
            "when",
            "where",
            "which",
            "who",
            "with",
            "does",
            "do",
            "can",
            "could",
            "would",
            "this",
            "that",
            "about",
        }

        return [
            token
            for token in raw
            if token not in stop
            and len(token) > 1
        ]

    @classmethod
    def _keyword_score(
        cls,
        content: str,
        terms: list[str],
    ) -> float:

        if not terms:
            return 0.0

        text = content.lower()

        matches = 0

        for term in terms:

            if re.search(
                rf"\b{re.escape(term)}\b",
                text,
            ):
                matches += 1

        return matches / len(terms)

    @classmethod
    def _rerank(
        cls,
        candidates,
        query: str,
    ):
        terms = cls._terms(query)

        ranked = []

        for position, (
            document,
            distance,
        ) in enumerate(candidates):

            keyword = cls._keyword_score(
                document.page_content,
                terms,
            )

            if distance is not None:

                semantic_score = 1.0 / (
                    1.0 + max(distance, 0.0)
                )

            else:

                # MMR already ordered these.
                # Give earlier candidates higher base score.
                semantic_score = 1.0 / (
                    1.0 + position
                )

            final_score = (
                semantic_score * 0.85
                + keyword * 0.15
            )

            ranked.append(
                (
                    final_score,
                    position,
                    document,
                    distance,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return ranked

    def search(
        self,
        query: str,
        user_id: int,
        top_k: int = 5,
        search_type: str = "mmr",
        document_id: int | None = None,
        filename: str | None = None,
        is_superuser: bool = False,
    ) -> list[dict]:

        started = time.perf_counter()

        query = (query or "").strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if not 1 <= top_k <= 20:
            raise ValueError(
                "top_k must be between 1 and 20."
            )

        if search_type not in {
            "similarity",
            "mmr",
        }:
            raise ValueError(
                "Unsupported search type."
            )

        store = self._get_store()

        metadata_filter = (
            VectorStoreService.build_user_filter(
                user_id=user_id,
                is_superuser=is_superuser,
                document_id=document_id,
                filename=filename,
            )
        )

        fetch_k = max(
            settings.RETRIEVAL_FETCH_K,
            top_k * 5,
        )

        if search_type == "similarity":

            raw = (
                store
                .similarity_search_with_score(
                    query=query,
                    k=fetch_k,
                    filter=metadata_filter,
                )
            )

            candidates = [
                (
                    document,
                    float(distance),
                )
                for document, distance in raw
            ]

        else:

            docs = (
                store
                .max_marginal_relevance_search(
                    query=query,
                    k=top_k,
                    fetch_k=fetch_k,
                    lambda_mult=(
                        settings.RETRIEVAL_MMR_LAMBDA
                    ),
                    filter=metadata_filter,
                )
            )

            candidates = [
                (document, None)
                for document in docs
            ]

        ranked = self._rerank(
            candidates,
            query,
        )

        results = []
        seen = set()

        for (
            final_score,
            _position,
            document,
            distance,
        ) in ranked:

            metadata = document.metadata or {}

            key = (
                metadata.get("document_id"),
                metadata.get("chunk_index"),
            )

            if key in seen:
                continue

            seen.add(key)

            results.append(
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
                    "content": self._clean_content(
                        document.page_content
                    ),
                    "score": final_score,
                    "vector_distance": distance,
                    "source": metadata.get(
                        "source"
                    ),
                    "chunk_index": metadata.get(
                        "chunk_index"
                    ),
                }
            )

            if len(results) >= top_k:
                break

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        metrics_service.increment("search_requests_total")

        logger.info(
            "Vector search completed "
            f"| user_id={user_id} "
            f"| type={search_type} "
            f"| fetch_k={fetch_k} "
            f"| results={len(results)} "
            f"| duration={elapsed_ms:.2f} ms"
        )

        return results


search_service = SearchService()