import re

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict


class KeywordAwareRetriever(BaseRetriever):
    """
    Retriever that combines semantic similarity with
    lightweight keyword matching and relative relevance filtering.

    Semantic similarity finds conceptually related chunks.

    Keyword matching gives a small boost when important query
    terms actually appear in the chunk.

    Relative filtering removes weak candidates while keeping
    enough context for questions that require multiple chunks.
    """

    vector_store: VectorStore
    k: int = 5
    fetch_k: int = 20

    keyword_weight: float = 0.20

    relevance_threshold: float = 0.15

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:

        results = (
            self.vector_store.similarity_search_with_score(
                query=query,
                k=max(self.fetch_k, self.k),
            )
        )

        if not results:
            return []

        query_terms = self._extract_terms(query)

        ranked_results = []

        for document, semantic_score in results:

            keyword_score = (
                self._calculate_keyword_score(
                    document.page_content,
                    query_terms,
                )
            )

            #
            # PGVector similarity_search_with_score()
            # returns a distance where LOWER is better.
            #
            final_score = (
                float(semantic_score)
                - (
                    keyword_score
                    * self.keyword_weight
                )
            )

            ranked_results.append(
                {
                    "document": document,
                    "semantic_score": float(
                        semantic_score
                    ),
                    "keyword_score": keyword_score,
                    "final_score": final_score,
                }
            )

        #
        # Lower final_score = more relevant.
        #
        ranked_results.sort(
            key=lambda item: item["final_score"]
        )

        #
        # Always keep the best result.
        #
        best_score = ranked_results[0]["final_score"]

        #
        # Relative filtering.
        #
        # Example:
        #
        # best score = 0.20
        # threshold = 0.15
        #
        # maximum acceptable score:
        # 0.20 + 0.15 = 0.35
        #
        # This avoids hardcoding an absolute distance
        # threshold because embedding distance values
        # can vary between models and datasets.
        #
        maximum_score = (
            best_score
            + self.relevance_threshold
        )

        filtered_results = [
            item
            for item in ranked_results
            if item["final_score"]
            <= maximum_score
        ]

        #
        # Safety fallback:
        #
        # If filtering becomes too aggressive,
        # return at least the best few results.
        #
        if not filtered_results:

            filtered_results = (
                ranked_results[:1]
            )

        #
        # Respect configured TOP_K.
        #
        filtered_results = filtered_results[
            : self.k
        ]

        return [
            item["document"]
            for item in filtered_results
        ]

    @staticmethod
    def _extract_terms(
        query: str,
    ) -> list[str]:
        """
        Extract meaningful terms from the query.
        """

        terms = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower(),
        )

        stop_words = {
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
        }

        return [
            term
            for term in terms
            if term not in stop_words
        ]

    @staticmethod
    def _calculate_keyword_score(
        content: str,
        query_terms: list[str],
    ) -> float:
        """
        Calculate the percentage of query terms
        appearing in the document chunk.
        """

        if not query_terms:
            return 0.0

        content_lower = content.lower()

        matched_terms = sum(
            1
            for term in query_terms
            if re.search(
                rf"\b{re.escape(term)}\b",
                content_lower,
            )
        )

        return (
            matched_terms
            / len(query_terms)
        )


class RetrieverService:
    """
    Creates and manages the retriever instance.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chat_config: dict,
    ):
        self.vector_store = vector_store

        retrieval_config = chat_config.get(
            "RETRIEVAL",
            {},
        )

        k = retrieval_config.get(
            "TOP_K",
            5,
        )

        fetch_k = retrieval_config.get(
            "FETCH_K",
            20,
        )

        keyword_weight = retrieval_config.get(
            "KEYWORD_WEIGHT",
            0.20,
        )

        relevance_threshold = (
            retrieval_config.get(
                "RELEVANCE_THRESHOLD",
                0.15,
            )
        )

        self.retriever = KeywordAwareRetriever(
            vector_store=vector_store,
            k=k,
            fetch_k=fetch_k,
            keyword_weight=keyword_weight,
            relevance_threshold=relevance_threshold,
        )

    def get_retriever(self):
        return self.retriever