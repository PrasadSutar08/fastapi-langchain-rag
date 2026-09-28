from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

try:
    # Optional dependency: only required when LLM_PROVIDER=groq. Kept
    # as a soft import so `pip install -r requirements.txt` (which
    # now includes langchain-groq) works, while still failing with a
    # clear error rather than an ImportError traceback if someone
    # strips it out and sets LLM_PROVIDER=groq anyway.
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None  # type: ignore[assignment]

from app.core.config import logger, settings


class LLMService:
    """
    Builds and reuses the chat model used for both RAG answers and
    conversation titling.

    LLM_PROVIDER switches which backend this talks to:
      - "ollama" (default): self-hosted local inference. Used for
        local development and by anyone running this project
        open-source against their own GPU/machine.
      - "groq": hosted inference via Groq's free API. Used for the
        publicly deployed demo, so it stays fast and always-on
        without needing a self-hosted GPU/VM running 24/7.

    Both code paths always exist; only _build_llm()'s branch changes
    based on config. Nothing else in the app needs to know which
    provider is active -- callers just use get_llm() / ainvoke().
    """

    _llm: BaseChatModel | None = None
    _title_llm: BaseChatModel | None = None
    _semaphore: asyncio.Semaphore | None = None

    @staticmethod
    def _build_llm(
        *,
        num_predict: int | None = None,
        temperature: float | None = None,
    ) -> BaseChatModel:
        provider = settings.LLM_PROVIDER.strip().lower()

        if provider == "groq":
            if ChatGroq is None:
                raise RuntimeError(
                    "LLM_PROVIDER=groq but the 'langchain-groq' package "
                    "is not installed. Run: pip install langchain-groq"
                )
            if not settings.GROQ_API_KEY:
                raise RuntimeError(
                    "LLM_PROVIDER=groq but GROQ_API_KEY is not set. Get "
                    "a free key at https://console.groq.com/keys and "
                    "set it in .env."
                )

            return ChatGroq(
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=(
                    settings.OLLAMA_TEMPERATURE
                    if temperature is None
                    else temperature
                ),
                max_tokens=(
                    settings.OLLAMA_MAX_TOKENS
                    if num_predict is None
                    else num_predict
                ),
                timeout=settings.GROQ_TIMEOUT_SECONDS,
            )

        if provider != "ollama":
            logger.warning(
                f"Unrecognized LLM_PROVIDER={settings.LLM_PROVIDER!r}, "
                "falling back to 'ollama'."
            )

        kwargs: dict[str, Any] = {
            "model": settings.OLLAMA_MODEL,
            "base_url": settings.OLLAMA_BASE_URL,
            "temperature": (
                settings.OLLAMA_TEMPERATURE if temperature is None else temperature
            ),
            "num_predict": (
                settings.OLLAMA_MAX_TOKENS if num_predict is None else num_predict
            ),
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            "num_gpu": settings.OLLAMA_NUM_GPU,
        }
        if settings.OLLAMA_NUM_THREAD is not None:
            kwargs["num_thread"] = settings.OLLAMA_NUM_THREAD

        return ChatOllama(**kwargs)

    def get_llm(self) -> BaseChatModel:
        if self.__class__._llm is None:
            active_model = (
                settings.GROQ_MODEL
                if settings.LLM_PROVIDER.strip().lower() == "groq"
                else settings.OLLAMA_MODEL
            )
            logger.info(
                "Initializing LLM "
                f"| provider={settings.LLM_PROVIDER} "
                f"| model={active_model}"
            )
            self.__class__._llm = self._build_llm()
            logger.success("LLM client initialized.")
        return self.__class__._llm

    def _get_title_llm(self) -> BaseChatModel:
        """
        Separate low-token-budget model instance used only for title
        generation.

        NOTE: overriding num_predict/temperature via `.bind()` on the
        shared instance does NOT work reliably across langchain-ollama
        versions -- some versions forward bound kwargs straight to the
        underlying ollama.Client().chat() call, which rejects them
        ("Client.chat() got an unexpected keyword argument
        'num_predict'"), silently breaking title generation. Building
        a second instance with these fields set at construction time
        (the same way get_llm() does) avoids that entirely, and works
        identically for the Groq path.
        """
        if self.__class__._title_llm is None:
            self.__class__._title_llm = self._build_llm(
                num_predict=24,
                temperature=0.0,
            )
        return self.__class__._title_llm

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(settings.OLLAMA_MAX_CONCURRENCY)
        return cls._semaphore

    async def ainvoke(self, messages):
        started = time.perf_counter()
        semaphore = self._get_semaphore()
        timeout = (
            settings.GROQ_TIMEOUT_SECONDS
            if settings.LLM_PROVIDER.strip().lower() == "groq"
            else settings.OLLAMA_TIMEOUT_SECONDS
        )
        try:
            async with semaphore:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.get_llm().invoke, messages),
                    timeout=timeout,
                )
            logger.info(
                f"LLM request completed | duration={(time.perf_counter() - started) * 1000:.2f} ms"
            )
            return result
        except asyncio.TimeoutError as exc:
            logger.error(f"LLM request timed out | timeout={timeout}s")
            raise RuntimeError("The AI service timed out.") from exc
        except Exception as exc:
            logger.exception(f"LLM request failed | error={exc}")
            raise RuntimeError("The AI service is unavailable.") from exc

    TITLE_SYSTEM_PROMPT = (
        "You name chat conversations. Given the user's message, output "
        "a short title (2-5 words) that names its TOPIC. "
        "Never repeat the message verbatim or near-verbatim. Never "
        "phrase the title as a question, even if the message is one. "
        "Plain text only. No quotes. No trailing punctuation. No "
        "markdown. Title Case. Return only the title and nothing else."
    )

    async def generate_title(self, question: str) -> str:
        """
        Generate a short (2-6 word) conversation title from the user's
        first message, using a low token budget so it stays fast
        regardless of which provider is active.
        """

        started = time.perf_counter()
        semaphore = self._get_semaphore()

        # Few-shot examples so the model has a concrete pattern to
        # follow -- without them, small instruct models tend to just
        # restate the question back as the "title".
        messages = [
            SystemMessage(content=self.TITLE_SYSTEM_PROMPT),
            HumanMessage(content="what is this document about?"),
            AIMessage(content="Document Overview"),
            HumanMessage(content="how many sick leaves do I get in a year?"),
            AIMessage(content="Sick Leave Policy"),
            HumanMessage(content="is pinodinol sinus treatment covered?"),
            AIMessage(content="Pinodinol Coverage Check"),
            HumanMessage(content=question[:500]),
        ]

        try:
            async with semaphore:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._get_title_llm().invoke, messages),
                    timeout=30,
                )

            title = getattr(result, "content", str(result)).strip()
            title = title.strip(" \n\"'.?!")

            logger.info(
                "Title generation completed "
                f"| duration={(time.perf_counter() - started) * 1000:.2f} ms "
                f"| title={title!r}"
            )

            return title

        except asyncio.TimeoutError as exc:
            logger.warning("Title generation timed out")
            raise RuntimeError("Title generation timed out.") from exc
        except Exception as exc:
            logger.warning(f"Title generation failed | error={exc}")
            raise RuntimeError("Title generation failed.") from exc

    async def health(self) -> bool:
        provider = settings.LLM_PROVIDER.strip().lower()

        if provider == "groq":
            if not settings.GROQ_API_KEY:
                return False
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(
                        "https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                    )
                return response.status_code == 200
            except Exception as exc:
                logger.warning(f"Groq readiness check failed | error={exc}")
                return False

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(
                    f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
                )
            if response.status_code != 200:
                return False
            models = {item.get("name") for item in response.json().get("models", [])}
            return settings.OLLAMA_MODEL in models
        except Exception as exc:
            logger.warning(f"Ollama readiness check failed | error={exc}")
            return False

    @classmethod
    def reset(cls) -> None:
        cls._llm = None
        cls._title_llm = None
        cls._semaphore = None


llm_service = LLMService()
