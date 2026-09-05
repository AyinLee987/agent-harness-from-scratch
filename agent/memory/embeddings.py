"""Embedding providers independent from chat-model implementations."""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from ..llm import BaseLLM
from ..retry import RetryPolicy, call_with_retry, client_kwargs
from .errors import MemoryConfigurationError


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class LLMEmbeddingProvider:
    """Compatibility adapter for existing ``BaseLLM.embed`` implementations."""

    def __init__(self, llm: BaseLLM, model_id: str | None = None) -> None:
        self._llm = llm
        self._model_id = model_id or f"legacy:{type(llm).__name__}"
        self._dimension: int | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise MemoryConfigurationError(
                "Embedding dimension is unknown until the first embedding call."
            )
        return self._dimension

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> List[float]:
        vector = list(self._llm.embed(text))
        if not vector:
            raise MemoryConfigurationError("Embedding provider returned an empty vector.")
        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise MemoryConfigurationError(
                f"Embedding dimension changed from {self._dimension} to {len(vector)}."
            )
        return vector


class OpenAICompatibleEmbeddingProvider:
    """Dedicated embedding client for OpenAI-compatible embedding endpoints.

    The chat model may use an entirely different provider. ``client`` is
    injectable for tests; otherwise the optional ``openai`` package is used.

    An injected ``client`` is used exactly as given -- ``retry_policy`` still
    governs :func:`~agent.retry.call_with_retry` around each request, but the
    caller owns that client's own timeout settings.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        provider_name: str = "openai-compatible",
        client: Any = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self.retry_policy = retry_policy or RetryPolicy()
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise MemoryConfigurationError(
                    "OpenAICompatibleEmbeddingProvider requires the 'openai' package."
                ) from exc
            kwargs: dict[str, Any] = dict(client_kwargs(self.retry_policy))
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
        self._client = client
        self.model = model
        self._model_id = f"{provider_name}:{model}"
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise MemoryConfigurationError(
                "Embedding dimension is unknown until the first embedding call."
            )
        return self._dimension

    def embed_query(self, text: str) -> List[float]:
        return self._embed_many([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        values = list(texts)
        return self._embed_many(values) if values else []

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        response = call_with_retry(
            lambda timeout: self._client.embeddings.create(
                model=self.model, input=texts, timeout=timeout
            ),
            policy=self.retry_policy,
            operation="embed",
        )
        items = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        vectors = [list(item.embedding) for item in items]
        if len(vectors) != len(texts):
            raise MemoryConfigurationError(
                f"Embedding endpoint returned {len(vectors)} vectors for {len(texts)} texts."
            )
        for vector in vectors:
            if not vector:
                raise MemoryConfigurationError(
                    "Embedding endpoint returned an empty vector."
                )
            if self._dimension is None:
                self._dimension = len(vector)
            elif len(vector) != self._dimension:
                raise MemoryConfigurationError(
                    f"Embedding dimension changed from {self._dimension} to {len(vector)}."
                )
        return vectors


def provider_dimension(provider: EmbeddingProvider, sample: List[float]) -> int:
    """Validate a vector against the provider's declared or lazy dimension."""

    try:
        dimension = provider.dimension
    except MemoryConfigurationError:
        dimension = len(sample)
    if len(sample) != dimension:
        raise MemoryConfigurationError(
            f"Embedding vector has {len(sample)} dimensions; expected {dimension}."
        )
    return dimension
