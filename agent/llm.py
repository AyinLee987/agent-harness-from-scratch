"""Thin LLM client wrappers.

This module defines a small, provider-agnostic interface (:class:`BaseLLM`) for
chat completions with tool-calling, plus two concrete implementations:

* :class:`MockLLM` -- a deterministic, dependency-free "LLM" that is good enough
  to drive the ReAct loop over the sample eval tasks. It lets the whole repo run
  (and CI pass) **without an API key**.
* :class:`OpenAILLM` -- a real client that talks to the OpenAI chat-completions
  API when ``OPENAI_API_KEY`` is set and the ``openai`` package is installed.

Keeping the surface area tiny is deliberate: the point of this project is to show
the agent *internals*, not to wrap every provider feature.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Usage:
    """Token accounting for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Normalized response returned by every :class:`BaseLLM`."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token).

    Good enough for the budget guard; avoids pulling in ``tiktoken``.
    """

    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseLLM(ABC):
    """Provider-agnostic chat interface used by the agent."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Run one chat completion, optionally exposing ``tools``."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for ``text`` (used by long-term memory)."""

    async def astream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """Async generator that yields SSE-style dicts during a chat completion.

        Each yield is a dict with ``type`` and ``data`` keys, suitable for
        Server-Sent Events in a web server.  Subclasses that support streaming
        override this; the default falls back to a single-text yield from
        :meth:`chat`.
        """
        response = self.chat(messages, tools)
        if response.content:
            yield {"type": "text", "data": response.content}
        for tc in response.tool_calls:
            yield {
                "type": "tool_call",
                "data": {"id": tc.id, "name": tc.name, "arguments": tc.arguments},
            }


# ---------------------------------------------------------------------------
# Mock implementation (zero-dependency, deterministic)
# ---------------------------------------------------------------------------
# Canned "knowledge base" so web-search-style tasks are deterministic.
_MOCK_SEARCH_KB: Dict[str, str] = {
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "tallest mountain": "Mount Everest is the tallest mountain on Earth at 8,849 m.",
    "speed of light": "The speed of light is approximately 299,792 km/s.",
    "python creator": "Python was created by Guido van Rossum.",
}

# Dropped from mock embeddings so similarity reflects content words.
_EMBED_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "with", "as", "at", "by", "from", "this", "that", "it",
    "what", "which", "who", "how", "when", "where", "why", "do", "does", "did",
    "about", "tell", "me", "my", "your", "i", "you",
}

_WORD_OPS = {
    "plus": "+",
    "add": "+",
    "added to": "+",
    "minus": "-",
    "subtract": "-",
    "times": "*",
    "multiplied by": "*",
    "multiply": "*",
    "divided by": "/",
    "divide": "/",
}


class MockLLM(BaseLLM):
    """A deterministic stand-in for a real LLM.

    The behaviour is intentionally simple but it is enough to exercise the full
    think -> act -> observe loop:

    1. If the latest turn contains a tool result (role ``tool``), synthesize a
       final answer that embeds the observation(s).
    2. Otherwise, inspect the user's request and decide which registered tool to
       call, extracting arguments heuristically.
    3. If no tool fits, answer directly.

    It never relies on randomness, so eval runs are reproducible.
    """

    def __init__(self, **_: Any) -> None:
        # Kept for API parity with OpenAILLM (e.g. model/temperature kwargs).
        pass

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        tools = tools or []
        tool_names = {t["function"]["name"] for t in tools}
        prompt_tokens = estimate_tokens(json.dumps(messages))

        # (0) Grader mode: if asked to act as an LLM-as-judge, return PASS/FAIL.
        if self._is_grader_prompt(messages):
            verdict = self._grade(messages)
            return LLMResponse(content=verdict, usage=Usage(prompt_tokens, 1))

        # (1) Do we already have observations to answer from?
        observations = [m for m in messages if m.get("role") == "tool"]
        if observations:
            latest = observations[-1]
            if (
                latest.get("name") == "spawn_subagent"
                and "wait_subagents" in tool_names
            ):
                try:
                    spawned = json.loads(str(latest.get("content") or ""))
                    task_id = spawned["task_id"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
                else:
                    return LLMResponse(
                        tool_calls=[
                            ToolCall(
                                id="call_wait_subagents",
                                name="wait_subagents",
                                arguments={
                                    "task_ids": [task_id],
                                    "timeout_seconds": 30.0,
                                },
                            )
                        ],
                        usage=Usage(prompt_tokens, 8),
                    )
            answer = self._synthesize_answer(messages, observations)
            return LLMResponse(
                content=answer,
                usage=Usage(prompt_tokens, estimate_tokens(answer)),
            )

        # (2) Decide on a tool based on the most recent user message.
        user_text = self._last_user_text(messages)
        decision = self._decide_tool(user_text, tool_names)
        if decision is not None:
            name, args = decision
            call = ToolCall(id=f"call_{name}", name=name, arguments=args)
            return LLMResponse(
                tool_calls=[call],
                usage=Usage(prompt_tokens, 8),
            )

        # (3) Nothing fits -> answer directly.
        answer = f"I don't have a tool to handle: {user_text!r}."
        return LLMResponse(
            content=answer,
            usage=Usage(prompt_tokens, estimate_tokens(answer)),
        )

    def embed(self, text: str) -> List[float]:
        """Deterministic hashing bag-of-words embedding (256 dims).

        Not semantically rich, but stable and dependency-free, which is all the
        in-memory vector store needs for a runnable demo. Common stopwords are
        dropped so similarity is driven by content words, not "the"/"is".
        """

        dims = 256
        vec = [0.0] * dims
        for token in re.findall(r"\w+", text.lower()):
            if token in _EMBED_STOPWORDS:
                continue
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _is_grader_prompt(messages: List[Dict[str, Any]]) -> bool:
        for m in messages:
            if m.get("role") == "system":
                content = str(m.get("content") or "").lower()
                if "grader" in content or "'pass' or" in content:
                    return True
        return False

    @staticmethod
    def _grade(messages: List[Dict[str, Any]]) -> str:
        """Heuristic judge: FAIL on empty/error answers, PASS otherwise."""

        answer = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                answer = str(m.get("content") or "").lower()
                break
        if not answer or "error" in answer or "stopped" in answer or "don't have a tool" in answer:
            return "FAIL"
        return "PASS"

    @staticmethod
    def _last_user_text(messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content") or "")
        return ""

    def _decide_tool(
        self, text: str, tool_names: set
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        lower = text.lower()

        if "spawn_subagent" in tool_names and any(
            keyword in lower
            for keyword in (
                "delegate",
                "subagent",
                "sub-agent",
                "worker agent",
                "委派",
                "子任务",
                "子代理",
            )
        ):
            role = "researcher" if any(
                keyword in lower
                for keyword in ("research", "search", "fetch", "web", "调查", "检索")
            ) else "analyst"
            return "spawn_subagent", {"role": role, "task": text}

        if "calculator" in tool_names:
            expr = self._extract_expression(text)
            if expr:
                return "calculator", {"expression": expr}

        if "datetime" in tool_names and any(
            kw in lower for kw in ("time", "date", "day", "today", "now")
        ):
            return "datetime", {}

        if "memory_search" in tool_names and any(
            kw in lower
            for kw in (
                "memory", "remember", "recall", "past", "previous",
                "stored", "history", "learned", "know about",
                "refund", "policy", "cost", "price", "product",
                "enterprise", "customer", "support", "office",
                "founded", "headquartered", "ceo",
            )
        ):
            return "memory_search", {"query": text}

        fetch_tools = [
            name for name in tool_names if name == "fetch" or name.endswith("__fetch")
        ]
        url_match = re.search(r"https?://[^\s<>'\"]+", text)
        if fetch_tools and ("fetch" in lower or url_match):
            return fetch_tools[0], {"url": url_match.group(0) if url_match else text}

        if "web_search" in tool_names and any(
            kw in lower
            for kw in ("search", "who", "what is", "capital", "tallest", "speed of")
        ):
            return "web_search", {"query": text}

        return None

    @staticmethod
    def _extract_expression(text: str) -> Optional[str]:
        """Pull an arithmetic expression out of free text."""

        lowered = text.lower()
        for word, op in _WORD_OPS.items():
            lowered = lowered.replace(word, op)
        # Keep only math-ish characters.
        candidate = re.sub(r"[^0-9\.\+\-\*\/\(\)% ]", " ", lowered)
        candidate = candidate.strip()
        # Require at least one operator and one digit to count as an expression.
        if re.search(r"\d", candidate) and re.search(r"[\+\-\*\/%]", candidate):
            return re.sub(r"\s+", " ", candidate).strip()
        return None

    @staticmethod
    def _synthesize_answer(
        messages: List[Dict[str, Any]], observations: List[Dict[str, Any]]
    ) -> str:
        latest = str(observations[-1].get("content") or "").strip()
        return f"Based on the tool result, the answer is: {latest}"


# ---------------------------------------------------------------------------
# Real OpenAI implementation
# ---------------------------------------------------------------------------
class OpenAILLM(BaseLLM):
    """Wraps the OpenAI chat-completions API.

    Imported lazily so the repo stays importable without the ``openai`` package.

    Set ``OPENAI_BASE_URL`` to point at any OpenAI-compatible endpoint
    (DeepSeek, Ollama, vLLM, etc.) without switching class.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        embed_model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only w/o openai
            raise RuntimeError(
                "The 'openai' package is required for OpenAILLM. "
                "Install it or use MockLLM."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; use MockLLM instead.")

        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.embed_model = embed_model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message

        tool_calls: List[ToolCall] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )

        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return LLMResponse(content=choice.content, tool_calls=tool_calls, usage=usage)

    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self.embed_model, input=text)
        return list(resp.data[0].embedding)


# ---------------------------------------------------------------------------
# DeepSeek implementation
# ---------------------------------------------------------------------------
class DeepSeekLLM(OpenAILLM):
    """DeepSeek API client — OpenAI-compatible, preset for DeepSeek models.

    DeepSeek's API is fully compatible with the OpenAI chat-completions protocol,
    so this is a thin preset over :class:`OpenAILLM`.  Set ``DEEPSEEK_API_KEY``
    (or ``OPENAI_API_KEY``) in your environment.

    .. code-block:: python

        from agent import DeepSeekLLM, ReActAgent

        llm = DeepSeekLLM()                 # defaults to deepseek-chat (V3)
        llm = DeepSeekLLM(model="deepseek-reasoner")  # DeepSeek-R1
        agent = ReActAgent(llm=llm, tools=registry)

    Caveats
    -------
    - DeepSeek does **not** offer an embeddings API.  ``embed()`` falls back to
      the deterministic hash-based embedding used by :class:`MockLLM`, so
      :class:`~agent.memory.LongTermMemory` works without external services.
      Swap in a real embedding provider for production.
    - DeepSeek-R1 (``deepseek-reasoner``) returns ``reasoning_content`` before
      the final answer; this client merges it into ``content`` so the agent sees
      the full chain-of-thought.
    """

    MODELS = {
        "chat": "deepseek-chat",           # DeepSeek-V3 (fast, general-purpose)
        "reasoner": "deepseek-reasoner",   # DeepSeek-R1 (reasoning / CoT)
    }

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required for DeepSeekLLM."
            ) from exc

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) environment variable."
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model or self.MODELS["chat"]
        self.temperature = temperature
        self.embed_model = ""  # not used — embed() is overridden below

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message

        # DeepSeek-R1 emits reasoning_content before the final answer.
        # Merge it so the agent retains the chain-of-thought in context.
        reasoning = getattr(choice, "reasoning_content", None)
        content = choice.content or ""
        if reasoning:
            content = f"[reasoning]\n{reasoning}\n[/reasoning]\n\n{content}"

        tool_calls: List[ToolCall] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )

        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)

    async def astream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """Stream tokens from DeepSeek via SSE-compatible async generator."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError(
                "The 'openai' package is required for DeepSeekLLM streaming."
            )

        aclient = AsyncOpenAI(
            api_key=self._client.api_key,
            base_url=str(self._client.base_url),
        )
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await aclient.chat.completions.create(**kwargs)

        # Accumulate tool_call deltas across chunks.
        tool_call_buf: Dict[int, Dict[str, Any]] = {}
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text token.
            if delta.content:
                yield {"type": "text", "data": delta.content}

            # Tool-call fragments (DeepSeek streams these as JSON fragments).
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_buf:
                        tool_call_buf[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_call_buf[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_buf[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_call_buf[idx]["arguments"] += tc.function.arguments

            # Emit completed tool calls on the final chunk.
            if chunk.choices[0].finish_reason == "tool_calls":
                for idx in sorted(tool_call_buf):
                    tc = tool_call_buf[idx]
                    try:
                        args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    yield {
                        "type": "tool_call",
                        "data": {"id": tc["id"], "name": tc["name"], "arguments": args},
                    }

    def embed(self, text: str) -> List[float]:
        """Fallback to deterministic hash-based embedding.

        DeepSeek does not provide an embeddings API.  This uses the same
        lightweight deterministic method as :class:`MockLLM` so that
        :class:`~agent.memory.LongTermMemory` stays functional.  Swap in
        an OpenAI / Voyage / Cohere embedding call for production use.
        """
        return MockLLM().embed(text)


# ---------------------------------------------------------------------------
# Bailian (Alibaba Bailian / DashScope) implementation
# ---------------------------------------------------------------------------
class BailianLLM(OpenAILLM):
    """Alibaba Bailian (百炼) API client — OpenAI-compatible, preset for Bailian models.

    百炼's API is fully compatible with the OpenAI chat-completions protocol,
    so this is a thin preset over :class:`OpenAILLM`.  Set ``BAILIAN_API_KEY``
    (or ``OPENAI_API_KEY``) in your environment.

    .. code-block:: python

        from agent import BailianLLM, ReActAgent

        llm = BailianLLM()                 # defaults to qwen-plus
        llm = BailianLLM(model="qwen-max") # use a different model
        agent = ReActAgent(llm=llm, tools=registry)

    Embeddings
    ----------
    Bailian provides an embeddings API via ``text-embedding-v3`` (1024-dim).
    The ``embed()`` method calls the real API — no fallback needed.

    Env vars
    --------
    - ``BAILIAN_API_KEY``: Your Bailian API key (sk-...). Falls back to
      ``OPENAI_API_KEY``.
    - ``BAILIAN_BASE_URL``: Override the default base URL. Defaults to
      ``https://dashscope.aliyuncs.com/compatible-mode/v1``.
    - ``BAILIAN_EMBED_MODEL``: Override the embedding model. Defaults to
      ``text-embedding-v3``.
    """

    MODELS = {
        "chat": "qwen-plus",
        "embed": "text-embedding-v3",
    }

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        embed_model: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for BailianLLM."
            ) from exc

        api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set BAILIAN_API_KEY (or OPENAI_API_KEY) environment variable."
            )

        base_url = os.environ.get(
            "BAILIAN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or self.MODELS["chat"]
        self.temperature = temperature
        self.embed_model = (
            embed_model
            or os.environ.get("BAILIAN_EMBED_MODEL")
            or self.MODELS["embed"]
        )
