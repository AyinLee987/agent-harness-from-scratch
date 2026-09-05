"""Test-wide safety net: no test may reach a real model provider.

``app/server.py`` calls ``load_dotenv()`` at import, and ``_auto_llm``
selects DeepSeek (or OpenAI) the moment the corresponding key is present.
So on any machine with a populated ``.env`` -- i.e. the developer's --
a server test that forgets to patch the model spends real money and
produces results that depend on a live third party. A test that passes for
that reason is not a passing test.

The autouse fixture below makes ``MockLLM`` the default for every test.
A test that wants a specific fake still overrides it: its own
``monkeypatch.setattr`` runs after this one and wins.

Scope, deliberately: it patches the **server's** model factories and
nothing else. It does not unset the API keys. A handful of tests
(``test_tool_scaling*``, ``test_hierarchical_agent``) are *intentionally*
live -- they measure a real model's tool-selection accuracy, which no fake
can stand in for -- and they gate themselves on a key being present. Those
are a deliberate choice to spend money; the failure this guards is the
accidental one, where a test never meant to touch a provider quietly does.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _server_tests_use_a_fake_model(monkeypatch):
    """Point the server's model factories at ``MockLLM`` for every test."""

    import app.server as server
    from agent import MockLLM

    monkeypatch.setattr(server, "_build_llm", lambda: MockLLM(), raising=False)
    # Delegates rather than returning its own MockLLM, mirroring the real
    # ``models.fast: provider: auto`` fallback -- otherwise a test that
    # patches only ``_build_llm`` (most of them) would silently lose control
    # of the fast tier, which is the exact trap ``_build_fast_llm``'s own
    # docstring exists to avoid.
    monkeypatch.setattr(
        server, "_build_fast_llm", lambda: server._build_llm(), raising=False
    )
