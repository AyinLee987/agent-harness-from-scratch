"""MCP_FETCH_IGNORE_ROBOTS_TXT wiring.

mcp-server-fetch respects robots.txt by default -- verified live that most
major sites (LinkedIn, Facebook, Reddit, Twitter/X) explicitly disallow
automated fetching in theirs, which is most of why a research Worker's
fetch calls fail so often in practice. _fetch_server_args() is the small
pure decision of whether to append --ignore-robots-txt; the actual MCP
subprocess launch (lifespan()) isn't unit-tested anywhere in this suite --
this is the testable seam.
"""

from __future__ import annotations

from app import server


def test_robots_txt_respected_by_default(monkeypatch):
    monkeypatch.delenv("MCP_FETCH_IGNORE_ROBOTS_TXT", raising=False)
    assert server._fetch_server_args(["tool", "run", "mcp-server-fetch"]) == [
        "tool", "run", "mcp-server-fetch",
    ]


def test_robots_txt_ignored_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("MCP_FETCH_IGNORE_ROBOTS_TXT", "1")
    assert server._fetch_server_args(["tool", "run", "mcp-server-fetch"]) == [
        "tool", "run", "mcp-server-fetch", "--ignore-robots-txt",
    ]


def test_robots_txt_flag_does_not_mutate_the_caller_s_list(monkeypatch):
    monkeypatch.delenv("MCP_FETCH_IGNORE_ROBOTS_TXT", raising=False)
    original = ["tool", "run", "mcp-server-fetch"]
    server._fetch_server_args(original)
    assert original == ["tool", "run", "mcp-server-fetch"]
