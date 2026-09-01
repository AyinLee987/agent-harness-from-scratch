"""Tests for app/config.py — the config/agent.yaml loader."""

from __future__ import annotations

from app.config import AgentConfig, load_agent_config


def test_missing_file_falls_back_to_every_shipped_default(tmp_path):
    config = load_agent_config(tmp_path / "does-not-exist.yaml")

    assert config == AgentConfig()


def test_empty_file_falls_back_to_every_shipped_default(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("", encoding="utf-8")

    assert load_agent_config(path) == AgentConfig()


def test_partial_overrides_leave_untouched_fields_at_their_default(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "leader:\n  max_steps: 25\nworker:\n  max_steps: 4\n",
        encoding="utf-8",
    )

    config = load_agent_config(path)

    assert config.leader.max_steps == 25
    assert config.leader.max_tokens == AgentConfig().leader.max_tokens
    assert config.worker.max_steps == 4
    assert config.run_budget == AgentConfig().run_budget


def test_every_section_can_be_overridden(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
leader:
  max_steps: 20
  max_tokens: 50000
worker:
  max_steps: 5
run_budget:
  max_subagents: 3
  max_parallel_tasks: 2
  max_depth: 2
  max_repeated_task: 2
  subagent_timeout_seconds: 30.0
react_loop:
  max_tool_retries: 2
  loop_same_call_limit: 5
  compress_at_fraction: 0.8
session:
  recent_window: 6
  summarize_beyond: 12
""",
        encoding="utf-8",
    )

    config = load_agent_config(path)

    assert config.leader.max_steps == 20
    assert config.leader.max_tokens == 50000
    assert config.worker.max_steps == 5
    assert config.run_budget.max_subagents == 3
    assert config.run_budget.max_parallel_tasks == 2
    assert config.run_budget.max_depth == 2
    assert config.run_budget.max_repeated_task == 2
    assert config.run_budget.subagent_timeout_seconds == 30.0
    assert config.react_loop.max_tool_retries == 2
    assert config.react_loop.loop_same_call_limit == 5
    assert config.react_loop.compress_at_fraction == 0.8
    assert config.session.recent_window == 6
    assert config.session.summarize_beyond == 12


def test_unknown_keys_are_ignored_rather_than_raising(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "leader:\n  max_steps: 15\n  totally_made_up_key: 1\nnot_a_real_section:\n  x: 1\n",
        encoding="utf-8",
    )

    config = load_agent_config(path)

    assert config.leader.max_steps == 15


def test_non_mapping_top_level_raises_a_clear_error(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    try:
        load_agent_config(path)
    except ValueError as exc:
        assert "YAML mapping" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a non-mapping YAML document")


def test_agent_config_path_env_var_is_used_when_no_explicit_path_given(monkeypatch, tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("leader:\n  max_steps: 42\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_CONFIG_PATH", str(path))

    config = load_agent_config()

    assert config.leader.max_steps == 42


def test_repo_shipped_config_file_matches_the_dataclass_defaults():
    """config/agent.yaml documents the shipped defaults in its comments --
    keep them honest: loading it should produce exactly AgentConfig()."""

    from app.config import DEFAULT_CONFIG_PATH

    assert DEFAULT_CONFIG_PATH.exists()
    assert load_agent_config(DEFAULT_CONFIG_PATH) == AgentConfig()
