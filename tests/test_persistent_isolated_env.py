from __future__ import annotations

import os
from pathlib import Path

import pcstubgen._persistent_isolated_env as persistent_isolated_env_module
from pcstubgen._persistent_isolated_env import PersistentIsolatedEnv


class _FakeEnvBackend:
    def __init__(self, created_paths: list[str]) -> None:
        self.created_paths = created_paths
        self.display_name = "fake-backend"

    def create(self, path: str) -> None:
        self.created_paths.append(path)
        Path(path).mkdir(parents=True, exist_ok=True)


def test_get_build_env_path_returns_new_directory_name(tmp_path: Path) -> None:
    build_env_path = PersistentIsolatedEnv.get_build_env_path(tmp_path)

    assert build_env_path == tmp_path / ".pcstubgen-build-env"


def test_persistent_isolated_env_recreates_existing_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    build_env_path = PersistentIsolatedEnv.get_build_env_path(tmp_path)
    gitignore_path = build_env_path / ".gitignore"
    build_env_path.mkdir()
    stale_file = build_env_path / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")
    created_paths: list[str] = []
    logged_messages: list[tuple[str, tuple[str, ...] | None]] = []

    monkeypatch.setattr(
        persistent_isolated_env_module.build_env,
        "_PipBackend",
        lambda: _FakeEnvBackend(created_paths),
    )
    monkeypatch.setattr(
        persistent_isolated_env_module.build_env._ctx,
        "log",
        lambda message, kind=None: logged_messages.append((message, kind)),
    )

    with PersistentIsolatedEnv(tmp_path) as env:
        assert env.path == os.path.realpath(build_env_path.resolve())

    assert created_paths == [os.path.realpath(build_env_path.resolve())]
    assert gitignore_path.read_text(encoding="utf-8") == "*\n"
    assert not stale_file.exists()
    assert logged_messages == [
        ("Creating isolated environment: fake-backend...", ("step",))
    ]
