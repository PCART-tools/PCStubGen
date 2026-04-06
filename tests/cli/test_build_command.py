from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from build import env as build_env

import pcstubgen.__main__ as main_module
import pcstubgen._build as build_module


RUNNER = CliRunner()


@pytest.mark.parametrize(
    ("cli_args", "expected_verbosity", "should_log"),
    [
        ([], 0, False),
        (["-v"], 1, True),
        (["-vv"], 2, True),
    ],
)
def test_build_command_sets_build_verbosity_and_restores_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cli_args: list[str],
    expected_verbosity: int,
    should_log: bool,
) -> None:
    observed_verbosity: list[int] = []
    original_logger = build_env._ctx.LOGGER.get()
    original_verbosity = build_env._ctx.verbosity

    def fake_resolve_build_context(srcdir: Path) -> build_module.BuildContext:
        return build_module.BuildContext(
            builder=None,
            build_backend="test-backend",
            runner=build_module.clang_runner,
            config_settings={},
            compile_commands_path=srcdir / "compile_commands.json",
        )

    def fake_ensure_clang_compilers_available() -> None:
        return None

    def fake_build_wheel(
        srcdir: Path,
        runner: object,
        config_settings: object,
    ) -> Path:
        _ = (runner, config_settings)
        observed_verbosity.append(build_env._ctx.verbosity)
        build_env._ctx.log("build verbose message")
        return srcdir / "dist" / "demo.whl"

    monkeypatch.setattr(build_module, "resolve_build_context", fake_resolve_build_context)
    monkeypatch.setattr(
        build_module,
        "ensure_clang_compilers_available",
        fake_ensure_clang_compilers_available,
    )
    monkeypatch.setattr(build_module, "build_wheel", fake_build_wheel)

    result = RUNNER.invoke(
        main_module.app,
        ["build", *cli_args, str(tmp_path)],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert observed_verbosity == [expected_verbosity]
    assert build_env._ctx.LOGGER.get() is original_logger
    assert build_env._ctx.verbosity == original_verbosity
    assert "构建完成" in result.output
    assert "- build-backend: test-backend" in result.output
    assert "- wheel 文件:" in result.output
    if should_log:
        assert "build verbose message" in result.output
    else:
        assert "build verbose message" not in result.output
