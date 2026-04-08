from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from pcstubgen.signature_completion.c_extension import address_resolver as resolver_module


@pytest.fixture(autouse=True)
def stub_llvm_symbolizer_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resolver_module.llvm_symbolizer,
        "require_llvm_symbolizer",
        lambda: "/usr/bin/llvm-symbolizer",
    )


def test_resolve_symbolized_address_parses_json_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "sample.so"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (binary_path, address - 0x1000),
    )

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                '[{"Address":"0x234","ModuleName":"sample.so","Symbol":[{"FileName":"%s",'
                '"FunctionName":"foo_impl","Line":17,"Column":3,"StartAddress":"0x200",'
                '"StartFileName":"%s","StartLine":11,"Discriminator":0}]}]'
            )
            % (tmp_path / "foo_body.c", tmp_path / "foo_impl.c"),
            stderr="",
        )

    monkeypatch.setattr(resolver_module.llvm_symbolizer.subprocess, "run", fake_run)

    result = resolver_module.resolve_symbolized_address(0x1234)

    assert captured["cmd"] == [
        "/usr/bin/llvm-symbolizer",
        "--output-style=JSON",
        "--relative-address",
        f"--obj={binary_path}",
        "0x234",
    ]
    assert result.binary_path == binary_path
    assert result.relative_address == 0x234
    assert result.function_name == "foo_impl"
    assert result.resolved_path == (tmp_path / "foo_body.c").resolve()
    assert result.resolved_line == 17
    assert result.function_start_path == (tmp_path / "foo_impl.c").resolve()
    assert result.function_start_line == 11


def test_resolve_symbolized_address_rejects_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(
        resolver_module.llvm_symbolizer.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="boom",
        ),
    )

    with pytest.raises(RuntimeError, match="执行失败"):
        resolver_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(
        resolver_module.llvm_symbolizer.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="[",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="非预期JSON结果"):
        resolver_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_error_result_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(
        resolver_module.llvm_symbolizer.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                '[{"Address":"0x1234","ModuleName":"sample.so",'
                '"Error":{"Message":"No such file or directory","Code":2,"Type":"IO"}}]'
            ),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="No such file or directory"):
        resolver_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(
        resolver_module.llvm_symbolizer.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="[]",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="非单地址结果"):
        resolver_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_missing_symbol_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(
        resolver_module.llvm_symbolizer.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='[{"Address":"0x1234","ModuleName":"sample.so"}]',
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="未返回任何符号信息"):
        resolver_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_missing_symbol_information(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(
        resolver_module.llvm_symbolizer.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='[{"Address":"0x1234","ModuleName":"sample.so","Symbol":[]}]',
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="未返回任何符号信息"):
        resolver_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_invalid_first_symbol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(
        resolver_module.llvm_symbolizer.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='[{"Address":"0x1234","ModuleName":"sample.so","Symbol":["bad"]}]',
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="非预期JSON结果"):
        resolver_module.resolve_symbolized_address(0x1234)
