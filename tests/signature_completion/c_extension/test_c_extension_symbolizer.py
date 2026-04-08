from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from pcstubgen.signature_completion.c_extension import symbolizer as symbolizer_module


def test_require_llvm_symbolizer_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(symbolizer_module, "find_llvm_symbolizer", lambda: None)

    with pytest.raises(RuntimeError, match="PATH 中未找到"):
        symbolizer_module.require_llvm_symbolizer()


def test_resolve_symbolized_address_parses_json_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "sample.so"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        symbolizer_module,
        "_resolve_binary_address",
        lambda address: (binary_path, address - 0x1000),
    )
    monkeypatch.setattr(symbolizer_module, "require_llvm_symbolizer", lambda: "/usr/bin/llvm-symbolizer")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                '[{"Address":"0x234","ModuleName":"sample.so","Symbol":[{"FileName":"%s",'
                '"FunctionName":"foo_impl","Line":17,"StartFileName":"%s","StartLine":11}]}]'
            )
            % (tmp_path / "foo_body.c", tmp_path / "foo_impl.c"),
            stderr="",
        )

    monkeypatch.setattr(symbolizer_module.subprocess, "run", fake_run)

    result = symbolizer_module.resolve_symbolized_address(0x1234)

    assert captured["cmd"] == [
        "/usr/bin/llvm-symbolizer",
        "--output-style=JSON",
        "--relative-address",
        f"--obj={binary_path}",
        "--no-demangle",
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
        symbolizer_module,
        "_resolve_binary_address",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(symbolizer_module, "require_llvm_symbolizer", lambda: "/usr/bin/llvm-symbolizer")
    monkeypatch.setattr(
        symbolizer_module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="boom",
        ),
    )

    with pytest.raises(RuntimeError, match="执行失败"):
        symbolizer_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        symbolizer_module,
        "_resolve_binary_address",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(symbolizer_module, "require_llvm_symbolizer", lambda: "/usr/bin/llvm-symbolizer")
    monkeypatch.setattr(
        symbolizer_module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="{not-json}",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="无效JSON"):
        symbolizer_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_error_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        symbolizer_module,
        "_resolve_binary_address",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(symbolizer_module, "require_llvm_symbolizer", lambda: "/usr/bin/llvm-symbolizer")
    monkeypatch.setattr(
        symbolizer_module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"Error":{"Message":"No such file or directory"}}',
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="No such file or directory"):
        symbolizer_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        symbolizer_module,
        "_resolve_binary_address",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(symbolizer_module, "require_llvm_symbolizer", lambda: "/usr/bin/llvm-symbolizer")
    monkeypatch.setattr(
        symbolizer_module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="[]",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="非单地址结果"):
        symbolizer_module.resolve_symbolized_address(0x1234)


def test_resolve_symbolized_address_rejects_missing_symbol_information(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        symbolizer_module,
        "_resolve_binary_address",
        lambda address: (tmp_path / "sample.so", address),
    )
    monkeypatch.setattr(symbolizer_module, "require_llvm_symbolizer", lambda: "/usr/bin/llvm-symbolizer")
    monkeypatch.setattr(
        symbolizer_module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='[{"Address":"0x1234","ModuleName":"sample.so","Symbol":[]}]',
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="未返回任何符号信息"):
        symbolizer_module.resolve_symbolized_address(0x1234)
