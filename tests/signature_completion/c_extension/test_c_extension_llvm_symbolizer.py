from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from pcstubgen.signature_completion.c_extension import llvm_symbolizer as llvm_symbolizer_module


def test_require_llvm_symbolizer_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llvm_symbolizer_module, "find_llvm_symbolizer", lambda: None)

    with pytest.raises(RuntimeError, match="PATH 中未找到"):
        llvm_symbolizer_module.require_llvm_symbolizer()


def test_run_invokes_cli_and_parses_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "sample.so"
    captured: dict[str, object] = {}

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

    monkeypatch.setattr(llvm_symbolizer_module.subprocess, "run", fake_run)

    monkeypatch.setattr(
        llvm_symbolizer_module,
        "require_llvm_symbolizer",
        lambda: "/usr/bin/llvm-symbolizer",
    )

    payload = llvm_symbolizer_module.run(binary_path, 0x234)

    assert captured["cmd"] == [
        "/usr/bin/llvm-symbolizer",
        "--output-style=JSON",
        "--relative-address",
        f"--obj={binary_path}",
        "0x234",
    ]
    assert len(payload) == 1
    entry = payload[0]
    assert entry.Address == "0x234"
    assert entry.ModuleName == "sample.so"
    assert entry.Error is None
    assert entry.Symbol is not None
    assert len(entry.Symbol) == 1
    assert entry.Symbol[0].StartAddress == "0x200"


def test_run_allows_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        llvm_symbolizer_module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                '[{"Address":"0x1234","ModuleName":"sample.so","Extra":"ignored",'
                '"Symbol":[{"FileName":"/tmp/foo.c","FunctionName":"foo_impl","Line":17,'
                '"Column":3,"StartAddress":"0x1200","StartFileName":"/tmp/foo_impl.c",'
                '"StartLine":11,"Discriminator":0,"Extra":"ignored"}]}]'
            ),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        llvm_symbolizer_module,
        "require_llvm_symbolizer",
        lambda: "/usr/bin/llvm-symbolizer",
    )

    payload = llvm_symbolizer_module.run(tmp_path / "sample.so", 0x1234)

    assert payload[0].Address == "0x1234"


@pytest.mark.parametrize(
    ("stdout", "pattern"),
    [
        ('{"Error":{"Message":"boom"}}', "非预期JSON结果"),
        ("[", "非预期JSON结果"),
        (
            '[{"Address":"0x1234","ModuleName":"sample.so","Symbol":[{"FileName":"/tmp/foo.c",'
            '"FunctionName":"foo_impl","Line":"17","Column":3,"StartAddress":"0x1200",'
            '"StartFileName":"/tmp/foo_impl.c","StartLine":11,"Discriminator":0}]}]',
            "非预期JSON结果",
        ),
        (
            '[{"ModuleName":"sample.so","Symbol":[{"FileName":"/tmp/foo.c",'
            '"FunctionName":"foo_impl","Line":17,"Column":3,"StartAddress":"0x1200",'
            '"StartFileName":"/tmp/foo_impl.c","StartLine":11,"Discriminator":0}]}]',
            "非预期JSON结果",
        ),
        (
            '[{"Address":"0x1234","ModuleName":"sample.so","Error":{"Code":"2","Message":"boom"}}]',
            "非预期JSON结果",
        ),
    ],
)
def test_run_rejects_unexpected_json_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stdout: str,
    pattern: str,
) -> None:
    monkeypatch.setattr(
        llvm_symbolizer_module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=stdout,
            stderr="",
        ),
    )

    monkeypatch.setattr(
        llvm_symbolizer_module,
        "require_llvm_symbolizer",
        lambda: "/usr/bin/llvm-symbolizer",
    )

    with pytest.raises(RuntimeError, match=pattern):
        llvm_symbolizer_module.run(tmp_path / "sample.so", 0x1234)
