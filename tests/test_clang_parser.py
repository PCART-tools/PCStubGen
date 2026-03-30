from __future__ import annotations

from tests._c_signature_test_support import *


def test_c_signature_engine_returns_translation_unit_when_error_present(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, c_std="c11")
    source = tmp_path / "module.c"
    translation_unit = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Warning,
                message="warning detail",
                file_name=str(source),
                line=3,
                column=1,
            ),
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Error,
                message="error detail",
                file_name=str(source),
                line=7,
                column=9,
            ),
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Fatal,
                message="fatal detail",
                file_name=str(source),
                line=11,
                column=4,
            ),
        ]
    )

    result = translation_unit_module.parse(
        index=_FakeIndex(translation_unit),
        file_path=source,
        source_root=config["source_root"],
        include=config["include"],
        include_directory=config["include_directory"],
        c_std=config["c_std"],
        cpp_std=config["cpp_std"],
    )

    assert result is translation_unit


def test_c_signature_engine_auto_adds_include_dir_for_nested_header_literal(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, c_std="c11")
    source = tmp_path / "src" / "module.c"
    header_path = tmp_path / "numpy_core" / "include" / "numpy" / "npy_common.h"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text("/* header */", encoding="utf-8")

    first = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'numpy/npy_common.h' file not found",
                file_name=str(source),
                line=1,
                column=1,
            )
        ]
    )
    second = _FakeTranslationUnit(diagnostics=[])
    index = _SequentialIndex([first, second])

    result = translation_unit_module.parse(
        index=index,
        file_path=source,
        source_root=config["source_root"],
        include=config["include"],
        include_directory=config["include_directory"],
        c_std=config["c_std"],
        cpp_std=config["cpp_std"],
    )

    assert result is second
    expected_include_root = header_path.parents[1]
    assert expected_include_root in config["include_directory"]
    assert header_path.parent not in config["include_directory"]
    assert len(index.calls) == 2
    assert _has_std_arg(index.calls[0][1], "c11")
    assert _has_std_arg(index.calls[1][1], "c11")


def test_c_signature_engine_retries_until_missing_includes_converge(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, c_std="c11")
    source = tmp_path / "pkg" / "src" / "module.c"

    include_one = tmp_path / "vendor1" / "include"
    include_two = tmp_path / "vendor2" / "include"
    (include_one / "numpy").mkdir(parents=True, exist_ok=True)
    (include_two / "pkg").mkdir(parents=True, exist_ok=True)
    (include_one / "numpy" / "npy_common.h").write_text("/* one */", encoding="utf-8")
    (include_two / "pkg" / "extra.h").write_text("/* two */", encoding="utf-8")

    first = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'numpy/npy_common.h' file not found",
                file_name=str(source),
                line=2,
                column=7,
            )
        ]
    )
    second = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'pkg/extra.h' file not found",
                file_name=str(source),
                line=3,
                column=5,
            )
        ]
    )
    third = _FakeTranslationUnit(diagnostics=[])
    index = _SequentialIndex([first, second, third])

    result = translation_unit_module.parse(
        index=index,
        file_path=source,
        source_root=config["source_root"],
        include=config["include"],
        include_directory=config["include_directory"],
        c_std=config["c_std"],
        cpp_std=config["cpp_std"],
    )

    assert result is third
    assert include_one in config["include_directory"]
    assert include_two in config["include_directory"]
    assert len(index.calls) == 3
    assert _has_std_arg(index.calls[0][1], "c11")
    assert _has_std_arg(index.calls[1][1], "c11")
    assert _has_std_arg(index.calls[2][1], "c11")
    assert not _has_include_directory_arg(index.calls[0][1], include_one)
    assert _has_include_directory_arg(index.calls[1][1], include_one)
    assert not _has_include_directory_arg(index.calls[1][1], include_two)
    assert _has_include_directory_arg(index.calls[2][1], include_one)
    assert _has_include_directory_arg(index.calls[2][1], include_two)


def test_c_signature_engine_does_not_retry_when_missing_header_is_unresolved(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, c_std="c11")
    source = tmp_path / "src" / "module.c"
    initial_include_dirs = list(config["include_directory"])

    unrelated_header = tmp_path / "include" / "numpy" / "arrayobject.h"
    unrelated_header.parent.mkdir(parents=True, exist_ok=True)
    unrelated_header.write_text("/* unrelated */", encoding="utf-8")

    unresolved = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'numpy/npy_common.h' file not found",
                file_name=str(source),
                line=6,
                column=3,
            )
        ]
    )
    index = _SequentialIndex([unresolved])

    result = translation_unit_module.parse(
        index=index,
        file_path=source,
        source_root=config["source_root"],
        include=config["include"],
        include_directory=config["include_directory"],
        c_std=config["c_std"],
        cpp_std=config["cpp_std"],
    )

    assert result is unresolved
    assert config["include_directory"] == initial_include_dirs
    assert len(index.calls) == 1
