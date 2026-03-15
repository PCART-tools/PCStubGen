from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tool import libclang_ast


def test_parse_args_accepts_clang_include_and_include_directory() -> None:
    args = libclang_ast.parse_args(
        [
            "sample.c",
            "--clang-include",
            "Python.h",
            "--clang-include=numpy/arrayobject.h",
            "--clang-include-directory",
            "C:/IncludeA",
            "--clang-include-directory=C:/IncludeB",
        ]
    )

    assert args.clang_include == ["Python.h", "numpy/arrayobject.h"]
    assert args.clang_include_directory == ["C:/IncludeA", "C:/IncludeB"]


def test_build_parse_args_places_include_before_include_directory() -> None:
    parse_args = libclang_ast._build_parse_args(
        source_path=Path("sample.c"),
        clang_args=[],
        include_headers=["Python.h", "numpy/arrayobject.h"],
        include_paths=["C:/IncludeA"],
        clang_c_std="c11",
        clang_cpp_std="c++17",
    )

    assert parse_args == [
        "--std",
        "c11",
        "--include",
        "Python.h",
        "--include",
        "numpy/arrayobject.h",
        "--include-directory",
        "C:/IncludeA",
    ]


def test_normalize_include_headers_rejects_option_like_values() -> None:
    with pytest.raises(ValueError, match="option-like"):
        libclang_ast._normalize_include_headers(["-Winvalid"])
