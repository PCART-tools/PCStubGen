from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core import cli


def test_parse_args_rejects_removed_enable_c_signature_inference_flag() -> None:
    with pytest.raises(SystemExit) as ex:
        cli.parse_args(["math", "--enable-c-signature-inference"])

    assert ex.value.code == 2


def test_build_options_treats_empty_source_root_as_none() -> None:
    args = cli.parse_args(["math", "--source-root", ""])
    options = cli._build_options(args)

    assert options.source_root is None


def test_build_options_treats_whitespace_source_root_as_none() -> None:
    args = cli.parse_args(["math", "--source-root", "   "])
    options = cli._build_options(args)

    assert options.source_root is None


def test_build_options_keeps_non_empty_source_root_path() -> None:
    args = cli.parse_args(["math", "--source-root", "C:/tmp/src"])
    options = cli._build_options(args)

    assert options.source_root == Path("C:/tmp/src")


def test_parse_args_accepts_clang_include_directory() -> None:
    args = cli.parse_args(
        [
            "math",
            "--clang-include-directory",
            "C:/IncludeA",
            "--clang-include-directory=C:/IncludeB",
        ]
    )
    options = cli._build_options(args)

    assert options.clang_include_directory == ["C:/IncludeA", "C:/IncludeB"]


def test_parse_args_accepts_clang_include() -> None:
    args = cli.parse_args(
        [
            "math",
            "--clang-include",
            "Python.h",
            "--clang-include=numpy/arrayobject.h",
        ]
    )
    options = cli._build_options(args)

    assert options.clang_include == ["Python.h", "numpy/arrayobject.h"]
