from __future__ import annotations

import tomllib

import pytest

from pcstubgen.models import Argument, Class, Function, Module, QualifiedName
from pcstubgen.stub_output import StubRenderer, TomlWriter
from pcstubgen.type_models import RawType
from tests._c_extension_test_support import _signature


def test_toml_writer_writes_single_module_function_record(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="foo",
                signatures=[
                    _signature(
                        args=[Argument(name="value", type=RawType.int_)],
                        return_type=RawType.bool_,
                    )
                ],
                comment="src/foo_impl.c:12:3\nstatic int foo_impl(int value) { return value; }",
            )
        ],
    )

    TomlWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    output_path = tmp_path / "mod.toml"
    assert output_path.exists()
    assert tomllib.loads(output_path.read_text(encoding="utf-8")) == {
        "entries": [
            {
                "module_name": "pkg.mod",
                "function_name": "foo",
                "signature": "def foo(value: int) -> bool:",
                "comment": "src/foo_impl.c:12:3\nstatic int foo_impl(int value) { return value; }",
            }
        ]
    }


def test_toml_writer_splits_overloads_into_multiple_records(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="foo",
                signatures=[
                    _signature(
                        args=[Argument(name="value", type=RawType.int_)],
                        return_type=RawType.int_,
                    ),
                    _signature(
                        args=[Argument(name="value", type=RawType.str_)],
                        return_type=RawType.str_,
                    ),
                ],
                comment="src/foo_impl.c:21:7\nstatic PyObject* foo_impl(PyObject* self, PyObject* args);",
            )
        ],
    )

    TomlWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    assert tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8")) == {
        "entries": [
            {
                "module_name": "pkg.mod",
                "function_name": "foo",
                "signature": "def foo(value: int) -> int:",
                "comment": "src/foo_impl.c:21:7\nstatic PyObject* foo_impl(PyObject* self, PyObject* args);",
            },
            {
                "module_name": "pkg.mod",
                "function_name": "foo",
                "signature": "def foo(value: str) -> str:",
                "comment": "src/foo_impl.c:21:7\nstatic PyObject* foo_impl(PyObject* self, PyObject* args);",
            },
        ]
    }


def test_toml_writer_prefers_signature_comment_over_function_comment(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="foo",
                signatures=[
                    _signature(
                        args=[Argument(name="value", type=RawType.int_)],
                        return_type=RawType.int_,
                        comment="pybind11\n(value: int) -> int",
                    ),
                    _signature(
                        args=[Argument(name="value", type=RawType.str_)],
                        return_type=RawType.str_,
                        comment="pybind11\n(value: str) -> str",
                    ),
                ],
                comment="legacy function comment",
            )
        ],
    )

    TomlWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    assert tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8")) == {
        "entries": [
            {
                "module_name": "pkg.mod",
                "function_name": "foo",
                "signature": "def foo(value: int) -> int:",
                "comment": "pybind11\n(value: int) -> int",
            },
            {
                "module_name": "pkg.mod",
                "function_name": "foo",
                "signature": "def foo(value: str) -> str:",
                "comment": "pybind11\n(value: str) -> str",
            },
        ]
    }


def test_toml_writer_rejects_function_without_exportable_signature(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="missing",
                signatures=[],
            )
        ],
    )

    with pytest.raises(RuntimeError, match="pkg.mod.missing"):
        TomlWriter().write(
            module,
            StubRenderer(include_docstrings=False),
            to=tmp_path,
        )


def test_toml_writer_exports_nested_class_methods_with_full_class_name(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[
            Class(
                name="Outer",
                methods=[
                    Function(
                        name="build",
                        signatures=[
                            _signature(
                                args=[Argument(name="self"), Argument(name="value", type=RawType.int_)],
                                return_type=RawType.int_,
                            )
                        ],
                    )
                ],
                classes=[
                    Class(
                        name="Inner",
                        methods=[
                            Function(
                                name="build_inner",
                                signatures=[_signature(args=[Argument(name="self")], return_type=RawType.int_)],
                            )
                        ],
                    )
                ],
            )
        ],
    )

    TomlWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    entries = tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8"))["entries"]

    assert [*entries[0].keys()] == ["module_name", "class_name", "function_name", "signature"]
    assert [*entries[1].keys()] == ["module_name", "class_name", "function_name", "signature"]
    assert {"entries": entries} == {
        "entries": [
            {
                "module_name": "pkg.mod",
                "class_name": "Outer",
                "function_name": "build",
                "signature": "def build(\n    self,\n    value: int,\n) -> int:",
            },
            {
                "module_name": "pkg.mod",
                "class_name": "Outer.Inner",
                "function_name": "build_inner",
                "signature": "def build_inner(self) -> int:",
            },
        ]
    }


def test_toml_writer_sorts_class_methods_only_by_name(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[
            Class(
                name="Factory",
                methods=[
                    Function(
                        name="zeta",
                        signatures=[_signature(args=[Argument(name="cls"), Argument(name="value")])],
                        decorator="classmethod",
                    ),
                    Function(
                        name="alpha",
                        signatures=[_signature(args=[Argument(name="self"), Argument(name="value")])],
                    ),
                    Function(
                        name="middle",
                        signatures=[_signature(args=[Argument(name="value")])],
                        decorator="staticmethod",
                    ),
                ],
            )
        ],
    )

    TomlWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    entries = tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8"))["entries"]

    assert [entry["function_name"] for entry in entries] == ["alpha", "middle", "zeta"]
