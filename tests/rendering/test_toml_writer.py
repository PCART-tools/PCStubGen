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
                handle=object(),
                signatures=[
                    _signature(
                        args=[Argument(name="value", type=RawType("int"))],
                        return_type=RawType("bool"),
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
                handle=object(),
                signatures=[
                    _signature(
                        args=[Argument(name="value", type=RawType("int"))],
                        return_type=RawType("int"),
                    ),
                    _signature(
                        args=[Argument(name="value", type=RawType("str"))],
                        return_type=RawType("str"),
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


def test_toml_writer_renders_multi_argument_signature_on_multiple_lines(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="foo",
                handle=object(),
                signatures=[
                    _signature(
                        args=[
                            Argument(name="value", type=RawType("int")),
                            Argument(name="flag", type=RawType("bool")),
                        ],
                        return_type=RawType("str"),
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

    assert tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8")) == {
        "entries": [
            {
                "module_name": "pkg.mod",
                "function_name": "foo",
                "signature": "def foo(\n    value: int,\n    flag: bool,\n) -> str:",
            }
        ]
    }


def test_toml_writer_rejects_function_without_exportable_signature(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="missing",
                handle=object(),
                signatures=[],
            )
        ],
    )

    with pytest.raises(RuntimeError, match="pkg.mod.missing 缺少可导出签名"):
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
                        handle=object(),
                        signatures=[
                            _signature(
                                args=[Argument(name="self"), Argument(name="value", type=RawType("int"))],
                                return_type=RawType("int"),
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
                                handle=object(),
                                signatures=[_signature(args=[Argument(name="self")], return_type=RawType("int"))],
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

    assert tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8")) == {
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


def test_toml_writer_inserts_cls_and_skips_staticmethod_receiver(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[
            Class(
                name="Factory",
                methods=[
                    Function(
                        name="build",
                        handle=object(),
                        signatures=[
                            _signature(
                                args=[Argument(name="cls"), Argument(name="value", type=RawType("int"))],
                                return_type=RawType("int"),
                            )
                        ],
                        decorator="classmethod",
                    ),
                    Function(
                        name="make",
                        handle=object(),
                        signatures=[
                            _signature(
                                args=[Argument(name="value", type=RawType("str"))],
                                return_type=RawType("str"),
                            )
                        ],
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

    assert tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8")) == {
        "entries": [
            {
                "module_name": "pkg.mod",
                "class_name": "Factory",
                "function_name": "build",
                "signature": "def build(\n    cls,\n    value: int,\n) -> int:",
            },
            {
                "module_name": "pkg.mod",
                "class_name": "Factory",
                "function_name": "make",
                "signature": "def make(value: str) -> str:",
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
                        handle=object(),
                        signatures=[_signature(args=[Argument(name="cls"), Argument(name="value")])],
                        decorator="classmethod",
                    ),
                    Function(
                        name="alpha",
                        handle=object(),
                        signatures=[_signature(args=[Argument(name="self"), Argument(name="value")])],
                    ),
                    Function(
                        name="middle",
                        handle=object(),
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
