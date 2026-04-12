from __future__ import annotations

import tomllib

from pcstubgen.ir_modules import IRArgument, IRClass, IRFunction, IRMethod, IRModule, QualifiedName
from pcstubgen.stub_output import StubRenderer, TomlWriter
from pcstubgen.types import RawType
from tests._c_extension_test_support import _signature


def test_toml_writer_writes_single_module_function_record(tmp_path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="foo",
                runtime_handle=object(),
                signatures=[
                    _signature(
                        args=[IRArgument(name="value", type=RawType("int"))],
                        return_type=RawType("bool"),
                    )
                ],
                c_inferred_source_comment="static int foo_impl(int value) { return value; }",
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
                "source_comment": "static int foo_impl(int value) { return value; }",
            }
        ]
    }


def test_toml_writer_splits_overloads_into_multiple_records(tmp_path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="foo",
                runtime_handle=object(),
                signatures=[
                    _signature(
                        args=[IRArgument(name="value", type=RawType("int"))],
                        return_type=RawType("int"),
                    ),
                    _signature(
                        args=[IRArgument(name="value", type=RawType("str"))],
                        return_type=RawType("str"),
                    ),
                ],
                c_inferred_source_comment="static PyObject* foo_impl(PyObject* self, PyObject* args);",
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
                "source_comment": "static PyObject* foo_impl(PyObject* self, PyObject* args);",
            },
            {
                "module_name": "pkg.mod",
                "function_name": "foo",
                "signature": "def foo(value: str) -> str:",
                "source_comment": "static PyObject* foo_impl(PyObject* self, PyObject* args);",
            },
        ]
    }


def test_toml_writer_renders_multi_argument_signature_on_single_line(tmp_path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="foo",
                runtime_handle=object(),
                signatures=[
                    _signature(
                        args=[
                            IRArgument(name="value", type=RawType("int")),
                            IRArgument(name="flag", type=RawType("bool")),
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
                "signature": "def foo(value: int, flag: bool) -> str:",
            }
        ]
    }


def test_toml_writer_keeps_unknown_function_record_without_signature(tmp_path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="missing",
                runtime_handle=object(),
                signatures=[],
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
                "function_name": "missing",
            }
        ]
    }


def test_toml_writer_exports_nested_class_methods_with_full_class_name(tmp_path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[
            IRClass(
                name="Outer",
                methods=[
                    IRMethod(
                        function=IRFunction(
                            name="build",
                            runtime_handle=object(),
                            signatures=[
                                _signature(
                                    args=[IRArgument(name="value", type=RawType("int"))],
                                    return_type=RawType("int"),
                                )
                            ],
                        ),
                        decorator=None,
                    )
                ],
                classes=[
                    IRClass(
                        name="Inner",
                        methods=[
                            IRMethod(
                                function=IRFunction(
                                    name="build_inner",
                                    runtime_handle=object(),
                                    signatures=[_signature(return_type=RawType("int"))],
                                ),
                                decorator=None,
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
                "signature": "def build(value: int) -> int:",
            },
            {
                "module_name": "pkg.mod",
                "class_name": "Outer.Inner",
                "function_name": "build_inner",
                "signature": "def build_inner() -> int:",
            },
        ]
    }
