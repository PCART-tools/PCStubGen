from __future__ import annotations

import json

from pcstubgen.ir_modules import IRArgument, IRClass, IRFunction, IRMethod, IRModule, QualifiedName
from pcstubgen.stub_output import JsonWriter, StubRenderer
from pcstubgen.types import RawType
from tests._c_extension_test_support import _signature


def test_json_writer_writes_single_module_function_record(tmp_path) -> None:
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

    JsonWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    output_path = tmp_path / "mod.json"
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {
            "module_name": "pkg.mod",
            "class_name": None,
            "function_name": "foo",
            "signature": "def foo(value: int) -> bool:",
            "source_comment": "static int foo_impl(int value) { return value; }",
        }
    ]


def test_json_writer_splits_overloads_into_multiple_records(tmp_path) -> None:
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

    JsonWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    assert json.loads((tmp_path / "mod.json").read_text(encoding="utf-8")) == [
        {
            "module_name": "pkg.mod",
            "class_name": None,
            "function_name": "foo",
            "signature": "def foo(value: int) -> int:",
            "source_comment": "static PyObject* foo_impl(PyObject* self, PyObject* args);",
        },
        {
            "module_name": "pkg.mod",
            "class_name": None,
            "function_name": "foo",
            "signature": "def foo(value: str) -> str:",
            "source_comment": "static PyObject* foo_impl(PyObject* self, PyObject* args);",
        },
    ]


def test_json_writer_renders_multi_argument_signature_on_single_line(tmp_path) -> None:
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

    JsonWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    assert json.loads((tmp_path / "mod.json").read_text(encoding="utf-8")) == [
        {
            "module_name": "pkg.mod",
            "class_name": None,
            "function_name": "foo",
            "signature": "def foo(value: int, flag: bool) -> str:",
            "source_comment": None,
        }
    ]


def test_json_writer_keeps_placeholder_record_for_unknown_function(tmp_path) -> None:
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

    JsonWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    assert json.loads((tmp_path / "mod.json").read_text(encoding="utf-8")) == [
        {
            "module_name": "pkg.mod",
            "class_name": None,
            "function_name": "missing",
            "signature": None,
            "source_comment": None,
        }
    ]


def test_json_writer_exports_direct_class_methods_only(tmp_path) -> None:
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
                                    name="skip_me",
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

    JsonWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    assert json.loads((tmp_path / "mod.json").read_text(encoding="utf-8")) == [
        {
            "module_name": "pkg.mod",
            "class_name": "Outer",
            "function_name": "build",
            "signature": "def build(value: int) -> int:",
            "source_comment": None,
        }
    ]
