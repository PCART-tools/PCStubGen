from __future__ import annotations

import typing

from pcstubgen.types import RawType, Type
from pcstubgen.ir_modules import IRArgument, IRArgumentKind, IRClass, IRFunction, IRMethod, IRModule, IRModuleType, IRSignature, QualifiedName
from pcstubgen.stub_output import StubRenderer


def _signature(
    *,
    args: list[IRArgument] | None = None,
    return_type: Type | None = None,
) -> IRSignature:
    """构造测试用签名。"""
    return IRSignature(
        args=list(args or ()),
        return_type=return_type,
    )


def _function(
    name: str,
    *,
    signatures: list[IRSignature] | None = None,
    doc: str | None = None,
    c_inferred_source_comment: str | None = None,
) -> IRFunction:
    return IRFunction(
        name=name,
        runtime_handle=object(),
        signatures=list(signatures or ()),
        doc=doc,
        c_inferred_source_comment=c_inferred_source_comment,
    )


def _unknown_function(name: str, *, doc: str | None = None) -> IRFunction:
    """构造签名未知的测试函数。"""
    return _function(name=name, doc=doc)


def test_renderer_preserves_raw_optional_annotation_text() -> None:
    func = _function(
        name="foo",
        signatures=[
            _signature(
                args=[IRArgument(name="value", type=RawType("typing.Optional[int]", imports=("typing",)))],
                return_type=RawType("typing.Optional[int]", imports=("typing",)),
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(value: typing.Optional[int]) -> typing.Optional[int]:",
        "    ...",
    ]


def test_renderer_prints_placeholder_signature_for_unknown_function() -> None:
    func = _function(name="foo")

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(",
        "    *args,",
        "    **kwargs,",
        "):",
        "    ...",
    ]


def test_renderer_prints_function_doc_for_single_signature() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[IRArgument(name="value", type=RawType("int"))])],
        doc="original docs",
    )

    lines = StubRenderer(include_docstrings=True).print_function(func)

    assert lines == [
        "def foo(value: int):",
        '    """',
        "    original docs",
        '    """',
    ]


def test_renderer_prints_function_doc_for_placeholder_signature() -> None:
    func = _function(name="foo", doc="original docs")

    lines = StubRenderer(include_docstrings=True).print_function(func)

    assert lines == [
        "def foo(",
        "    *args,",
        "    **kwargs,",
        "):",
        '    """',
        "    original docs",
        '    """',
    ]


def test_renderer_repeats_original_function_doc_for_each_overload() -> None:
    doc = (
        "foo(*args, **kwargs)\n"
        "Overloaded function.\n"
        "1. foo(value: int) -> str\n"
        "\n"
        "first overload\n"
        "2. foo(value: str) -> int\n"
        "\n"
        "second overload"
    )
    func = _function(
        name="foo",
        signatures=[
            _signature(args=[IRArgument(name="value", type=RawType("int"))], return_type=RawType("str")),
            _signature(args=[IRArgument(name="value", type=RawType("str"))], return_type=RawType("int")),
        ],
        doc=doc,
    )

    lines = StubRenderer(include_docstrings=True).print_function(func)

    assert lines.count("@typing.overload") == 2
    assert "def foo(value: int) -> str:" in lines
    assert "def foo(value: str) -> int:" in lines
    assert lines.count('    """') == 4
    assert lines.count("    Overloaded function.") == 2
    assert lines.count("    first overload") == 2
    assert lines.count("    second overload") == 2


def test_renderer_preserves_original_doc_when_signature_conflicts_with_doc_text() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[IRArgument(name="value", type=RawType("int"))], return_type=RawType("bool"))],
        doc="foo(value: str) -> str\n\nparsed from docstring",
    )

    lines = StubRenderer(include_docstrings=True).print_function(func)

    assert lines[0] == "def foo(value: int) -> bool:"
    assert "    foo(value: str) -> str" in lines
    assert "    parsed from docstring" in lines


def test_renderer_prints_c_inferred_source_comment_after_function() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[IRArgument(name="value", type=RawType("int"))])],
        c_inferred_source_comment="static int foo_impl(int value) {\n    return value;\n}",
    )

    lines = StubRenderer(
        include_docstrings=False,
        include_c_inferred_source_comment=True,
    ).print_function(func)

    assert lines[:2] == [
        "def foo(value: int):",
        "    ...",
    ]
    assert lines[2] == "#   C inferred source for foo:"
    assert lines[3:] == [
        "#   static int foo_impl(int value) {",
        "#       return value;",
        "#   }",
    ]


def test_renderer_prints_c_inferred_source_comment_once_after_overloads() -> None:
    func = _function(
        name="foo",
        signatures=[
            _signature(
                args=[IRArgument(name="value", type=RawType("int")), IRArgument(name="flag", type=RawType("bool"))],
                return_type=RawType("int"),
            ),
            _signature(
                args=[IRArgument(name="value", type=RawType("str")), IRArgument(name="flag", type=RawType("bool"))],
                return_type=RawType("str"),
            ),
        ],
        c_inferred_source_comment="static PyObject* foo_impl(PyObject* self, PyObject* args) {\n    return self;\n}",
    )

    lines = StubRenderer(
        include_docstrings=False,
        include_c_inferred_source_comment=True,
    ).print_function(func)

    assert lines.count("@typing.overload") == 2
    assert lines.count("#   C inferred source for foo:") == 1
    assert lines[-4:] == [
        "#   C inferred source for foo:",
        "#   static PyObject* foo_impl(PyObject* self, PyObject* args) {",
        "#       return self;",
        "#   }",
    ]


def test_renderer_skips_c_inferred_source_comment_when_disabled() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[IRArgument(name="value", type=RawType("int"))])],
        c_inferred_source_comment="static int foo_impl(int value) { return value; }",
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(value: int):",
        "    ...",
    ]


def test_renderer_adds_typing_import_for_overloads() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="foo",
                runtime_handle=object(),
                signatures=[
                    _signature(args=[IRArgument(name="x", type=RawType("int"))], return_type=RawType("int")),
                    _signature(
                        args=[IRArgument(name="x", type=RawType("typing.Optional[int]", imports=("typing",)))],
                        return_type=RawType("typing.Optional[int]", imports=("typing",)),
                    ),
                ],
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).print_module(module)

    assert lines == [
        "import typing",
        "@typing.overload",
        "def foo(x: int) -> int:",
        "    ...",
        "@typing.overload",
        "def foo(x: typing.Optional[int]) -> typing.Optional[int]:",
        "    ...",
    ]


def test_renderer_repeats_method_decorator_for_each_overload() -> None:
    method = IRMethod(
        function=IRFunction(
            name="build",
            runtime_handle=object(),
            signatures=[
                _signature(args=[IRArgument(name="x", type=RawType("int"))], return_type=RawType("int")),
                _signature(args=[IRArgument(name="x", type=RawType("str"))], return_type=RawType("str")),
            ],
        ),
        decorator="classmethod",
    )

    lines = StubRenderer(include_docstrings=False).print_method(method)

    assert lines == [
        "@classmethod",
        "@typing.overload",
        "def build(x: int) -> int:",
        "    ...",
        "@classmethod",
        "@typing.overload",
        "def build(x: str) -> str:",
        "    ...",
    ]
