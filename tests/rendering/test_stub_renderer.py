from __future__ import annotations

import typing

from pcstubgen.type_models import RawType, Type
from pcstubgen.models import Argument, ArgumentKind, Class, Function, Module, Signature, QualifiedName
from pcstubgen.stub_output import StubRenderer


def _signature(
    *,
    args: list[Argument] | None = None,
    return_type: Type | None = None,
) -> Signature:
    """构造测试用签名。"""
    return Signature(
        args=list(args or ()),
        return_type=return_type,
    )


def _function(
    name: str,
    *,
    signatures: list[Signature] | None = None,
    doc: str | None = None,
    comment: str | None = None,
) -> Function:
    return Function(
        name=name,
        runtime_handle=object(),
        signatures=list(signatures or ()),
        doc=doc,
        comment=comment,
    )


def _unknown_function(name: str, *, doc: str | None = None) -> Function:
    """构造签名未知的测试函数。"""
    return _function(name=name, doc=doc)


def test_renderer_preserves_raw_optional_annotation_text() -> None:
    func = _function(
        name="foo",
        signatures=[
            _signature(
                args=[Argument(name="value", type=RawType("typing.Optional[int]", imports=("typing",)))],
                return_type=RawType("typing.Optional[int]", imports=("typing",)),
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).render_function(func)

    assert lines == [
        "def foo(value: typing.Optional[int]) -> typing.Optional[int]:",
        "    ...",
    ]


def test_renderer_prints_placeholder_signature_for_unknown_function() -> None:
    func = _function(name="foo")

    lines = StubRenderer(include_docstrings=False).render_function(func)

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
        signatures=[_signature(args=[Argument(name="value", type=RawType("int"))])],
        doc="original docs",
    )

    lines = StubRenderer(include_docstrings=True).render_function(func)

    assert lines == [
        "def foo(value: int):",
        '    """',
        "    original docs",
        '    """',
    ]


def test_renderer_renders_unknown_default_value_as_ellipsis() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[Argument(name="value", type=RawType("int"), default_value="...")])],
    )

    lines = StubRenderer(include_docstrings=False).render_function(func)

    assert lines == [
        "def foo(value: int = ...):",
        "    ...",
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
            _signature(args=[Argument(name="value", type=RawType("int"))], return_type=RawType("str")),
            _signature(args=[Argument(name="value", type=RawType("str"))], return_type=RawType("int")),
        ],
        doc=doc,
    )

    lines = StubRenderer(include_docstrings=True).render_function(func)

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
        signatures=[_signature(args=[Argument(name="value", type=RawType("int"))], return_type=RawType("bool"))],
        doc="foo(value: str) -> str\n\nparsed from docstring",
    )

    lines = StubRenderer(include_docstrings=True).render_function(func)

    assert lines[0] == "def foo(value: int) -> bool:"
    assert "    foo(value: str) -> str" in lines
    assert "    parsed from docstring" in lines


def test_renderer_prints_comment_after_function() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[Argument(name="value", type=RawType("int"))])],
        comment="src/foo_impl.c:12:3\nstatic int foo_impl(int value) {\n    return value;\n}",
    )

    lines = StubRenderer(include_docstrings=False).render_function(func)

    assert lines[:2] == [
        "def foo(value: int):",
        "    ...",
    ]
    assert lines[2:] == [
        "#   src/foo_impl.c:12:3",
        "#   static int foo_impl(int value) {",
        "#       return value;",
        "#   }",
    ]


def test_renderer_prints_comment_once_after_overloads() -> None:
    func = _function(
        name="foo",
        signatures=[
            _signature(
                args=[Argument(name="value", type=RawType("int")), Argument(name="flag", type=RawType("bool"))],
                return_type=RawType("int"),
            ),
            _signature(
                args=[Argument(name="value", type=RawType("str")), Argument(name="flag", type=RawType("bool"))],
                return_type=RawType("str"),
            ),
        ],
        comment="src/foo_impl.c:21:7\nstatic PyObject* foo_impl(PyObject* self, PyObject* args) {\n    return self;\n}",
    )

    lines = StubRenderer(include_docstrings=False).render_function(func)

    assert lines.count("@typing.overload") == 2
    assert lines[-4:] == [
        "#   src/foo_impl.c:21:7",
        "#   static PyObject* foo_impl(PyObject* self, PyObject* args) {",
        "#       return self;",
        "#   }",
    ]


def test_renderer_adds_typing_import_for_overloads() -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="foo",
                runtime_handle=object(),
                signatures=[
                    _signature(args=[Argument(name="x", type=RawType("int"))], return_type=RawType("int")),
                    _signature(
                        args=[Argument(name="x", type=RawType("typing.Optional[int]", imports=("typing",)))],
                        return_type=RawType("typing.Optional[int]", imports=("typing",)),
                    ),
                ],
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).render_module(module)

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
    method = Function(
        name="build",
        runtime_handle=object(),
        signatures=[
            _signature(args=[Argument(name="x", type=RawType("int"))], return_type=RawType("int")),
            _signature(args=[Argument(name="x", type=RawType("str"))], return_type=RawType("str")),
        ],
        decorator="classmethod",
    )

    lines = StubRenderer(include_docstrings=False).render_method(method)

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
