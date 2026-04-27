from __future__ import annotations

import pytest

from pcstubgen.type_models import RawType, Type
from pcstubgen.models import Argument, ArgumentKind, Class, Function, Module, Signature, QualifiedName
from pcstubgen.stub_output import StubRenderer, StubWriter


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
        signatures=list(signatures or ()),
        doc=doc,
        comment=comment,
    )


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


def test_renderer_rejects_function_without_exportable_signature() -> None:
    func = _function(name="foo")

    with pytest.raises(RuntimeError, match="foo"):
        StubRenderer(include_docstrings=False).render_function(func)


def test_renderer_prints_function_doc_for_single_signature() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[Argument(name="value", type=RawType.int_)])],
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
        signatures=[_signature(args=[Argument(name="value", type=RawType.int_, default_value="...")])],
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
            _signature(args=[Argument(name="value", type=RawType.int_)], return_type=RawType.str_),
            _signature(args=[Argument(name="value", type=RawType.str_)], return_type=RawType.int_),
        ],
        doc=doc,
    )

    lines = StubRenderer(include_docstrings=True).render_function(func)

    assert lines.count("@typing.overload") == 2
    assert "def foo(value: int) -> str:" in lines
    assert "def foo(value: str) -> int:" in lines
    assert "    Overloaded function." in lines
    assert "    first overload" in lines
    assert "    second overload" in lines


def test_renderer_preserves_original_doc_when_signature_conflicts_with_doc_text() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[Argument(name="value", type=RawType.int_)], return_type=RawType.bool_)],
        doc="foo(value: str) -> str\n\nparsed from docstring",
    )

    lines = StubRenderer(include_docstrings=True).render_function(func)

    assert lines[0] == "def foo(value: int) -> bool:"
    assert "    foo(value: str) -> str" in lines
    assert "    parsed from docstring" in lines


def test_renderer_prints_comment_after_function() -> None:
    func = _function(
        name="foo",
        signatures=[_signature(args=[Argument(name="value", type=RawType.int_)])],
        comment="src/foo_impl.c:12:3\nstatic int foo_impl(int value) {\n    return value;\n}",
    )

    lines = StubRenderer(include_docstrings=False).render_function(func)

    assert lines[:2] == [
        "def foo(value: int):",
        "    ...",
    ]
    output = "\n".join(lines)
    assert "src/foo_impl.c:12:3" in output
    assert "static int foo_impl(int value)" in output
    assert "return value" in output


def test_renderer_adds_typing_import_for_overloads() -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="foo",
                signatures=[
                    _signature(args=[Argument(name="x", type=RawType.int_)], return_type=RawType.int_),
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


def test_renderer_adds_typing_import_for_self_return() -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[
            Class(
                name="Connection",
                methods=[
                    Function(
                        name="__enter__",
                        signatures=[
                            _signature(
                                args=[Argument(name="self")],
                                return_type=RawType.self_,
                            )
                        ],
                    )
                ],
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).render_module(module)

    assert lines == [
        "import typing",
        "class Connection:",
        "    def __enter__(self) -> typing.Self:",
        "        ...",
    ]


def test_renderer_repeats_method_decorator_for_each_overload() -> None:
    method = Function(
        name="build",
        signatures=[
            _signature(
                args=[Argument(name="cls"), Argument(name="x", type=RawType.int_)],
                return_type=RawType.int_,
            ),
            _signature(
                args=[Argument(name="cls"), Argument(name="x", type=RawType.str_)],
                return_type=RawType.str_,
            ),
        ],
        decorator="classmethod",
    )

    lines = StubRenderer(include_docstrings=False).render_method(method)

    assert lines == [
        "@classmethod",
        "@typing.overload",
        "def build(",
        "    cls,",
        "    x: int,",
        ") -> int:",
        "    ...",
        "@classmethod",
        "@typing.overload",
        "def build(",
        "    cls,",
        "    x: str,",
        ") -> str:",
        "    ...",
    ]


def test_renderer_uses_instance_method_signature_as_is() -> None:
    method = Function(
        name="append",
        signatures=[_signature(args=[Argument(name="self"), Argument(name="value", type=RawType.int_)])],
    )

    lines = StubRenderer(include_docstrings=False).render_method(method)

    assert lines == [
        "def append(",
        "    self,",
        "    value: int,",
        "):",
        "    ...",
    ]


def test_renderer_does_not_insert_receiver_for_staticmethod_output() -> None:
    method = Function(
        name="build",
        signatures=[_signature(args=[Argument(name="value", type=RawType.int_)])],
        decorator="staticmethod",
    )

    lines = StubRenderer(include_docstrings=False).render_method(method)

    assert lines == [
        "@staticmethod",
        "def build(value: int):",
        "    ...",
    ]


def test_renderer_sorts_class_methods_only_by_name() -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[
            Class(
                name="Example",
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

    lines = StubRenderer(include_docstrings=False).render_module(module)

    assert "class Example:" in lines
    alpha_idx = next(i for i, l in enumerate(lines) if "def alpha(" in l)
    middle_idx = next(i for i, l in enumerate(lines) if "def middle(" in l)
    zeta_idx = next(i for i, l in enumerate(lines) if "def zeta(" in l)
    assert alpha_idx < middle_idx < zeta_idx


def test_stub_writer_writes_regular_module_file(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="foo",
                signatures=[
                    _signature(
                        return_type=RawType.int_,
                    )
                ],
            )
        ],
    )

    StubWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    stub_path = tmp_path / "mod.pyi"
    assert stub_path.exists()
    assert stub_path.read_text(encoding="utf-8") == "def foo() -> int:\n    ...\n"
