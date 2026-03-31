from __future__ import annotations

import typing

from pcstubgen.type_system import RawType, Type
from pcstubgen.ir import IRArgument, IRArgumentKind, IRFunction, IRMethod, IRModule, IRSignature, QualifiedName
from pcstubgen.stub_output import StubRenderer


def _signature(
    *,
    args: list[IRArgument] | None = None,
    return_type: Type | None = None,
    doc: str | None = None,
) -> IRSignature:
    """构造测试用签名。"""
    return IRSignature(
        args=list(args or ()),
        return_type=return_type,
        doc=doc,
    )


def _raw(text: str, *, imports: tuple[str, ...] = ()) -> RawType:
    return RawType(text, imports=imports)

def _unknown_function(name: str, *, doc: str | None = None) -> IRFunction:
    """构造签名未知的测试函数。"""
    return IRFunction(name=name, doc=doc)


def test_renderer_preserves_raw_optional_annotation_text() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[IRArgument(name="value", type=_raw("typing.Optional[int]", imports=("typing",)))],
                return_type=_raw("typing.Optional[int]", imports=("typing",)),
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(value: typing.Optional[int]) -> typing.Optional[int]:",
        "    ...",
    ]


def test_renderer_prints_default_values_as_is() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[IRArgument(name="value", default_value="unknown_default()")],
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(value = unknown_default()):",
        "    ...",
    ]


def test_renderer_prints_ellipsis_for_unknown_default_value() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[IRArgument(name="value", has_default=True)],
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(value = ...):",
        "    ...",
    ]


def test_renderer_keeps_zero_argument_function_on_single_line() -> None:
    func = IRFunction(
        name="foo",
        signatures=[_signature()],
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo():",
        "    ...",
    ]


def test_renderer_prints_placeholder_signature_for_unknown_function() -> None:
    func = IRFunction(name="foo")

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(",
        "    *args,",
        "    **kwargs,",
        "):",
        "    ...",
    ]


def test_renderer_prints_positional_only_marker_on_its_own_line() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[
                    IRArgument(name="x", kind=IRArgumentKind.POSITIONAL_ONLY),
                    IRArgument(name="y"),
                ],
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(",
        "    x,",
        "    /,",
        "    y,",
        "):",
        "    ...",
    ]


def test_renderer_prints_bare_star_on_its_own_line() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[
                    IRArgument(name="x"),
                    IRArgument(name="y", kind=IRArgumentKind.KEYWORD_ONLY),
                    IRArgument(name="z", kind=IRArgumentKind.KEYWORD_ONLY),
                ],
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(",
        "    x,",
        "    *,",
        "    y,",
        "    z,",
        "):",
        "    ...",
    ]


def test_renderer_prints_var_args_and_var_kwargs_on_their_own_lines() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[
                    IRArgument(name="x"),
                    IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
                    IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
                ],
            )
        ],
    )

    lines = StubRenderer(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(",
        "    x,",
        "    *args,",
        "    **kwargs,",
        "):",
        "    ...",
    ]


def test_renderer_prints_c_inferred_source_comment_after_function() -> None:
    func = IRFunction(
        name="foo",
        signatures=[_signature(args=[IRArgument(name="value", type=_raw("int"))])],
        c_inferred_source_comment="static int foo_impl(int value) {\n    return value;\n}",
    )

    lines = StubRenderer(
        include_docstrings=False,
        include_c_inferred_source_comment=True,
    ).print_function(func)

    assert lines == [
        "def foo(value: int):",
        "    ...",
        "#   C inferred source for foo:",
        "#   static int foo_impl(int value) {",
        "#       return value;",
        "#   }",
    ]


def test_renderer_prints_c_inferred_source_comment_once_after_multiline_overloads() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[IRArgument(name="value", type=_raw("int")), IRArgument(name="flag", type=_raw("bool"))],
                return_type=_raw("int"),
            ),
            _signature(
                args=[IRArgument(name="value", type=_raw("str")), IRArgument(name="flag", type=_raw("bool"))],
                return_type=_raw("str"),
            ),
        ],
        c_inferred_source_comment="static PyObject* foo_impl(PyObject* self, PyObject* args) {\n    return self;\n}",
    )

    lines = StubRenderer(
        include_docstrings=False,
        include_c_inferred_source_comment=True,
    ).print_function(func)

    assert lines == [
        "@typing.overload",
        "def foo(",
        "    value: int,",
        "    flag: bool,",
        ") -> int:",
        "    ...",
        "@typing.overload",
        "def foo(",
        "    value: str,",
        "    flag: bool,",
        ") -> str:",
        "    ...",
        "#   C inferred source for foo:",
        "#   static PyObject* foo_impl(PyObject* self, PyObject* args) {",
        "#       return self;",
        "#   }",
    ]


def test_renderer_prints_c_inferred_source_comment_once_after_overloads() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(args=[IRArgument(name="value", type=_raw("int"))], return_type=_raw("int")),
            _signature(args=[IRArgument(name="value", type=_raw("str"))], return_type=_raw("str")),
        ],
        c_inferred_source_comment="static PyObject* foo_impl(PyObject* self, PyObject* args) {\n    return self;\n}",
    )

    lines = StubRenderer(
        include_docstrings=False,
        include_c_inferred_source_comment=True,
    ).print_function(func)

    assert lines == [
        "@typing.overload",
        "def foo(value: int) -> int:",
        "    ...",
        "@typing.overload",
        "def foo(value: str) -> str:",
        "    ...",
        "#   C inferred source for foo:",
        "#   static PyObject* foo_impl(PyObject* self, PyObject* args) {",
        "#       return self;",
        "#   }",
    ]


def test_renderer_skips_c_inferred_source_comment_when_disabled() -> None:
    func = IRFunction(
        name="foo",
        signatures=[_signature(args=[IRArgument(name="value", type=_raw("int"))])],
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
                signatures=[
                    _signature(args=[IRArgument(name="x", type=_raw("int"))], return_type=_raw("int")),
                    _signature(
                        args=[IRArgument(name="x", type=_raw("typing.Optional[int]", imports=("typing",)))],
                        return_type=_raw("typing.Optional[int]", imports=("typing",)),
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
            signatures=[
                _signature(args=[IRArgument(name="x", type=_raw("int"))], return_type=_raw("int")),
                _signature(args=[IRArgument(name="x", type=_raw("str"))], return_type=_raw("str")),
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
