from __future__ import annotations

import typing

import pytest

from core.ir import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRSignature,
    QualifiedName,
)
import core.module_builder as module_builder_module
from core.module_builder import build_function
from core.node_visitors.doc_string_signature_parser_visitor import (
    DocStringSignatureParserVisitor,
)
from core.node_visitors.node_visitor import NodeVisitor
from core.printer_visitor import PrinterVisitor


def _signature(
    *,
    args: list[IRArgument] | None = None,
    return_type_name: str | None = None,
    doc: str | None = None,
) -> IRSignature:
    """构造测试用签名。"""
    return IRSignature(
        args=list(args or ()),
        return_type_name=return_type_name,
        doc=doc,
    )


def _unknown_function(name: str, *, doc: str | None = None) -> IRFunction:
    """构造签名未知的测试函数。"""
    return IRFunction(name=name, doc=doc)


def test_docstring_parser_parses_generic_function_signature() -> None:
    visitor = DocStringSignatureParserVisitor()
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.functions = [
        _unknown_function(
            "foo",
            doc="foo(x: int, y: str) -> str\n\nparsed from docstring",
        )
    ]

    visitor.visit_module(ir_module)

    parsed = ir_module.functions[0]
    assert len(parsed.signatures) == 1
    signature = parsed.signatures[0]
    assert [arg.name for arg in signature.args] == ["x", "y"]
    assert signature.return_type_name == "str"
    assert signature.doc == "parsed from docstring"


def test_docstring_parser_parses_pybind11_style_signature_with_defaults() -> None:
    visitor = DocStringSignatureParserVisitor()
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.functions = [
        _unknown_function(
            "cdist_minkowski",
            doc=(
                "cdist_minkowski(x: object, y: object, w: object = None, "
                "out: object = None, p: typing.SupportsFloat = 2.0) -> numpy.ndarray"
            ),
        )
    ]

    visitor.visit_module(ir_module)

    parsed = ir_module.functions[0]
    assert len(parsed.signatures) == 1
    signature = parsed.signatures[0]
    assert [arg.name for arg in signature.args] == ["x", "y", "w", "out", "p"]
    assert [arg.type_name for arg in signature.args] == [
        "object",
        "object",
        "object",
        "object",
        "typing.SupportsFloat",
    ]
    assert [arg.default_value for arg in signature.args] == [
        None,
        None,
        "None",
        "None",
        "2.0",
    ]
    assert signature.return_type_name == "numpy.ndarray"


def test_docstring_parser_preserves_pybind11_enum_default_value_text() -> None:
    visitor = DocStringSignatureParserVisitor()
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.functions = [
        _unknown_function(
            "foo",
            doc="foo(value: object = <demo.Color.RED: 1>) -> None",
        )
    ]

    visitor.visit_module(ir_module)

    parsed = ir_module.functions[0]
    assert len(parsed.signatures) == 1
    signature = parsed.signatures[0]
    assert [arg.default_value for arg in signature.args] == ["<demo.Color.RED: 1>"]
    assert signature.return_type_name == "None"


def test_docstring_parser_preserves_complex_generic_annotation_text() -> None:
    visitor = DocStringSignatureParserVisitor()
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.functions = [
        _unknown_function(
            "foo",
            doc=(
                "foo(value: typing.Optional[list[int]], "
                "item: dict[str, tuple[int, str]]) -> typing.Union[int, str]"
            ),
        )
    ]

    visitor.visit_module(ir_module)

    parsed = ir_module.functions[0]
    assert len(parsed.signatures) == 1
    signature = parsed.signatures[0]
    assert [arg.type_name for arg in signature.args] == [
        "typing.Optional[list[int]]",
        "dict[str, tuple[int, str]]",
    ]
    assert signature.return_type_name == "typing.Union[int, str]"


def test_docstring_parser_parse_args_str_supports_nested_defaults_and_markers() -> None:
    visitor = DocStringSignatureParserVisitor()

    parsed = visitor.parse_args_str(
        'a: int, /, x: tuple[int, int] = (1, 2), *, '
        'mapping: dict[str, int] = {"a": 1, "b": 2}, flag: str = "x"'
    )

    assert parsed is not None
    assert [arg.name for arg in parsed] == ["a", "x", "mapping", "flag"]
    assert [arg.type_name for arg in parsed] == [
        "int",
        "tuple[int, int]",
        "dict[str, int]",
        "str",
    ]
    assert [arg.default_value for arg in parsed] == [
        None,
        "(1, 2)",
        '{"a": 1, "b": 2}',
        '"x"',
    ]
    assert [arg.kind for arg in parsed] == [
        IRArgumentKind.POSITIONAL_ONLY,
        IRArgumentKind.POSITIONAL_OR_KEYWORD,
        IRArgumentKind.KEYWORD_ONLY,
        IRArgumentKind.KEYWORD_ONLY,
    ]


def test_docstring_parser_parse_args_str_supports_var_args_and_var_kwargs() -> None:
    visitor = DocStringSignatureParserVisitor()

    parsed = visitor.parse_args_str(
        "value: typing.Optional[list[int]], *args: tuple[str, ...], **kwargs: object"
    )

    assert parsed is not None
    assert [arg.name for arg in parsed] == ["value", "args", "kwargs"]
    assert [arg.type_name for arg in parsed] == [
        "typing.Optional[list[int]]",
        "tuple[str, ...]",
        "object",
    ]
    assert [arg.kind for arg in parsed] == [
        IRArgumentKind.POSITIONAL_OR_KEYWORD,
        IRArgumentKind.VAR_POSITIONAL,
        IRArgumentKind.VAR_KEYWORD,
    ]


def test_docstring_parser_parse_args_str_supports_full_marker_ordering() -> None:
    visitor = DocStringSignatureParserVisitor()

    parsed = visitor.parse_args_str(
        "a, /, b, *args: tuple[str, ...], c: int, **kwargs: object"
    )

    assert parsed is not None
    assert [arg.name for arg in parsed] == ["a", "b", "args", "c", "kwargs"]
    assert [arg.type_name for arg in parsed] == [
        None,
        None,
        "tuple[str, ...]",
        "int",
        "object",
    ]
    assert [arg.kind for arg in parsed] == [
        IRArgumentKind.POSITIONAL_ONLY,
        IRArgumentKind.POSITIONAL_OR_KEYWORD,
        IRArgumentKind.VAR_POSITIONAL,
        IRArgumentKind.KEYWORD_ONLY,
        IRArgumentKind.VAR_KEYWORD,
    ]


def test_docstring_parser_parse_args_str_supports_slash_then_bare_star() -> None:
    visitor = DocStringSignatureParserVisitor()

    parsed = visitor.parse_args_str(
        'a: int, /, b: int = 1, *, c: str, d: str = "x"'
    )

    assert parsed is not None
    assert [arg.name for arg in parsed] == ["a", "b", "c", "d"]
    assert [arg.type_name for arg in parsed] == ["int", "int", "str", "str"]
    assert [arg.default_value for arg in parsed] == [None, "1", None, '"x"']
    assert [arg.kind for arg in parsed] == [
        IRArgumentKind.POSITIONAL_ONLY,
        IRArgumentKind.POSITIONAL_OR_KEYWORD,
        IRArgumentKind.KEYWORD_ONLY,
        IRArgumentKind.KEYWORD_ONLY,
    ]


def test_docstring_parser_parse_args_str_supports_whitespace_around_arg_heads() -> None:
    visitor = DocStringSignatureParserVisitor()

    parsed_with_markers = visitor.parse_args_str(
        ' value : int ,  /  ,  named : str = "x" ,  *  ,  kw : bool = True '
    )
    parsed_with_var_args = visitor.parse_args_str(
        ' value : int ,  *args : tuple[str, ...] ,  kw : bool = True ,'
        '  **kwargs : object '
    )

    assert parsed_with_markers is not None
    assert [arg.name for arg in parsed_with_markers] == ["value", "named", "kw"]
    assert [arg.type_name for arg in parsed_with_markers] == [
        "int",
        "str",
        "bool",
    ]
    assert [arg.default_value for arg in parsed_with_markers] == [None, '"x"', "True"]
    assert [arg.kind for arg in parsed_with_markers] == [
        IRArgumentKind.POSITIONAL_ONLY,
        IRArgumentKind.POSITIONAL_OR_KEYWORD,
        IRArgumentKind.KEYWORD_ONLY,
    ]

    assert parsed_with_var_args is not None
    assert [arg.name for arg in parsed_with_var_args] == ["value", "args", "kw", "kwargs"]
    assert [arg.type_name for arg in parsed_with_var_args] == [
        "int",
        "tuple[str, ...]",
        "bool",
        "object",
    ]
    assert [arg.default_value for arg in parsed_with_var_args] == [None, None, "True", None]
    assert [arg.kind for arg in parsed_with_var_args] == [
        IRArgumentKind.POSITIONAL_OR_KEYWORD,
        IRArgumentKind.VAR_POSITIONAL,
        IRArgumentKind.KEYWORD_ONLY,
        IRArgumentKind.VAR_KEYWORD,
    ]


@pytest.mark.parametrize(
    "args_str",
    [
        "value: tuple[int, str",
        'value: str = "unterminated',
        "x: int,, y: int",
        "x: int: str",
        "x = 1 = 2",
        "/, a: int",
        "a: int, /, /",
        "a: int, *, *",
        "a: int, *: int",
        "a: int, **kwargs: object, b: int",
        "*: int",
        "**: int",
        "*1x",
        "**class",
        "*args = ()",
        "**kwargs = {}",
        "a: int, *, /, b: int",
        "a: int, *args: tuple[int, ...], *, b: int",
    ],
)
def test_docstring_parser_parse_args_str_rejects_invalid_input(args_str: str) -> None:
    visitor = DocStringSignatureParserVisitor()

    assert visitor.parse_args_str(args_str) is None


def test_module_builder_keeps_raw_annotation_strings() -> None:
    def sample(a: int, b: list[int]) -> typing.Optional[int]:
        raise NotImplementedError

    parsed = build_function(QualifiedName.from_str("pkg.mod.sample"), sample)

    assert len(parsed.signatures) == 1
    signature = parsed.signatures[0]
    assert [arg.type_name for arg in signature.args] == ["int", "list[int]"]
    assert signature.return_type_name == "typing.Optional[int]"


def test_module_builder_keeps_default_values_as_strings() -> None:
    def sample(
        flag: bool = False,
        values: tuple[int, int] = (1, 2),
    ) -> None:
        raise NotImplementedError

    parsed = build_function(QualifiedName.from_str("pkg.mod.sample"), sample)

    assert len(parsed.signatures) == 1
    signature = parsed.signatures[0]
    assert [arg.default_value for arg in signature.args] == ["False", "(1, 2)"]


def test_module_builder_uses_empty_signatures_when_inspect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample() -> None:
        raise NotImplementedError

    def _raise_signature_error(obj: object) -> object:
        """模拟 inspect.signature 失败。"""
        raise TypeError(f"cannot inspect {obj!r}")

    monkeypatch.setattr(module_builder_module.inspect, "signature", _raise_signature_error)

    parsed = build_function(QualifiedName.from_str("pkg.mod.sample"), sample)

    assert parsed.signatures == []


def test_printer_preserves_raw_optional_annotation_text() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[IRArgument(name="value", type_name="typing.Optional[int]")],
                return_type_name="typing.Optional[int]",
            )
        ],
    )

    lines = PrinterVisitor(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(value: typing.Optional[int]) -> typing.Optional[int]:",
        "    ...",
    ]


def test_printer_prints_default_values_as_is() -> None:
    func = IRFunction(
        name="foo",
        signatures=[
            _signature(
                args=[IRArgument(name="value", default_value="unknown_default()")],
            )
        ],
    )

    lines = PrinterVisitor(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(value = unknown_default()):",
        "    ...",
    ]


def test_printer_prints_placeholder_signature_for_unknown_function() -> None:
    func = IRFunction(name="foo")

    lines = PrinterVisitor(include_docstrings=False).print_function(func)

    assert lines == [
        "def foo(*args, **kwargs):",
        "    ...",
    ]


def test_printer_adds_typing_import_for_overloads() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="foo",
                signatures=[
                    _signature(args=[IRArgument(name="x", type_name="int")], return_type_name="int"),
                    _signature(
                        args=[IRArgument(name="x", type_name="typing.Optional[int]")],
                        return_type_name="typing.Optional[int]",
                    ),
                ],
            )
        ],
    )

    lines = PrinterVisitor(include_docstrings=False).visit_module(module)

    assert lines == [
        "import typing",
        "@typing.overload",
        "def foo(x: int) -> int:",
        "    ...",
        "@typing.overload",
        "def foo(x: typing.Optional[int]) -> typing.Optional[int]:",
        "    ...",
    ]


def test_printer_repeats_method_decorator_for_each_overload() -> None:
    method = IRMethod(
        function=IRFunction(
            name="build",
            signatures=[
                _signature(args=[IRArgument(name="x", type_name="int")], return_type_name="int"),
                _signature(args=[IRArgument(name="x", type_name="str")], return_type_name="str"),
            ],
        ),
        decorator="classmethod",
    )

    lines = PrinterVisitor(include_docstrings=False).print_method(method)

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


def test_node_visitor_inplace_mutation_removes_classes_functions_and_methods() -> None:
    class DropByNameVisitor(NodeVisitor):
        def visit_module(self, node: IRModule) -> None:
            node.classes = [cls for cls in node.classes if not cls.name.startswith("Drop")]
            node.functions = [
                func for func in node.functions if not func.name.startswith("drop")
            ]
            super().visit_module(node)

        def visit_class(self, node: IRClass, module: IRModule) -> None:
            node.classes = [cls for cls in node.classes if not cls.name.startswith("Drop")]
            node.methods = [
                method
                for method in node.methods
                if not method.function.name.startswith("drop")
            ]
            super().visit_class(node, module)

    keep_class = IRClass(
        name="KeepClass",
        classes=[IRClass(name="DropNested"), IRClass(name="KeepNested")],
        methods=[
            IRMethod(function=IRFunction(name="keep_method"), decorator=None),
            IRMethod(function=IRFunction(name="drop_method"), decorator=None),
        ],
    )
    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[IRClass(name="DropClass"), keep_class],
        functions=[IRFunction(name="drop_func"), IRFunction(name="keep_func")],
    )

    DropByNameVisitor().visit_module(ir_module)

    assert [cls.name for cls in ir_module.classes] == ["KeepClass"]
    assert [func.name for func in ir_module.functions] == ["keep_func"]
    assert [cls.name for cls in keep_class.classes] == ["KeepNested"]
    assert [method.function.name for method in keep_class.methods] == ["keep_method"]


def test_node_visitor_visits_functions_in_module_and_methods() -> None:
    class RenameVisitedFunctionsVisitor(NodeVisitor):
        def visit_function(self, node: IRFunction) -> None:
            node.name = f"visited_{node.name}"
            super().visit_function(node)

    method = IRMethod(function=IRFunction(name="m"), decorator=None)
    ir_class = IRClass(name="C", methods=[method])
    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[ir_class],
        functions=[IRFunction(name="f")],
    )

    RenameVisitedFunctionsVisitor().visit_module(ir_module)

    assert [func.name for func in ir_module.functions] == ["visited_f"]
    assert [m.function.name for m in ir_class.methods] == ["visited_m"]
