from __future__ import annotations

import re
from pathlib import Path

import clang.cindex
import pytest

from pcstubgen.models import ArgumentKind
from pcstubgen.signature_completion.c_extension.method_flags import (
    METH_KEYWORDS,
    METH_VARARGS,
)
from pcstubgen.signature_completion.c_extension.signatures import inferencer as signature_rules_module
from pcstubgen.type_models import RawType, UnionType
from tests._c_extension_test_support import (
    _FakeCanonicalType,
    _FakeNode,
    _FakeToken,
    _address_of,
    _address_of_expr,
    _arg,
    _array_subscript,
    _assignment,
    _call_expr,
    _conditional_expr,
    _cxx_bool_literal,
    _expr_assignment,
    _extent_for_source_snippet,
    _fake_function_cursor_with_children,
    _float_literal,
    _identifier_node,
    _init_list,
    _int_literal,
    _location_text,
    _null_ptr_literal,
    _python_singleton_default_expr,
    _return_stmt,
    _string_literal,
    _token_identifier_node,
    _unary_default_expr,
    _var_decl,
    patch_inference_clang_helpers,
)


@pytest.fixture(autouse=True)
def _patch_fake_clang_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_inference_clang_helpers(monkeypatch, signature_rules_module)


def _infer_varargs_arguments(cursor: clang.cindex.Cursor) -> list[list[object]]:
    return signature_rules_module.infer_arguments_list(cursor, flags=METH_VARARGS)


def _infer_varargs_keywords_arguments(cursor: clang.cindex.Cursor) -> list[list[object]]:
    return signature_rules_module.infer_arguments_list(
        cursor,
        flags=METH_VARARGS | METH_KEYWORDS,
    )


def _typing_literal(text: str) -> RawType:
    return RawType(f"typing.Literal[{text}]", imports=("typing",))


def test_infer_argument_lists_parses_pyarg_parsetuple() -> None:
    count_decl = _var_decl("count", _int_literal("1"))
    label_decl = _var_decl("label", _identifier_node("Py_None"))
    label_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i|z"),
            _address_of("count", referenced=count_decl),
            _address_of("label", referenced=label_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [
        [
            _arg("count", "int"),
            _arg(
                "label",
                UnionType((RawType("str"), RawType("None"))),
                default_value="...",
            ),
        ]
    ]

def test_infer_argument_lists_maps_pyarg_p_unit_to_bool() -> None:
    flag_decl = _var_decl("flag")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("p"),
            _address_of("flag", referenced=flag_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("flag", "bool")]]

@pytest.mark.parametrize(
    ("struct_name", "expected_default"),
    [
        ("_Py_NoneStruct", "None"),
        ("_Py_TrueStruct", "True"),
        ("_Py_FalseStruct", "False"),
    ],
)
def test_infer_argument_lists_renders_python_singleton_default_values(
    struct_name: str,
    expected_default: str,
) -> None:
    value_decl = _var_decl("value", _python_singleton_default_expr(struct_name))
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|O"),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "object", default_value=expected_default)]]

def test_infer_argument_lists_renders_pointer_zero_default_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializer = _int_literal("0")
    value_decl = _var_decl("value", initializer)
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|O"),
            _address_of("value", referenced=value_decl),
        )
    )
    observed: list[_FakeNode] = []
    monkeypatch.setattr(
        signature_rules_module,
        "is_nullptr_or_zero",
        lambda received_cursor: observed.append(received_cursor) or True,
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "object", default_value="...")]]
    assert observed == [initializer]

@pytest.mark.parametrize(
    ("literal_value", "expected_default"),
    [
        ("", "''"),
        ("rates", "'rates'"),
    ],
)
def test_infer_default_value_for_pyarg_renders_string_default_as_python_literal(
    literal_value: str,
    expected_default: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializer = _string_literal(literal_value)
    value_decl = _var_decl("value", initializer)
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    monkeypatch.setattr(signature_rules_module, "is_nullptr_or_zero", lambda _: False)

    assert (
        signature_rules_module._infer_default_value_for_pyarg(
            _address_of("value", referenced=value_decl),
            RawType("str"),
        )
        == expected_default
    )

def test_infer_argument_lists_renders_char_pointer_null_default_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializer = _int_literal("0")
    value_decl = _var_decl("value", initializer)
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|z"),
            _address_of("value", referenced=value_decl),
        )
    )
    monkeypatch.setattr(signature_rules_module, "is_nullptr_or_zero", lambda _: True)

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [
        [
            _arg(
                "value",
                UnionType((RawType("str"), RawType("None"))),
                default_value="...",
            )
        ]
    ]

def test_infer_argument_lists_keeps_non_pointer_zero_default_as_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializer = _int_literal("0")
    value_decl = _var_decl("value", initializer)
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|i"),
            _address_of("value", referenced=value_decl),
        )
    )
    observed: list[_FakeNode] = []
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: observed.append(received_cursor) or 0,
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "int", default_value="0")]]
    assert observed == [initializer]

def test_infer_argument_lists_renders_default_from_assignment_before_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value_decl = _var_decl("value")
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    default_expr = _int_literal("12")
    cursor = _fake_function_cursor_with_children(
        value_decl,
        _assignment("value", default_expr, referenced=value_decl),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|i"),
            _address_of("value", referenced=value_decl),
        ),
    )
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 12)

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "int", default_value="12")]]

def test_infer_argument_lists_uses_last_assignment_before_parse_for_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializer = _int_literal("1")
    default_expr = _int_literal("2")
    value_decl = _var_decl("value", initializer)
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        value_decl,
        _assignment("value", default_expr, referenced=value_decl),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|i"),
            _address_of("value", referenced=value_decl),
        ),
    )
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 2)

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "int", default_value="2")]]

def test_infer_argument_lists_ignores_assignment_after_parse_for_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializer = _int_literal("1")
    ignored_expr = _int_literal("2")
    value_decl = _var_decl("value", initializer)
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        value_decl,
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|i"),
            _address_of("value", referenced=value_decl),
        ),
        _assignment("value", ignored_expr, referenced=value_decl),
    )
    evaluated_values = {
        id(initializer): 1,
        id(ignored_expr): 2,
    }
    observed: list[_FakeNode] = []
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: observed.append(received_cursor)
        or evaluated_values[id(received_cursor)],
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "int", default_value="1")]]
    assert observed == [initializer]

def test_infer_argument_lists_renders_integer_defaults_from_chained_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_decl = _var_decl("left")
    right_decl = _var_decl("right")
    left_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    right_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    default_expr = _int_literal("512")
    cursor = _fake_function_cursor_with_children(
        left_decl,
        right_decl,
        _assignment(
            "left",
            _assignment("right", default_expr, referenced=right_decl),
            referenced=left_decl,
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|(ii)"),
            _address_of("left", referenced=left_decl),
            _address_of("right", referenced=right_decl),
        ),
    )
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 512)

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [
        [_arg("left_right", "tuple[int, int]", default_value="(512, 512)")]
    ]

def test_infer_argument_lists_renders_float_defaults_from_chained_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_decl = _var_decl("left")
    right_decl = _var_decl("right")
    left_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.DOUBLE)
    right_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.DOUBLE)
    default_expr = _unary_default_expr(_float_literal("1.0"))
    cursor = _fake_function_cursor_with_children(
        left_decl,
        right_decl,
        _assignment(
            "left",
            _assignment("right", default_expr, referenced=right_decl),
            referenced=left_decl,
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|(dd)"),
            _address_of("left", referenced=left_decl),
            _address_of("right", referenced=right_decl),
        ),
    )
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: -1.0)

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [
        [_arg("left_right", "tuple[float, float]", default_value="(-1.0, -1.0)")]
    ]

@pytest.mark.parametrize(
    ("evaluated", "expected_default"),
    [
        (0, "False"),
        (1, "True"),
    ],
)
def test_infer_argument_lists_renders_integer_bool_default_values(
    monkeypatch: pytest.MonkeyPatch,
    evaluated: int,
    expected_default: str,
) -> None:
    initializer = _int_literal(str(evaluated))
    flag_decl = _var_decl("flag", initializer)
    flag_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|p"),
            _address_of("flag", referenced=flag_decl),
        )
    )
    observed: list[_FakeNode] = []
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: observed.append(received_cursor) or evaluated,
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("flag", "bool", default_value=expected_default)]]
    assert observed == [initializer]

@pytest.mark.parametrize(
    ("format_unit", "argument_type", "evaluated", "expected_default"),
    [
        ("p", "bool", 0, "False"),
        ("p", "bool", 1, "True"),
        ("i", "int", 0, "0"),
    ],
)
def test_infer_argument_lists_renders_cxx_bool_default_values(
    monkeypatch: pytest.MonkeyPatch,
    format_unit: str,
    argument_type: str,
    evaluated: int,
    expected_default: str,
) -> None:
    initializer = _cxx_bool_literal()
    value_decl = _var_decl("value", initializer)
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal(f"|{format_unit}"),
            _address_of("value", referenced=value_decl),
        )
    )
    observed: list[_FakeNode] = []
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: observed.append(received_cursor) or evaluated,
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", argument_type, default_value=expected_default)]]
    assert observed == [initializer]

@pytest.mark.parametrize(
    ("format_unit", "argument_type", "type_kind", "evaluated", "expected_default"),
    [
        ("i", "int", clang.cindex.TypeKind.INT, -1, "-1"),
        ("i", "int", clang.cindex.TypeKind.INT, -10, "-10"),
        ("d", "float", clang.cindex.TypeKind.DOUBLE, -1.0, "-1.0"),
    ],
)
def test_infer_argument_lists_renders_numeric_unary_default_values(
    monkeypatch: pytest.MonkeyPatch,
    format_unit: str,
    argument_type: str,
    type_kind: object,
    evaluated: int | float,
    expected_default: str,
) -> None:
    initializer = _unary_default_expr(_int_literal("1"))
    value_decl = _var_decl("value", initializer)
    value_decl.type = _FakeCanonicalType(type_kind)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal(f"|{format_unit}"),
            _address_of("value", referenced=value_decl),
        )
    )
    observed: list[_FakeNode] = []
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: observed.append(received_cursor) or evaluated,
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", argument_type, default_value=expected_default)]]
    assert observed == [initializer]

def test_infer_argument_lists_keeps_pointer_unary_default_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value_decl = _var_decl("value", _unary_default_expr(_int_literal("1")))
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|O"),
            _address_of("value", referenced=value_decl),
        )
    )
    observed: list[_FakeNode] = []
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: observed.append(received_cursor) or -1,
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "object", default_value="...")]]
    assert observed == []

def test_infer_type_object_type_for_pyarg_reads_name_from_extent_source_text(tmp_path: Path) -> None:
    source = tmp_path / "object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&PyUnicode_Type), &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent=_extent_for_source_snippet(source, "(&PyUnicode_Type)"),
    )

    inferred = signature_rules_module._infer_type_object_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "str"

def test_infer_type_object_type_for_pyarg_reads_numpy_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "numpy_object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&PyArray_Type), &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent=_extent_for_source_snippet(source, "(&PyArray_Type)"),
    )

    inferred = signature_rules_module._infer_type_object_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray"

def test_infer_type_object_type_for_pyarg_reads_pillow_imaging_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pillow_imaging_object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", &Imaging_Type, &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent=_extent_for_source_snippet(source, "&Imaging_Type"),
    )

    inferred = signature_rules_module._infer_type_object_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "PIL.Image.core.ImagingCore"
    assert inferred.collect_imports() == {"PIL.Image"}

def test_infer_type_object_type_for_pyarg_reads_pillow_cms_profile_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pillow_cms_profile_object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", &CmsProfile_Type, &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent=_extent_for_source_snippet(source, "&CmsProfile_Type"),
    )

    inferred = signature_rules_module._infer_type_object_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "PIL.ImageCms.core.CmsProfile"
    assert inferred.collect_imports() == {"PIL.ImageCms"}

@pytest.mark.parametrize(
    ("call_name", "expected", "imports"),
    [
        ("PyImagingNew", "PIL.Image.core.ImagingCore", {"PIL.Image"}),
        ("PyImaging_DecoderNew", "PIL.Image.core.ImagingDecoder", {"PIL.Image"}),
        ("PyImaging_EncoderNew", "PIL.Image.core.ImagingEncoder", {"PIL.Image"}),
        ("cms_profile_new", "PIL.ImageCms.core.CmsProfile", {"PIL.ImageCms"}),
        ("cms_transform_new", "PIL.ImageCms.core.CmsTransform", {"PIL.ImageCms"}),
        ("_outline_new", "PIL.Image.core._Outline", {"PIL.Image"}),
        ("path_new", "PIL.ImagePath.Path", {"PIL.ImagePath"}),
    ],
)
def test_infer_expr_type_detects_pillow_factory_mappings(
    call_name: str,
    expected: str,
    imports: set[str],
) -> None:
    inferred = signature_rules_module.infer_expr_type(_call_expr(call_name, _identifier_node("arg")))

    assert inferred.render() == expected
    assert inferred.collect_imports() == imports

@pytest.mark.parametrize(
    "call_name",
    [
        "ImagingError_MemoryError",
        "ImagingError_ValueError",
        "HandleMuxError",
        "geterror",
    ],
)
def test_infer_expr_type_detects_pillow_error_return_factories(call_name: str) -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            call_name,
            _identifier_node("arg"),
        )
    )

    assert inferred == UnionType(())

def test_infer_expr_type_detects_pyobject_new_from_type_object() -> None:
    type_object_arg = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent="&ImagingEncoder_Type",
        children=[_identifier_node("ImagingEncoder_Type")],
    )
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "PyObject_New",
            _identifier_node("ImagingEncoderObject"),
            type_object_arg,
        )
    )

    assert inferred.render() == "PIL.Image.core.ImagingEncoder"
    assert inferred.collect_imports() == {"PIL.Image"}

def test_infer_expr_type_detects_pyobject_new_macro_expansion_shape() -> None:
    type_object_arg = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent="&ImagingDraw_Type",
        children=[_identifier_node("ImagingDraw_Type")],
    )
    call_expr = _FakeNode(
        kind=clang.cindex.CursorKind.CALL_EXPR,
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, "PyObject_New")],
        spelling="PyObject_New",
        children=[
            _FakeNode(
                kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
                spelling="PyObject_New",
                extent="PyObject_New(ImagingDrawObject, &ImagingDraw_Type)",
                children=[_token_identifier_node("PyObject_New")],
            ),
            type_object_arg,
        ],
    )

    inferred = signature_rules_module.infer_expr_type(call_expr)

    assert inferred.render() == "PIL.Image.core.ImagingDraw"
    assert inferred.collect_imports() == {"PIL.Image"}

def test_infer_expr_type_detects_private_pyobject_new_call() -> None:
    type_object_arg = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent="&Font_Type",
        children=[_identifier_node("Font_Type")],
    )
    with pytest.raises(RuntimeError, match="无法识别的返回值工厂调用: _PyObject_New"):
        signature_rules_module.infer_expr_type(
            _call_expr(
                "_PyObject_New",
                type_object_arg,
            )
        )

def test_return_type_traces_local_decl_ref_initialized_from_pyobject_new() -> None:
    type_object_arg = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent="&ImagingDecoder_Type",
        children=[_identifier_node("ImagingDecoder_Type")],
    )
    self_decl = _var_decl(
        "self",
        _call_expr(
            "PyObject_New",
            _identifier_node("ImagingDecoderObject"),
            type_object_arg,
        ),
    )
    cursor = _fake_function_cursor_with_children(
        self_decl,
        _return_stmt(_token_identifier_node("self", referenced=self_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "PIL.Image.core.ImagingDecoder"
    assert inferred.collect_imports() == {"PIL.Image"}

def test_return_type_drops_pillow_error_return_factory_branch() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("HandleMuxError", _identifier_node("mux"))),
        _return_stmt(_call_expr("PyImagingNew", _identifier_node("image"))),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType(
        "PIL.Image.core.ImagingCore",
        imports=("PIL.Image",),
    )

def test_infer_type_object_type_for_pyarg_propagates_extent_source_text_read_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent="boom",
    )
    monkeypatch.setattr(
        signature_rules_module,
        "get_cursor_source_text",
        lambda node: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        signature_rules_module._infer_type_object_type_for_pyarg(cursor)

def test_infer_converter_type_for_pyarg_reads_numpy_converter_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "numpy_converter_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O&\", NI_ObjectToInputArray, &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _identifier_node("converter")
    cursor.extent = _extent_for_source_snippet(source, "NI_ObjectToInputArray")

    inferred = signature_rules_module._infer_converter_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray"

def test_infer_converter_type_for_pyarg_reads_optional_numpy_converter_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "optional_numpy_converter_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O&\", NI_ObjectToOptionalInputArray, &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _identifier_node("converter")
    cursor.extent = _extent_for_source_snippet(source, "NI_ObjectToOptionalInputArray")

    inferred = signature_rules_module._infer_converter_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray | None"

def test_infer_converter_type_for_pyarg_reads_tuple_converter_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tuple_converter_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O&\", PyArray_IntpConverter, &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _identifier_node("converter")
    cursor.extent = _extent_for_source_snippet(source, "PyArray_IntpConverter")

    inferred = signature_rules_module._infer_converter_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "tuple[int, ...]"


@pytest.mark.parametrize(
    ("converter_name", "expected"),
    [
        (
            "PyArray_OutputConverter",
            UnionType((RawType("numpy.ndarray", imports=("numpy",)), RawType("None"))),
        ),
        (
            "PyArray_AxisConverter",
            UnionType((RawType("int"), RawType("None"))),
        ),
        ("PyArray_BoolConverter", RawType("bool")),
        (
            "PyArray_OptionalBoolConverter",
            UnionType((RawType("bool"), RawType("None"))),
        ),
        (
            "PyArray_OrderConverter",
            UnionType((_typing_literal('"K", "A", "C", "F"'), RawType("None"))),
        ),
        (
            "PyArray_ByteorderConverter",
            _typing_literal('"S", "<", "L", "little", ">", "B", "big", "=", "N", "native", "|", "I"'),
        ),
        (
            "PyArray_CastingConverter",
            _typing_literal('"no", "equiv", "safe", "same_kind", "unsafe"'),
        ),
        (
            "PyArray_SearchsideConverter",
            _typing_literal('"left", "right"'),
        ),
        (
            "PyArray_SelectkindConverter",
            _typing_literal('"introselect"'),
        ),
        (
            "PyArray_SortkindConverter",
            _typing_literal('"Q", "quick", "quicksort", "M", "merge", "mergesort", "H", "heap", "heapsort", "S", "stable", "stablesort"'),
        ),
        (
            "PyArray_ClipmodeConverter",
            UnionType((_typing_literal('"clip", "wrap", "raise"'), RawType("int"))),
        ),
    ],
)
def test_infer_argument_lists_maps_numpy_low_risk_converters(
    converter_name: str,
    expected: RawType | UnionType,
) -> None:
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("O&"),
            _FakeNode(
                kind=clang.cindex.CursorKind.DECL_REF_EXPR,
                extent=converter_name,
            ),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", expected)]]


@pytest.mark.parametrize(
    ("type_object_name", "expected"),
    [
        ("PyArrayDescr_Type", RawType("numpy.dtype", imports=("numpy",))),
        (
            "NpyBusDayCalendar_Type",
            RawType("numpy.busdaycalendar", imports=("numpy",)),
        ),
        ("PyUFunc_Type", RawType("numpy.ufunc", imports=("numpy",))),
    ],
)
def test_infer_argument_lists_maps_numpy_low_risk_type_objects(
    type_object_name: str,
    expected: RawType,
) -> None:
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("O!"),
            _FakeNode(
                kind=clang.cindex.CursorKind.UNARY_OPERATOR,
                extent=f"&{type_object_name}",
            ),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", expected)]]

def test_infer_argument_lists_falls_back_to_object_for_unknown_o_bang_type(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unknown_object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&UnknownRuntimeType), &value);",
            ]
        ),
        encoding="utf-8",
    )
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("O!"),
            _FakeNode(
                kind=clang.cindex.CursorKind.UNARY_OPERATOR,
                extent=_extent_for_source_snippet(source, "(&UnknownRuntimeType)"),
            ),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "object")]]

def test_infer_argument_lists_falls_back_to_object_for_unknown_o_ampersand_converter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unknown_converter_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O&\", UnknownConverter, &value);",
            ]
        ),
        encoding="utf-8",
    )
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("O&"),
            _FakeNode(
                kind=clang.cindex.CursorKind.DECL_REF_EXPR,
                extent=_extent_for_source_snippet(source, "UnknownConverter"),
            ),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("value", "object")]]

def test_infer_argument_lists_falls_back_to_unknown_default_value_when_default_parse_fails() -> None:
    label_decl = _var_decl("label", _identifier_node("UNSUPPORTED_DEFAULT"))
    label_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|z"),
            _address_of("label", referenced=label_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [
        [
            _arg(
                "label",
                UnionType((RawType("str"), RawType("None"))),
                default_value="...",
            )
        ]
    ]

def test_infer_argument_lists_joins_decl_ref_names_for_tuple_arguments() -> None:
    left_decl = _var_decl("left")
    right_decl = _var_decl("right")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("(ii)"),
            _address_of("left", referenced=left_decl),
            _address_of("right", referenced=right_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("left_right", "tuple[int, int]")]]

def test_infer_argument_lists_parses_array_subscript_tuple_slots_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extent_decl = _var_decl("extent")
    extent_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.CONSTANTARRAY)
    indexes = [_int_literal(str(index)) for index in range(4)]
    values = [
        _unary_default_expr(_int_literal("3")),
        _unary_default_expr(_float_literal("2.5")),
        _int_literal("2"),
        _float_literal("2.5"),
    ]
    evaluated_values = {
        id(indexes[0]): 0,
        id(indexes[1]): 1,
        id(indexes[2]): 2,
        id(indexes[3]): 3,
        id(values[0]): -3,
        id(values[1]): -2.5,
        id(values[2]): 2,
        id(values[3]): 2.5,
    }
    cursor = _fake_function_cursor_with_children(
        extent_decl,
        _expr_assignment(
            _array_subscript("extent", indexes[0], referenced=extent_decl),
            values[0],
        ),
        _expr_assignment(
            _array_subscript("extent", indexes[1], referenced=extent_decl),
            values[1],
        ),
        _expr_assignment(
            _array_subscript("extent", indexes[2], referenced=extent_decl),
            values[2],
        ),
        _expr_assignment(
            _array_subscript("extent", indexes[3], referenced=extent_decl),
            values[3],
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|(dddd)"),
            _address_of_expr(_array_subscript("extent", indexes[0], referenced=extent_decl)),
            _address_of_expr(_array_subscript("extent", indexes[1], referenced=extent_decl)),
            _address_of_expr(_array_subscript("extent", indexes[2], referenced=extent_decl)),
            _address_of_expr(_array_subscript("extent", indexes[3], referenced=extent_decl)),
        ),
    )
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: evaluated_values[id(received_cursor)],
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [
        [
            _arg(
                "extent",
                "tuple[float, float, float, float]",
                default_value="(-3.0, -2.5, 2.0, 2.5)",
            )
        ]
    ]

def test_infer_argument_lists_marks_array_initializer_defaults_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extent_decl = _var_decl("extent", _init_list(_float_literal("1.0"), _float_literal("2.5")))
    extent_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.CONSTANTARRAY)
    indexes = [_int_literal("0"), _int_literal("1")]
    index_values = {
        id(indexes[0]): 0,
        id(indexes[1]): 1,
    }
    cursor = _fake_function_cursor_with_children(
        extent_decl,
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|(dd)"),
            _address_of_expr(_array_subscript("extent", indexes[0], referenced=extent_decl)),
            _address_of_expr(_array_subscript("extent", indexes[1], referenced=extent_decl)),
        ),
    )
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: index_values[id(received_cursor)],
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("extent", "tuple[float, float]", default_value="...")]]

def test_infer_argument_lists_marks_missing_array_initializer_items_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extent_decl = _var_decl("extent", _init_list(_int_literal("1")))
    extent_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.CONSTANTARRAY)
    index = _int_literal("2")
    cursor = _fake_function_cursor_with_children(
        extent_decl,
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|i"),
            _address_of_expr(_array_subscript("extent", index, referenced=extent_decl)),
        ),
    )
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: 2,
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("extent", "int", default_value="...")]]

def test_infer_argument_lists_array_assignment_overrides_initializer_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extent_decl = _var_decl("extent", _init_list(_float_literal("1.0"), _float_literal("2.5")))
    extent_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.CONSTANTARRAY)
    index = _int_literal("1")
    override_value = _float_literal("3.5")
    evaluated_values = {
        id(index): 1,
        id(override_value): 3.5,
    }
    cursor = _fake_function_cursor_with_children(
        extent_decl,
        _expr_assignment(
            _array_subscript("extent", index, referenced=extent_decl),
            override_value,
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|d"),
            _address_of_expr(_array_subscript("extent", index, referenced=extent_decl)),
        ),
    )
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: evaluated_values[id(received_cursor)],
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("extent", "float", default_value="3.5")]]

def test_infer_argument_lists_array_defaults_follow_chained_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extent_decl = _var_decl("extent")
    extent_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.CONSTANTARRAY)
    value_decl = _var_decl("value")
    value_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    index = _int_literal("0")
    default_expr = _int_literal("5")
    evaluated_values = {
        id(index): 0,
        id(default_expr): 5,
    }
    cursor = _fake_function_cursor_with_children(
        extent_decl,
        value_decl,
        _expr_assignment(
            _array_subscript("extent", index, referenced=extent_decl),
            _assignment("value", default_expr, referenced=value_decl),
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|i"),
            _address_of_expr(_array_subscript("extent", index, referenced=extent_decl)),
        ),
    )
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: evaluated_values[id(received_cursor)],
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("extent", "int", default_value="5")]]

def test_infer_default_value_for_pyarg_rejects_designated_array_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    designated_entry = _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        tokens=[_FakeToken(clang.cindex.TokenKind.PUNCTUATION, "[")],
    )
    extent_decl = _var_decl("extent", _init_list(designated_entry))
    extent_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.CONSTANTARRAY)
    index = _int_literal("0")
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0)

    with pytest.raises(RuntimeError, match="数组元素没有可用定值表达式"):
        signature_rules_module._infer_default_value_for_pyarg(
            _address_of_expr(_array_subscript("extent", index, referenced=extent_decl)),
            RawType("int"),
        )

def test_infer_argument_lists_falls_back_to_minimal_varargs_when_no_supported_pyarg_calls_exist() -> None:
    cursor = _fake_function_cursor_with_children(
        _call_expr("PyLong_FromLong", _identifier_node("value"))
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL)]]

def test_infer_argument_lists_skips_pyarg_parsetuple_sizet_alias() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "_PyArg_ParseTuple_SizeT",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL)]]

def test_infer_argument_lists_skips_pyarg_parsetuple_and_keywords_sizet_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0.0)
    kwlist_decl = _var_decl("kwlist", _init_list(_string_literal("x"), _null_ptr_literal()))
    x_decl = _var_decl("x", _float_literal("0.0"))
    x_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.DOUBLE)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "_PyArg_ParseTupleAndKeywords_SizeT",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("|d"),
            _token_identifier_node("kwlist", referenced=kwlist_decl),
            _address_of("x", referenced=x_decl),
        )
    )

    inferred = _infer_varargs_keywords_arguments(cursor)

    assert inferred == [[
        _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
        _arg("kwargs", "object", kind=ArgumentKind.VAR_KEYWORD),
    ]]

def test_infer_argument_lists_does_not_match_pyarg_from_single_character_name() -> None:
    cursor = _fake_function_cursor_with_children(
        _call_expr("e", _identifier_node("args"))
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL)]]

def test_infer_argument_lists_skips_parse_tuple_and_keywords_without_valid_kwlist() -> None:
    invalid_entry = _identifier_node("bad")
    invalid_entry.location = _location_text("kwlist.c:7:9")
    invalid_kwlist_decl = _var_decl("kwlist", _init_list(invalid_entry, _null_ptr_literal()))
    x_decl = _var_decl("x", _float_literal("0.0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTupleAndKeywords",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("|d"),
            _token_identifier_node("kwlist", referenced=invalid_kwlist_decl),
            _address_of("x", referenced=x_decl),
        )
    )

    inferred = _infer_varargs_keywords_arguments(cursor)

    assert inferred == [[
        _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
        _arg("kwargs", "object", kind=ArgumentKind.VAR_KEYWORD),
    ]]

def test_infer_argument_lists_skips_parse_tuple_when_format_string_is_not_literal() -> None:
    value_decl = _var_decl("value")
    format_cursor = _identifier_node("fmt")
    format_cursor.location = _location_text("parse_tuple.c:20:6")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            format_cursor,
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [[_arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL)]]

def test_infer_default_value_for_pyarg_raises_with_cursor_location_for_unsupported_expr() -> None:
    expr_cursor = _call_expr("PyLong_FromLong", _identifier_node("value"))
    expr_cursor.location = _location_text("default_value.c:15:3")
    target_decl = _var_decl("value", expr_cursor)
    target_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)

    with pytest.raises(
        RuntimeError,
        match=rf"不支持的默认值表达式类型.*{re.escape('default_value.c:15:3')}",
    ):
        signature_rules_module._infer_default_value_for_pyarg(
            _address_of("value", referenced=target_decl),
            RawType("int"),
        )

def test_infer_default_value_for_pyarg_uses_evaluated_float_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[_FakeNode] = []
    cursor = _float_literal("1e+06")
    target_decl = _var_decl("value", cursor)
    target_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.DOUBLE)

    monkeypatch.setattr(signature_rules_module, "is_nullptr_or_zero", lambda _: False)
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: observed.append(received_cursor) or 1000000.0,
    )

    assert (
        signature_rules_module._infer_default_value_for_pyarg(
            _address_of("value", referenced=target_decl),
            RawType("float"),
        )
        == "1000000.0"
    )
    assert observed == [cursor]

def test_infer_argument_lists_keeps_matching_pyarg_calls() -> None:
    first_decl = _var_decl("value")
    second_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=first_decl),
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=second_decl),
        ),
    )

    inferred = _infer_varargs_arguments(cursor)

    assert inferred == [
        [_arg("value", "int")],
        [_arg("value", "int")],
    ]
