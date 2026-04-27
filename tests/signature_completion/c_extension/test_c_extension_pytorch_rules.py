from __future__ import annotations

import clang.cindex
import pytest

from pcstubgen.models import ArgumentKind
from pcstubgen.signature_completion.c_extension.method_flags import METH_KEYWORDS, METH_VARARGS
from pcstubgen.signature_completion.c_extension import inferencer as signature_rules_module
from pcstubgen.signature_completion.c_extension.signatures.rules import pytorch_rules
from pcstubgen.type_models import RawType, UnionType
from tests._c_extension_test_support import (
    _FakeNode,
    _arg,
    _fake_function_cursor_with_children,
    _identifier_node,
    _init_list,
    _string_literal,
    _var_decl,
    patch_inference_clang_helpers,
)


@pytest.fixture(autouse=True)
def _patch_fake_clang_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_inference_clang_helpers(monkeypatch, signature_rules_module)


class _FakeType:
    def __init__(self, spelling: str) -> None:
        self.spelling = spelling


def _python_arg_parser_decl(*entries: _FakeNode) -> _FakeNode:
    """构造 PyTorch `PythonArgParser` 变量声明。"""
    parser_decl = _var_decl("parser", _init_list(*entries))
    parser_decl.type = _FakeType("torch::PythonArgParser")
    parser_decl._children.insert(
        0,
        _FakeNode(
            kind=clang.cindex.CursorKind.TYPE_REF,
            spelling="struct torch::PythonArgParser",
        ),
    )
    return parser_decl


def _union_type(*members: RawType) -> UnionType:
    """构造与生产逻辑一致的规范化 union 类型。"""
    canonical = UnionType(tuple(members)).canonicalize()
    assert isinstance(canonical, UnionType)
    return canonical


def test_parse_python_arg_parser_signature_maps_torch_types() -> None:
    arguments = pytorch_rules.parse_python_arg_parser_signature(
        "short(Tensor input, *, MemoryFormat? memory_format=None)"
    )

    assert arguments == [
        _arg("input", "torch.Tensor", imports=("torch",)),
        _arg(
            "memory_format",
            _union_type(RawType("torch.memory_format", imports=("torch",)), RawType.none_),
            default_value="None",
            kind=ArgumentKind.KEYWORD_ONLY,
        ),
    ]


def test_parse_python_arg_parser_signature_uses_torch_size_aliases() -> None:
    arguments = pytorch_rules.parse_python_arg_parser_signature(
        "new(IntArrayRef size, SymIntArrayRef? sym_size=None, TensorList tensors)"
    )

    assert arguments == [
        _arg("size", "torch.types._size", imports=("torch.types",)),
        _arg(
            "sym_size",
            _union_type(RawType("torch.types._symsize", imports=("torch.types",)), RawType.none_),
            default_value="None",
        ),
        _arg(
            "tensors",
            "collections.abc.Sequence[torch.Tensor]",
            imports=("collections.abc", "torch"),
        ),
    ]


def test_parse_python_arg_parser_signature_keeps_unknown_type_text() -> None:
    arguments = pytorch_rules.parse_python_arg_parser_signature("demo(CustomType value)")

    assert arguments == [_arg("value", "CustomType")]


def test_parse_python_arg_parser_signature_skips_hidden_and_deprecated() -> None:
    assert pytorch_rules.parse_python_arg_parser_signature("demo(int64_t value)|hidden") is None
    assert pytorch_rules.parse_python_arg_parser_signature("demo(int64_t value)|deprecated") is None


def test_parse_python_arg_parser_signature_rejects_unknown_syntax() -> None:
    with pytest.raises(pytorch_rules.PyTorchSignatureParserError):
        pytorch_rules.parse_python_arg_parser_signature("demo(int64_t)")


def test_infer_arguments_list_uses_python_arg_parser_overloads() -> None:
    cursor = _fake_function_cursor_with_children(
        _python_arg_parser_decl(
            _string_literal("device(Device device)"),
            _string_literal("device(std::string_view type, int64_t? index=-1)"),
        )
    )

    inferred = signature_rules_module.infer_arguments_list(
        cursor,
        flags=METH_VARARGS | METH_KEYWORDS,
    )

    assert inferred == [
        [_arg("device", "torch.device", imports=("torch",))],
        [
            _arg("type", "str"),
            _arg(
                "index",
                _union_type(RawType.int_, RawType.none_),
                default_value="-1",
            ),
        ],
    ]


def test_infer_arguments_list_does_not_follow_python_arg_parser_variable() -> None:
    cursor = _fake_function_cursor_with_children(
        _python_arg_parser_decl(_identifier_node("signature")),
    )

    inferred = signature_rules_module.infer_arguments_list(
        cursor,
        flags=METH_VARARGS | METH_KEYWORDS,
    )

    assert inferred == [
        [
            _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
            _arg("kwargs", "object", kind=ArgumentKind.VAR_KEYWORD),
        ]
    ]


def test_infer_arguments_list_uses_libclang_merged_macro_string() -> None:
    cursor = _fake_function_cursor_with_children(
        _python_arg_parser_decl(
            _string_literal("torch.UntypedStorage(*, int64_t allocator=None)")
        )
    )

    inferred = signature_rules_module.infer_arguments_list(
        cursor,
        flags=METH_VARARGS | METH_KEYWORDS,
    )

    assert inferred == [
        [
            _arg(
                "allocator",
                _union_type(RawType.int_, RawType.none_),
                default_value="None",
                kind=ArgumentKind.KEYWORD_ONLY,
            )
        ]
    ]
