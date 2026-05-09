from __future__ import annotations

from collections.abc import Callable

from clang.cindex import Cursor, CursorKind, StorageClass, TypeKind
from loguru import logger

from ...models import Argument, ArgumentKind, Signature
from ...type_models import AnyType, RawType, Type, UnionType
from .libclang import ast_utils
from .libclang.ast_utils import (
    DECL_CURSOR_KINDS,
    IDENTIFIER_RE,
    get_first_token_str,
    get_cursor_source_text,
    get_string_literal,
    is_nullptr_or_zero,
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk,
)
from .libclang.libclang_wrap import (
    CX_BINARY_OPERATOR_ASSIGN,
    evaluate_cursor,
    get_cursor_binary_operator_kind,
)
from .method_flags import (
    METH_FASTCALL,
    METH_CLASS,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_STATIC,
    METH_VARARGS,
)
from .signatures.py_arg_parse.tuple_and_keywords_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
)
from .signatures.py_arg_parse.tuple_parser import PyArgParseTupleTypeParser
from .signatures.py_build_value.parser import PyBuildValueTypeParser
from .signatures.rules import numpy_rules, pytorch_rules
from .signatures.rules import (
    CALL_NAME_TO_TYPE,
    OBJECT_USE_FUNCTION_NAME_TO_TYPE,
    OBJECT_NAME_TO_TYPE,
    PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
    PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
    get_name_to_type,
)

_PYTHON_SINGLETON_DEFAULT_NAME_TO_VALUE = {
    "_Py_NoneStruct": "None",
    "_Py_TrueStruct": "True",
    "_Py_FalseStruct": "False",
}


class Inferencer:
    def __init__(self, func_cursor: Cursor, flags: int, owner_class: type | None):
        """保存当前待推断函数的上下文。"""
        self._func_cursor = func_cursor
        self._flags = flags
        self._owner_class = owner_class
        self._param_cursors = [
            child
            for child in self._func_cursor.get_children()
            if child.kind == CursorKind.PARM_DECL
        ]

    def run(self) -> list[Signature]:
        """汇合参数推断与返回值推断结果，直接生成签名。"""
        return_type = self._infer_return_type()
        arguments_list = self._infer_arguments_list()
        for args in arguments_list:
            self._try_add_receiver(args)

        return [
            Signature(
                args=arguments,
                return_type=return_type,
            )
            for arguments in arguments_list
        ]

    def _infer_arguments_list(self) -> list[list[Argument]]:
        """按 flags 决定业务参数骨架，并在允许时读取 `PyArg_*` 细化。"""

        if self._flags & METH_NOARGS:
            return [[]]

        if self._flags & METH_O:
            return [[self._build_meth_o_argument()]]

        if self._flags & METH_FASTCALL:
            arguments_list = self._infer_arguments_for_call_name(
                "npy_parse_arguments",
                self._infer_npy_parse_arguments,
            )
            if arguments_list:
                return arguments_list
            return [self._build_minimal_arguments()]

        if self._flags & METH_VARARGS and self._flags & METH_KEYWORDS:
            arguments_list = self._infer_arguments_for_call_name(
                "PyArg_ParseTuple",
                self._infer_pyarg_parse_tuple_arguments,
            )
            arguments_list.extend(
                self._infer_arguments_for_call_name(
                    "PyArg_ParseTupleAndKeywords",
                    self._infer_pyarg_parse_tuple_and_keywords_arguments,
                )
            )
            arguments_list.extend(pytorch_rules.infer_python_arg_parser_arguments(self._func_cursor))
            if arguments_list:
                return arguments_list
            return [self._build_minimal_arguments()]

        if self._flags & METH_VARARGS and self._flags & METH_KEYWORDS == 0:
            arguments_list = self._infer_arguments_for_call_name(
                "PyArg_ParseTuple",
                self._infer_pyarg_parse_tuple_arguments,
            )
            arguments_list.extend(pytorch_rules.infer_python_arg_parser_arguments(self._func_cursor))
            if arguments_list:
                return arguments_list
            return [self._build_minimal_arguments()]

        logger.error("不应该到达此处, func: {}", ast_utils.to_str(self._func_cursor))
        return [[]]

    def _build_minimal_arguments(self) -> list[Argument]:
        """构造 variadic 调用约定的最小业务参数骨架。"""
        arguments = [
            Argument(
                name="args",
                type=RawType.object_,
                kind=ArgumentKind.VAR_POSITIONAL,
            )
        ]
        if self._flags & METH_KEYWORDS:
            arguments.append(
                Argument(
                    name="kwargs",
                    type=RawType.object_,
                    kind=ArgumentKind.VAR_KEYWORD,
                )
            )
        return arguments

    def _infer_arguments_for_call_name(
        self,
        expected_call_name: str,
        parser: Callable[[Cursor], list[Argument]],
    ) -> list[list[Argument]]:
        """只按指定 `PyArg_*` 入口扫描参数解析调用。"""
        arguments_list: list[list[Argument]] = []

        for call_expr in walk(self._func_cursor):
            if call_expr.kind != CursorKind.CALL_EXPR:
                continue

            call_name = ast_utils.get_first_token_str(call_expr)
            spelling = call_expr.spelling
            if call_name != expected_call_name and spelling != expected_call_name:
                continue
            try:
                arguments_list.append(parser(call_expr))
            except Exception as ex:
                logger.warning(
                    "跳过无法推断的参数列表, func_name: {}, call_name: {}, reason: {!r}",
                    self._func_cursor.spelling,
                    call_name,
                    ex,
                )

        if len(arguments_list) > 1:
            logger.warning("多个参数列表, func_name: {}", self._func_cursor.spelling)

        return arguments_list

    def _try_add_receiver(self, arguments: list[Argument]) -> list[Argument]:
        """按方法绑定类型在参数列表头部原地插入 receiver。"""
        if self._owner_class is not None and self._flags & METH_STATIC == 0:
            if self._flags & METH_CLASS:
                arg_name = "cls"
            else:
                arg_name = "self"
            kind = ArgumentKind.POSITIONAL_OR_KEYWORD
            if arguments and arguments[0].kind is ArgumentKind.POSITIONAL_ONLY:
                kind = ArgumentKind.POSITIONAL_ONLY
            arguments.insert(0, Argument(name=arg_name, kind=kind))
        return arguments

    def _infer_return_type(self) -> Type:
        """遍历函数子树中的 return 语句并汇总返回类型。"""
        return_type_list: list[Type] = []

        for cursor in walk(self._func_cursor):
            if cursor.kind != CursorKind.RETURN_STMT:
                continue

            try:
                return_expr = list(cursor.get_children())[0]
                return_type_list.append(self._infer_expr_type(return_expr))
            except Exception as ex:
                logger.warning(
                    "跳过无法推断的 return 表达式, func_name: {}, reason: {!r}",
                    self._func_cursor.spelling,
                    ex,
                )

        return_type = UnionType(tuple(return_type_list)).canonicalize()
        if isinstance(return_type, UnionType) and len(return_type.members) == 0:
            return AnyType()
        return return_type

    def _infer_pyarg_parse_tuple_arguments(self, call_expr: Cursor) -> list[Argument]:
        """调用 `PyArg_ParseTuple` parser 解析参数列表。"""
        args = list(call_expr.get_children())[1:]
        format_string = get_string_literal(args[1])

        return PyArgParseTupleTypeParser(
            format_string,
            args[2:],
            infer_name_func=self._infer_argument_name,
            infer_type_object_func=self._infer_type_object_type_for_pyarg,
            infer_converter_type_func=self._infer_converter_type_for_pyarg,
            infer_refined_object_type_func=self._infer_refined_object_type_for_cursor,
            infer_default_value_func=lambda cursor, expected_type: self._infer_default_value_for_pyarg(
                cursor,
                expected_type,
            ),
        ).parse()

    def _infer_pyarg_parse_tuple_and_keywords_arguments(self, call_expr: Cursor) -> list[Argument]:
        """调用 `PyArg_ParseTupleAndKeywords` parser 解析参数列表。"""
        args = list(call_expr.get_children())[1:]
        format_string = get_string_literal(args[2])
        kwlist = self._extract_kwlist(args[3])

        return PyArgParseTupleAndKeywordsTypeParser(
            format_string,
            kwlist,
            args[4:],
            infer_name_func=self._infer_argument_name,
            infer_type_object_func=self._infer_type_object_type_for_pyarg,
            infer_converter_type_func=self._infer_converter_type_for_pyarg,
            infer_refined_object_type_func=self._infer_refined_object_type_for_cursor,
            infer_default_value_func=lambda cursor, expected_type: self._infer_default_value_for_pyarg(
                cursor,
                expected_type,
            ),
        ).parse()

    def _infer_npy_parse_arguments(self, call_expr: Cursor) -> list[Argument]:
        """调用 NumPy 专用 parser，解析 `npy_parse_arguments` 的 FASTCALL 参数列表。"""
        return numpy_rules.infer_npy_parse_arguments(
            call_expr,
            infer_name_func=self._infer_argument_name,
            infer_converter_type_func=self._infer_converter_type_for_pyarg,
            infer_refined_object_type_func=self._infer_refined_object_type_for_cursor,
            infer_default_value_func=self._infer_default_value_for_pyarg,
        )

    def _build_meth_o_argument(self) -> Argument:
        """构造 `METH_O` 的单个业务参数，并在需要时细化为具体类型。"""
        arg_type: Type = RawType.object_
        if len(self._param_cursors) >= 2:
            arg_type = self._infer_refined_object_type_for_cursor(self._param_cursors[1])
        return Argument(
            name="arg",
            type=arg_type,
            kind=ArgumentKind.POSITIONAL_ONLY,
        )

    def _infer_refined_object_type_for_cursor(self, cursor: Cursor) -> Type:
        """扫描函数体中的对象检查调用，细化 `object` 参数类型。"""
        target_decl = self._get_target_decl_for_cursor(cursor)
        if target_decl is None:
            return RawType.object_

        matched_types: set[Type] = set()
        for call_expr in walk(self._func_cursor):
            if call_expr.kind != CursorKind.CALL_EXPR:
                continue

            matched_type = self._infer_object_type_from_call(
                call_expr,
                target_decl,
            )
            if matched_type is None:
                continue
            matched_types.add(matched_type)

        if not matched_types:
            return RawType.object_
        return UnionType(tuple(matched_types)).canonicalize()

    def _infer_object_type_from_call(
        self,
        call_expr: Cursor,
        target_decl: Cursor,
    ) -> Type | None:
        """根据单个调用表达式，提取可用于对象细化的类型证据。"""
        call_name = get_first_token_str(call_expr)

        refined_type = get_name_to_type(OBJECT_USE_FUNCTION_NAME_TO_TYPE, call_name, call_expr)
        if refined_type is None:
            return None

        refined_decl = self._get_refined_decl_for_call(call_expr)
        if refined_decl is None:
            return None
        if refined_decl != target_decl:
            return None
        return refined_type

    def _get_target_decl_for_cursor(self, cursor: Cursor) -> Cursor | None:
        """把参数槽位或形参 cursor 规约为目标声明节点。"""
        cursor = ast_utils.unwrap_single_unary_op(cursor)
        if cursor.kind in DECL_CURSOR_KINDS:
            return cursor
        if cursor.kind != CursorKind.DECL_REF_EXPR:
            return None
        return cursor.referenced

    def _get_refined_decl_for_call(self, call_expr: Cursor) -> Cursor | None:
        """提取对象细化函数调用作用到的目标声明节点。"""
        children = list(call_expr.get_children())
        if len(children) < 2:
            return None

        refined_cursor = self._unwrap_refined_object_cursor(children[1])
        if refined_cursor.kind != CursorKind.DECL_REF_EXPR:
            return None
        return refined_cursor.referenced

    def _unwrap_refined_object_cursor(self, cursor: Cursor) -> Cursor:
        """剥离一层 `Py_TYPE(x)`，并返回真正参与对象细化的表达式。"""
        cursor = unwrap_transparent(cursor)
        if cursor.kind == CursorKind.CALL_EXPR and cursor.spelling == "Py_TYPE":
            children = list(cursor.get_children())
            if len(children) < 2:
                return cursor
            return unwrap_transparent(children[1])
        return cursor

    def _infer_expr_type(self, cursor: Cursor) -> Type:
        """对单个表达式做 Python 类型推断。"""
        cursor = unwrap_transparent(cursor)

        if cursor.kind == CursorKind.CONDITIONAL_OPERATOR:
            return self._infer_conditional_operator_type(cursor)

        if cursor.kind == CursorKind.CALL_EXPR:
            return self._infer_call_expr_type(cursor)

        if cursor.kind == CursorKind.DECL_REF_EXPR:
            return self._infer_decl_ref_expr_type(cursor)

        if cursor.kind == CursorKind.UNARY_OPERATOR:
            """可能是取地址符&"""
            child = ast_utils.unwrap_single_unary_op(cursor)
            if child.kind == CursorKind.DECL_REF_EXPR:
                return self._infer_decl_ref_expr_type(child)

        if is_nullptr_or_zero(cursor):
            """return NULL异常返回分支 union后就不存在了"""
            return UnionType(())

        raise RuntimeError(f"不支持的表达式类型: {cursor.kind.name}, cursor: {ast_utils.to_str(cursor)}")

    def _infer_conditional_operator_type(self, op_cursor: Cursor) -> Type:
        """推断标准三元表达式 `cond ? a : b` 的结果类型。"""
        assert op_cursor.kind == CursorKind.CONDITIONAL_OPERATOR
        children = list(op_cursor.get_children())

        branch_types: list[Type] = []
        for branch in children[1:]:
            try:
                branch_types.append(self._infer_expr_type(branch))
            except Exception as ex:
                logger.warning("跳过无法推断的条件分支表达式, reason: {!r}", ex)
        return UnionType(tuple(branch_types))

    def _infer_decl_ref_expr_type(self, cursor: Cursor) -> Type:
        """识别 `DECL_REF_EXPR` 形式的直接对象类型。"""
        assert cursor.kind == CursorKind.DECL_REF_EXPR

        if self._is_receiver_ref(cursor):
            return RawType.self_

        identifier_name = cursor.spelling
        mapped = get_name_to_type(OBJECT_NAME_TO_TYPE, identifier_name, cursor)
        if mapped is not None:
            return mapped
        try:
            return self._infer_decl_ref_expr_reaching_definition_type(cursor)
        except RuntimeError as ex:
            raise RuntimeError(
                f"无法识别的对象返回标识符: {identifier_name}, cursor: {ast_utils.to_str(cursor)}"
            ) from ex

    def _is_receiver_ref(self, cursor: Cursor) -> bool:
        """判断 `DECL_REF_EXPR` 是否引用实例方法 receiver。"""
        if self._owner_class is None:
            return False
        if self._flags & METH_STATIC or self._flags & METH_CLASS:
            return False
        return cursor.referenced == self._param_cursors[0]

    def _infer_decl_ref_expr_reaching_definition_type(self, cursor: Cursor) -> Type:
        """从函数内局部变量的定值表达式中推断 `DECL_REF_EXPR` 类型。"""
        ret = self._get_decl_ref_expr_reaching_definition_cursor(cursor)
        if is_nullptr_or_zero(ret):
            raise RuntimeError(f"局部变量没有可用定值表达式: {ast_utils.to_str(cursor)}")
        return self._infer_expr_type(ret).canonicalize()

    def _infer_call_expr_type(self, cursor: Cursor) -> Type:
        """
        从调用表达式推断返回类型。
        调用名按源码调用表达式起点 token 提取，避免函数式宏的 callee source range
        扩成整段调用文本。
        """
        assert cursor.kind == CursorKind.CALL_EXPR
        call_name = get_first_token_str(cursor)
        spelling = cursor.spelling

        if call_name == "Py_BuildValue" or spelling == "Py_BuildValue":
            return self._infer_py_build_value_type(cursor)
        if call_name == "PyObject_New" or spelling == "PyObject_New":
            return self._infer_pyobject_new_type(cursor)
        mapped = get_name_to_type(CALL_NAME_TO_TYPE, call_name, cursor)
        if mapped is None:
            raise RuntimeError(
                f"无法识别的返回值工厂调用: {call_name}, cursor: {ast_utils.to_str(cursor)}"
            )
        return mapped

    def _infer_py_build_value_type(self, call_cursor: Cursor) -> Type:
        """解析 `Py_BuildValue` 的格式串并返回 parser 推断结果。"""
        args = list(call_cursor.get_children())[1:]
        format_string = get_string_literal(args[0])

        return PyBuildValueTypeParser(
            format_string,
            args[1:],
            infer_object_type_func=self._infer_expr_type,
        ).parse()

    def _infer_pyobject_new_type(self, call_cursor: Cursor) -> Type:
        """从 `PyObject_New` 调用的类型对象参数推断 Python 类型。"""
        args = list(call_cursor.get_children())[1:]
        return self._infer_type_object_type_for_pyarg(args[-1])

    def _infer_argument_name(self, c_args: list[Cursor]) -> str:
        """将 parser 提供的槽位变量名按首次出现顺序拼接为参数名。"""
        names: list[str] = []
        seen_names: set[str] = set()
        for arg in c_args:
            arg = ast_utils.unwrap_single_unary_op(arg)
            if arg.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
                arg = unwrap_transparent(list(arg.get_children())[0])
            if arg.spelling in seen_names:
                continue
            names.append(arg.spelling)
            seen_names.add(arg.spelling)

        return "_".join(names)

    def _infer_type_object_type_for_pyarg(self, cursor: Cursor) -> Type:
        """
        解析 `PyArg_*` 中 `O!` 类型对象地址槽位对应的 Python 类型名。
        取源码是因为有宏，展开后可能是API[123]形式，失去名字
        """
        source_text = get_cursor_source_text(cursor).strip()
        type_name = source_text[1:].strip() if source_text.startswith("&") else source_text
        mapped = get_name_to_type(PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE, type_name, cursor)
        if mapped is None:
            raise RuntimeError(
                f"无法识别的类型对象标识符: {type_name}, source_text: {source_text!r}, cursor: {ast_utils.to_str(cursor)}"
            )
        return mapped

    def _infer_converter_type_for_pyarg(self, cursor: Cursor) -> Type:
        """解析 `PyArg_*` 中 `O&` converter 槽位对应的 Python 类型名。"""
        source_text = get_cursor_source_text(cursor)
        match = IDENTIFIER_RE.search(source_text)
        if match is None:
            raise RuntimeError(
                f"converter 槽位源码中未找到标识符, source_text: {source_text!r}, cursor: {ast_utils.to_str(cursor)}"
            )
        converter_name = match.group(0)
        mapped = get_name_to_type(PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE, converter_name, cursor)
        if mapped is None:
            raise RuntimeError(
                f"无法识别的 converter 标识符: {converter_name}, source_text: {source_text!r}, cursor: {ast_utils.to_str(cursor)}"
            )
        return mapped

    def _infer_default_value_for_pyarg(
        self,
        cursor: Cursor,
        expected_type: Type,
    ) -> str:
        """从参数接收槽位的 reaching definition 解析默认值文本。"""
        cursor = ast_utils.unwrap_single_unary_op(cursor)
        if cursor.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            array_decl, _ = ast_utils.extract_array_subscript(cursor)
            expr = self._get_array_subscript_expr_reaching_definition(cursor)
            return self._render_default_value_expr(expr, array_decl, expected_type)

        if cursor.kind == CursorKind.DECL_REF_EXPR:
            target_decl = cursor.referenced
            expr = self._get_decl_ref_expr_reaching_definition_cursor(cursor)
            return self._render_default_value_expr(expr, target_decl, expected_type)

        raise RuntimeError(f"infer_default_value_for_pyarg，不支持的cursor类型, cursor: {ast_utils.to_str(cursor)}")

    def _render_default_value_expr(
        self,
        expr: Cursor,
        target_decl: Cursor,
        expected_type: Type,
    ) -> str:
        """将 C 默认值表达式渲染为 Python 字面量。"""
        expr = unwrap_transparent(expr)
        expected_type = expected_type.canonicalize()

        if (
            target_decl.type.get_canonical().kind == TypeKind.POINTER
            and is_nullptr_or_zero(expr)
        ):
            return "..."

        if expr.kind == CursorKind.STRING_LITERAL:
            return repr(get_string_literal(expr))

        if expr.kind == CursorKind.FLOATING_LITERAL:
            return self._render_number_default(self._evaluate_number_cursor(expr), expected_type)

        if expr.kind == CursorKind.CXX_BOOL_LITERAL_EXPR:
            evaluated = evaluate_cursor(expr)
            if type(evaluated) is not int:
                raise RuntimeError(
                    f"C++ bool 字面量求值结果不是整数: {evaluated!r}, cursor: {ast_utils.to_str(expr)}"
                )
            if expected_type == RawType.bool_:
                if evaluated == 0:
                    return "False"
                if evaluated == 1:
                    return "True"
                raise RuntimeError(
                    f"C++ bool 字面量求值结果不是 0 或 1: {evaluated!r}, cursor: {ast_utils.to_str(expr)}"
                )
            return self._render_number_default(int(evaluated), expected_type)

        if (
            expr.kind == CursorKind.INTEGER_LITERAL
            and target_decl.type.get_canonical().kind != TypeKind.POINTER
        ):
            evaluated = evaluate_cursor(expr)
            if type(evaluated) is not int:
                raise RuntimeError(
                    f"整数默认值求值结果不是整数: {evaluated!r}, cursor: {ast_utils.to_str(expr)}"
                )
            if expected_type == RawType.bool_:
                if evaluated == 0:
                    return "False"
                if evaluated == 1:
                    return "True"
                raise RuntimeError(
                    f"bool 默认值整数不是 0 或 1: {evaluated!r}, cursor: {ast_utils.to_str(expr)}"
                )
            return self._render_number_default(int(evaluated), expected_type)

        if expr.kind == CursorKind.UNARY_OPERATOR:
            children = list(expr.get_children())
            assert len(children) == 1
            child = unwrap_transparent(children[0])
            if child.kind == CursorKind.DECL_REF_EXPR:
                rendered = _PYTHON_SINGLETON_DEFAULT_NAME_TO_VALUE.get(child.spelling)
                if rendered is not None:
                    return rendered
            if target_decl.type.get_canonical().kind != TypeKind.POINTER:
                return self._render_number_default(self._evaluate_number_cursor(expr), expected_type)

        raise RuntimeError(
            f"不支持的默认值表达式类型: {expr.kind.name}, cursor: {ast_utils.to_str(expr)}"
        )

    def _evaluate_number_cursor(self, expr: Cursor) -> int | float:
        """求值 C 数字表达式，并拒绝非数字求值结果。"""
        evaluated = evaluate_cursor(expr)
        if isinstance(evaluated, int):
            return evaluated
        if isinstance(evaluated, float):
            return evaluated
        raise RuntimeError(f"数字默认值求值结果不是数字: {evaluated!r}, cursor: {ast_utils.to_str(expr)}")

    def _render_number_default(self, value: int | float, expected_type: Type) -> str:
        """将已求值的 C 数字默认值渲染为 Python 字面量。"""
        if expected_type == RawType.float_:
            return str(float(value))
        return str(value)

    def _get_array_subscript_expr_reaching_definition(
        self,
        array_subscript_expr: Cursor,
    ) -> Cursor:
        """查找数组元素在目标槽位引用前的最后一个定值表达式。"""
        assert array_subscript_expr.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR
        array_decl, index = ast_utils.extract_array_subscript(array_subscript_expr)
        ret = None

        for cursor in walk(self._func_cursor):
            if cursor == array_subscript_expr:
                break
            if cursor.kind != CursorKind.BINARY_OPERATOR:
                continue
            if get_cursor_binary_operator_kind(cursor) != CX_BINARY_OPERATOR_ASSIGN:
                continue

            children = list(cursor.get_children())
            left = unwrap_transparent(children[0])
            if left.kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
                continue
            got_decl, got_index = ast_utils.extract_array_subscript(left)
            if got_decl != array_decl or got_index != index:
                continue

            right = unwrap_transparent(children[1])
            while (right.kind == CursorKind.BINARY_OPERATOR
                and get_cursor_binary_operator_kind(right) == CX_BINARY_OPERATOR_ASSIGN):
                right = list(right.get_children())[1]
                right = unwrap_transparent(right)
            if right.kind != CursorKind.BINARY_OPERATOR:
                ret = right

        if ret is None:
            raise RuntimeError(
                f"数组元素没有可用定值表达式: {array_decl.spelling}[{index}]"
            )
        return ret

    def _get_decl_ref_expr_reaching_definition_cursor(self, decl_ref_expr: Cursor) -> Cursor:
        """查找目标声明在参数槽位引用前的最后一个定值表达式。"""
        target_decl = decl_ref_expr.referenced
        ret = ast_utils.try_get_decl_initializer(target_decl)

        for cursor in walk(self._func_cursor):
            if cursor == decl_ref_expr:
                break
            if cursor.kind != CursorKind.BINARY_OPERATOR:
                continue
            if get_cursor_binary_operator_kind(cursor) != CX_BINARY_OPERATOR_ASSIGN:
                continue

            children = list(cursor.get_children())

            target_expr = unwrap_transparent(children[0])
            if target_expr.kind != CursorKind.DECL_REF_EXPR:
                continue
            if target_expr.referenced != target_decl:
                continue
            right = unwrap_transparent(children[1])
            while (right.kind == CursorKind.BINARY_OPERATOR
                and get_cursor_binary_operator_kind(right) == CX_BINARY_OPERATOR_ASSIGN):
                right = list(right.get_children())[1]
                right = unwrap_transparent(right)
            if right.kind != CursorKind.BINARY_OPERATOR:
                ret = right

        if ret is None:
            raise RuntimeError(f"声明节点没有可用定值表达式: {ast_utils.to_str(target_decl)}")
        return ret

    def _extract_kwlist(self, cursor: Cursor) -> list[str]:
        """解析 `PyArg_ParseTupleAndKeywords` 的静态关键字名数组。"""
        cursor = unwrap_transparent(cursor)
        assert cursor.kind == CursorKind.DECL_REF_EXPR
        kwlist_decl = cursor.referenced

        init_list_expr = var_decl_to_init_list_expr(kwlist_decl)

        result: list[str] = []
        for child in init_list_expr.get_children():
            entry = unwrap_transparent(child)
            if is_nullptr_or_zero(entry):
                break

            keyword_name = get_string_literal(entry)
            result.append(keyword_name)

        return result
