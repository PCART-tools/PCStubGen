from __future__ import annotations

from typing import cast

from ..libclang import ast_utils
from clang.cindex import Cursor, CursorKind, StorageClass, TypeKind
from loguru import logger

from ....models import Argument, ArgumentKind, Signature
from ....type_models import AnyType, RawType, Type, UnionType
from ..libclang.ast_utils import (
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
from ..libclang.libclang_wrap import (
    CX_BINARY_OPERATOR_ASSIGN,
    evaluate_cursor,
    get_cursor_binary_operator_kind,
)
from ..method_flags import (
    METH_FASTCALL,
    METH_CLASS,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
)
from .py_arg_parse.tuple_and_keywords_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
)
from .py_arg_parse.tuple_parser import PyArgParseTupleTypeParser
from .py_build_value.parser import PyBuildValueTypeParser
from .rules import (
    CALL_NAME_TO_TYPE,
    OBJECT_NAME_TO_TYPE,
    PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
    PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
)

_PYARG_PARSETUPLE_CALL_NAMES = {
    "PyArg_ParseTuple",
    "_PyArg_ParseTuple_SizeT",
}

_PYARG_PARSETUPLE_AND_KEYWORDS_CALL_NAMES = {
    "PyArg_ParseTupleAndKeywords",
    "_PyArg_ParseTupleAndKeywords_SizeT",
}

_PYTHON_SINGLETON_DEFAULT_NAME_TO_VALUE = {
    "_Py_NoneStruct": "None",
    "_Py_TrueStruct": "True",
    "_Py_FalseStruct": "False",
}
_BOOL_TYPE = RawType("bool")
_FLOAT_TYPE = RawType("float")


class Inferencer:
    def __init__(self, func_cursor: Cursor, flags: int, is_method: bool):
        """保存当前待推断函数的上下文。"""
        self._func_cursor = func_cursor
        self._flags = flags
        self._is_method = is_method

    def run(self) -> list[Signature]:
        """汇合参数推断与返回值推断结果，直接生成签名。"""
        arguments_list = self._infer_arguments_list()
        return_type = self._infer_return_type()

        if arguments_list:
            return [
                Signature(
                    args=arguments,
                    return_type=return_type,
                )
                for arguments in arguments_list
            ]

        minimal_signatures = self._infer_minimal_signatures(return_type)
        if minimal_signatures:
            return minimal_signatures
        return [Signature(return_type=return_type)]

    def _infer_minimal_signatures(self, return_type: Type) -> list[Signature]:
        """根据来自 `PyMethodDef.ml_flags` 的 flags 值推断最小签名。"""
        argument_lists = self._infer_argument_lists_from_flags()
        if not argument_lists:
            return []
        return [
            Signature(
                args=arguments,
                return_type=return_type,
            )
            for arguments in argument_lists
        ]

    def _infer_argument_lists_from_flags(self) -> list[list[Argument]]:
        """根据来自 `PyMethodDef.ml_flags` 的 flags 值推断最小参数形状。"""
        if self._flags & METH_NOARGS:
            return [[]]

        if self._flags & METH_O:
            return [[
                Argument(
                    name="arg",
                    type=RawType("object"),
                    kind=ArgumentKind.POSITIONAL_ONLY,
                )
            ]]

        if self._flags & (METH_VARARGS | METH_FASTCALL):
            arguments = [
                Argument(
                    name="args",
                    type=RawType("object"),
                    kind=ArgumentKind.VAR_POSITIONAL,
                )
            ]
            if self._flags & METH_KEYWORDS:
                arguments.append(
                    Argument(
                        name="kwargs",
                        type=RawType("object"),
                        kind=ArgumentKind.VAR_KEYWORD,
                    )
                )
            return [arguments]

        return []

    # def try_add_bound(args: list[Argument], is_method: bool, flags: int) -> bool:
    #     if is_method:
    #         name = "self"
    #         if flags & METH_CLASS:
    #             name
    #         bound = Argument()

    def _infer_arguments_list(self) -> list[list[Argument]]:
        """遍历函数体内支持的 `PyArg_*` 调用并收集参数列表。"""
        if self._flags & METH_NOARGS:
            return [[]]

        arguments_list: list[list[Argument]] = []

        for call_expr in walk(self._func_cursor):
            if call_expr.kind != CursorKind.CALL_EXPR:
                continue

            call_name = ast_utils.get_first_token_str(call_expr)
            try:
                if call_name in "PyArg_ParseTuple":
                    arguments_list.append(self._infer_pyarg_parse_tuple_arguments(call_expr))
                elif call_name in "PyArg_ParseTupleAndKeywords":
                    arguments_list.append(self._infer_pyarg_parse_tuple_and_keywords_arguments(call_expr))
            except Exception as ex:
                logger.warning(
                    "跳过无法推断的 PyArg 参数列表, func_name: {}, call_name: {}, reason: {!r}",
                    self._func_cursor.spelling,
                    call_name,
                    ex,
                )

        if len(arguments_list) > 1:
            logger.warning("多个参数列表, func_name: {}", self._func_cursor.spelling)

        return arguments_list

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
            infer_type_object_func=self._infer_type_object_type_for_pyarg,
            infer_converter_type_func=self._infer_converter_type_for_pyarg,
            infer_default_value_func=lambda cursor, expected_type: self._infer_default_value_for_pyarg(
                cursor,
                expected_type,
            ),
        ).parse()

    def _infer_expr_type(self, expr: Cursor) -> Type:
        """对单个表达式做 Python 类型推断。"""
        expr = unwrap_transparent(expr)

        if expr.kind == CursorKind.CONDITIONAL_OPERATOR:
            return self._infer_conditional_operator_type(expr)

        if expr.kind == CursorKind.CALL_EXPR:
            return self._infer_call_expr_type(expr)

        if expr.kind == CursorKind.DECL_REF_EXPR:
            return self._infer_decl_ref_expr_type(expr)

        if expr.kind == CursorKind.UNARY_OPERATOR:
            child = ast_utils.unwrap_addr_of(expr)
            if child.kind == CursorKind.DECL_REF_EXPR:
                return self._infer_decl_ref_expr_type(child)

        if is_nullptr_or_zero(expr):
            """return NULL异常返回分支 union后就不存在了"""
            return UnionType(())

        raise RuntimeError(f"不支持的表达式类型: {expr.kind.name}, cursor: {expr.location}")

    def _infer_conditional_operator_type(self, op_cursor: Cursor) -> Type:
        """推断标准三元表达式 `cond ? a : b` 的结果类型。"""
        assert op_cursor.kind == CursorKind.CONDITIONAL_OPERATOR
        children = list(op_cursor.get_children())

        branch_types: list[Type] = []
        for branch in children[1:]:
            try:
                branch_types.append(self._infer_expr_type(branch))
            except Exception as ex:
                logger.warning(
                    "跳过无法推断的条件分支表达式, reason: {!r}",
                    ex,
                )
        return UnionType(tuple(branch_types))

    def _infer_decl_ref_expr_type(self, expr_cursor: Cursor) -> Type:
        """识别 `DECL_REF_EXPR` 形式的直接对象类型。"""
        assert expr_cursor.kind == CursorKind.DECL_REF_EXPR

        identifier_name = expr_cursor.spelling
        mapped = OBJECT_NAME_TO_TYPE.get(identifier_name)
        if mapped is not None:
            return mapped
        try:
            return self._infer_local_decl_ref_expr_type(expr_cursor)
        except RuntimeError as ex:
            raise RuntimeError(
                f"无法识别的对象返回标识符: {identifier_name}, cursor: {expr_cursor.location}"
            ) from ex

    def _infer_local_decl_ref_expr_type(self, expr_cursor: Cursor) -> Type:
        """从函数内局部变量的定值表达式中推断 `DECL_REF_EXPR` 类型。"""
        target_decl = expr_cursor.referenced
        if target_decl is None or target_decl.kind != CursorKind.VAR_DECL:
            raise RuntimeError(f"引用节点未指向局部变量声明, cursor: {expr_cursor.location}")
        if target_decl.storage_class == StorageClass.STATIC:
            raise RuntimeError(f"不追溯 static 局部变量, cursor: {target_decl.location}")

        function_cursor = self._find_local_decl_function_parent(target_decl)
        candidate_types: list[Type] = []
        for candidate_expr in self._iter_local_decl_assignment_exprs(function_cursor, target_decl):
            candidate_expr = self._unwrap_assignment_chain_value(candidate_expr)
            if is_nullptr_or_zero(candidate_expr):
                continue
            candidate_types.append(self._infer_expr_type(candidate_expr).canonicalize())

        if not candidate_types:
            raise RuntimeError(f"局部变量没有可用定值表达式: {target_decl.spelling}")

        inferred_type = candidate_types[0]
        for candidate_type in candidate_types[1:]:
            if candidate_type != inferred_type:
                raise RuntimeError(
                    f"局部变量定值表达式类型不收敛: {target_decl.spelling}, "
                    f"left: {inferred_type.render()}, right: {candidate_type.render()}"
                )
        return inferred_type

    def _unwrap_assignment_chain_value(self, expr_cursor: Cursor) -> Cursor:
        """剥离链式赋值表达式，定位到最终右值。"""
        value_expr = unwrap_transparent(expr_cursor)
        while value_expr.kind == CursorKind.BINARY_OPERATOR:
            if get_cursor_binary_operator_kind(value_expr) != CX_BINARY_OPERATOR_ASSIGN:
                break
            children = list(value_expr.get_children())
            assert len(children) == 2
            value_expr = unwrap_transparent(children[1])
        return value_expr

    def _find_local_decl_function_parent(self, decl_cursor: Cursor) -> Cursor:
        """从声明节点的语义父节点中定位所在函数。"""
        parent = decl_cursor.semantic_parent
        while parent is not None:
            if parent.kind == CursorKind.FUNCTION_DECL:
                return parent
            parent = parent.semantic_parent
        raise RuntimeError(
            f"局部变量声明不在函数内: {decl_cursor.spelling}, cursor: {decl_cursor.location}"
        )

    def _iter_local_decl_assignment_exprs(
        self,
        function_cursor: Cursor,
        target_decl: Cursor,
    ) -> list[Cursor]:
        """收集函数内目标局部变量的声明初始化和直接赋值右值表达式。"""
        candidates: list[Cursor] = []
        initializer = self._extract_optional_decl_initializer(target_decl)
        if initializer is not None:
            candidates.append(initializer)

        for cursor in walk(function_cursor):
            if cursor.kind != CursorKind.BINARY_OPERATOR:
                continue
            assignment_value = self._extract_direct_assignment_value(cursor, target_decl)
            if assignment_value is not None:
                candidates.append(assignment_value)
        return candidates

    def _extract_optional_decl_initializer(self, decl_cursor: Cursor) -> Cursor | None:
        """提取声明初始化表达式；无初始化式时返回 `None`。"""
        children = list(decl_cursor.get_children())
        if not children:
            return None

        initializer = unwrap_transparent(children[-1])
        if initializer.kind == CursorKind.TYPE_REF:
            return None
        return initializer

    def _extract_direct_assignment_value(
        self,
        assignment_cursor: Cursor,
        target_decl: Cursor,
    ) -> Cursor | None:
        """在 `x = expr` 中提取目标局部变量对应的右值表达式。"""
        if get_cursor_binary_operator_kind(assignment_cursor) != CX_BINARY_OPERATOR_ASSIGN:
            return None

        children = list(assignment_cursor.get_children())
        if len(children) != 2:
            return None

        target_expr = unwrap_transparent(children[0])
        if not self._is_decl_ref_to_decl(target_expr, target_decl):
            return None
        return children[1]

    def _is_decl_ref_to_decl(self, expr_cursor: Cursor, target_decl: Cursor) -> bool:
        """判断表达式是否直接引用目标声明节点。"""
        expr_cursor = unwrap_transparent(expr_cursor)
        if expr_cursor.kind != CursorKind.DECL_REF_EXPR:
            return False

        referenced = expr_cursor.referenced
        if referenced is None:
            return False
        return self._is_same_decl(referenced, target_decl)

    def _is_same_decl(self, left_decl: Cursor, right_decl: Cursor) -> bool:
        """判断两个声明节点是否指向同一个 C 声明。"""
        if left_decl == right_decl:
            return True

        left_usr = left_decl.get_usr()
        right_usr = right_decl.get_usr()
        return bool(left_usr and right_usr and left_usr == right_usr)

    def _infer_call_expr_type(self, cursor: Cursor) -> Type:
        """
        从调用表达式推断返回类型。
        调用名按源码调用表达式起点 token 提取，避免函数式宏的 callee source range
        扩成整段调用文本。
        """
        assert cursor.kind == CursorKind.CALL_EXPR
        call_name = get_first_token_str(cursor)

        if call_name == "Py_BuildValue":
            return self._infer_py_build_value_type(cursor)
        if call_name == "PyObject_New":
            return self._infer_pyobject_new_type(cursor)
        mapped = CALL_NAME_TO_TYPE.get(call_name)
        if mapped is None:
            raise RuntimeError(
                f"无法识别的返回值工厂调用: {call_name}, cursor: {cursor.location}"
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
        if not args:
            raise RuntimeError(
                f"PyObject_New 缺少类型对象参数, cursor: {call_cursor.location}"
            )
        return self._infer_type_object_type_for_pyarg(args[-1])

    def _infer_argument_name(self, c_args: list[Cursor]) -> str:
        """将 parser 提供的槽位变量名按首次出现顺序拼接为参数名。"""
        names: list[str] = []
        seen_names: set[str] = set()
        for arg in c_args:
            arg = ast_utils.unwrap_addr_of(arg)
            if arg.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
                arg = unwrap_transparent(list(arg.get_children())[0])
            if arg.spelling in seen_names:
                continue
            names.append(arg.spelling)
            seen_names.add(arg.spelling)

        return "_".join(names)

    def _infer_type_object_type_for_pyarg(self, cursor: Cursor) -> Type:
        """解析 `PyArg_*` 中 `O!` 类型对象槽位对应的 Python 类型名。"""
        source_text = get_cursor_source_text(cursor)
        match = IDENTIFIER_RE.search(source_text)
        if match is None:
            raise RuntimeError(
                f"类型对象槽位源码中未找到标识符, source_text: {source_text!r}, cursor: {cursor.location}"
            )
        type_name = match.group(0)
        mapped = PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE.get(type_name)
        if mapped is None:
            raise RuntimeError(
                f"无法识别的类型对象标识符: {type_name}, source_text: {source_text!r}, cursor: {cursor.location}"
            )
        return mapped

    def _infer_converter_type_for_pyarg(self, cursor: Cursor) -> Type:
        """解析 `PyArg_*` 中 `O&` converter 槽位对应的 Python 类型名。"""
        source_text = get_cursor_source_text(cursor)
        match = IDENTIFIER_RE.search(source_text)
        if match is None:
            raise RuntimeError(
                f"converter 槽位源码中未找到标识符, source_text: {source_text!r}, cursor: {cursor.location}"
            )
        converter_name = match.group(0)
        mapped = PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE.get(converter_name)
        if mapped is None:
            raise RuntimeError(
                f"无法识别的 converter 标识符: {converter_name}, source_text: {source_text!r}, cursor: {cursor.location}"
            )
        return mapped

    def _infer_default_value_for_pyarg(
        self,
        cursor: Cursor,
        expected_type: Type,
    ) -> str:
        """从参数接收槽位的 reaching definition 解析默认值文本。"""
        cursor = ast_utils.unwrap_addr_of(cursor)
        if cursor.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            array_decl, _ = ast_utils.extract_array_subscript(cursor)
            expr = self._get_array_element_reaching_definition_cursor(cursor)
            return self._render_default_value_expr(expr, array_decl, expected_type)

        if cursor.kind == CursorKind.DECL_REF_EXPR:
            target_decl = cursor.referenced
            if target_decl is None or target_decl.kind not in DECL_CURSOR_KINDS:
                raise RuntimeError(
                    f"引用节点未指向声明节点, cursor: {ast_utils.to_str(cursor)}"
                )
            expr = self._get_decl_ref_expr_reaching_definition_cursor(cursor)
            return self._render_default_value_expr(expr, target_decl, expected_type)

        raise RuntimeError(f"infer_default_value_for_pyarg，不支持的cursor类型, cursor: {ast_utils.to_str(cursor)}")

    def _render_default_value_expr(
        self,
        expr: Cursor | int,
        target_decl: Cursor,
        expected_type: Type,
    ) -> str:
        """将 C 默认值表达式渲染为 Python 字面量。"""
        if type(expr) is int:
            value = cast(int, expr)
            if target_decl.type.get_canonical().kind == TypeKind.POINTER and value == 0:
                return "..."
            if expected_type == _BOOL_TYPE:
                if value == 0:
                    return "False"
                if value == 1:
                    return "True"
                raise RuntimeError(f"bool 默认值整数不是 0 或 1: {value!r}")
            return self._render_number_default(value, expected_type)

        expr = unwrap_transparent(cast(Cursor, expr))
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
            if expected_type == _BOOL_TYPE:
                if evaluated == 0:
                    return "False"
                if evaluated == 1:
                    return "True"
                raise RuntimeError(
                    f"C++ bool 字面量求值结果不是 0 或 1: {evaluated!r}, cursor: {ast_utils.to_str(expr)}"
                )
            return self._render_number_default(cast(int, evaluated), expected_type)

        if (
            expr.kind == CursorKind.INTEGER_LITERAL
            and target_decl.type.get_canonical().kind != TypeKind.POINTER
        ):
            evaluated = evaluate_cursor(expr)
            if type(evaluated) is not int:
                raise RuntimeError(
                    f"整数默认值求值结果不是整数: {evaluated!r}, cursor: {ast_utils.to_str(expr)}"
                )
            if expected_type == _BOOL_TYPE:
                if evaluated == 0:
                    return "False"
                if evaluated == 1:
                    return "True"
                raise RuntimeError(
                    f"bool 默认值整数不是 0 或 1: {evaluated!r}, cursor: {ast_utils.to_str(expr)}"
                )
            return self._render_number_default(cast(int, evaluated), expected_type)

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
        if type(evaluated) in (int, float):
            return cast(int | float, evaluated)
        raise RuntimeError(f"数字默认值求值结果不是数字: {evaluated!r}, cursor: {ast_utils.to_str(expr)}")

    def _render_number_default(self, value: int | float, expected_type: Type) -> str:
        """将已求值的 C 数字默认值渲染为 Python 字面量。"""
        if expected_type == _FLOAT_TYPE:
            return str(float(value))
        return str(value)

    def _find_decl(self, cursor: Cursor) -> Cursor:
        """将实参槽位解析为被写入的目标声明节点。"""
        if cursor.kind in DECL_CURSOR_KINDS:
            return cursor

        if cursor.kind == CursorKind.DECL_REF_EXPR:
            referenced = cursor.referenced
            if referenced is not None and referenced.kind in DECL_CURSOR_KINDS:
                return referenced
        raise RuntimeError(f"无法将 C 参数槽位解析为声明节点, cursor: {cursor.location}")

    def _get_array_element_reaching_definition_cursor(
        self,
        array_subscript_expr: Cursor,
    ) -> Cursor | int:
        """查找数组元素在目标槽位引用前的最后一个定值表达式。"""
        assert array_subscript_expr.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR
        array_decl, index = ast_utils.extract_array_subscript(array_subscript_expr)
        ret = self._extract_array_initializer_value(array_decl, index)

        for cursor in walk(self._func_cursor):
            if cursor == array_subscript_expr:
                break
            if cursor.kind != CursorKind.BINARY_OPERATOR:
                continue
            if get_cursor_binary_operator_kind(cursor) != CX_BINARY_OPERATOR_ASSIGN:
                continue
            assignment_value = self._extract_array_element_assignment_value(cursor, array_decl, index)
            if assignment_value is not None:
                ret = assignment_value

        if ret is None:
            raise RuntimeError(
                f"数组元素没有可用定值表达式: {array_decl.spelling}[{index}]"
            )
        return ret

    def _get_decl_ref_expr_reaching_definition_cursor(self, decl_ref_expr: Cursor) -> Cursor:
        """查找目标声明在参数槽位引用前的最后一个定值表达式。"""
        target_decl = decl_ref_expr.referenced
        if target_decl is None or target_decl.kind not in DECL_CURSOR_KINDS:
            raise RuntimeError(f"引用节点未指向声明节点, cursor: {ast_utils.to_str(decl_ref_expr)}")
        ret = self._extract_optional_decl_initializer(target_decl)

        for cursor in walk(self._func_cursor):
            if cursor == decl_ref_expr:
                break
            if cursor.kind != CursorKind.BINARY_OPERATOR:
                continue
            if get_cursor_binary_operator_kind(cursor) != CX_BINARY_OPERATOR_ASSIGN:
                continue

            children = list(cursor.get_children())
            if len(children) != 2:
                continue

            target_expr = unwrap_transparent(children[0])
            if target_expr.kind != CursorKind.DECL_REF_EXPR:
                continue
            if target_expr.referenced != target_decl:
                continue
            ret = self._unwrap_assignment_chain_value(children[1])

        if ret is None:
            raise RuntimeError(f"声明节点没有可用定值表达式: {target_decl.spelling}")
        return ret

    def _extract_array_initializer_value(self, array_decl: Cursor, index: int) -> Cursor | int | None:
        """提取数组声明初始化中指定下标的初始值。"""
        initializer = self._extract_optional_decl_initializer(array_decl)
        if initializer is None:
            return None
        if initializer.kind != CursorKind.INIT_LIST_EXPR:
            raise RuntimeError(
                f"数组声明初始化仅支持顺序初始化列表: {array_decl.spelling}, cursor: {array_decl.location}"
            )

        children = list(initializer.get_children())
        for child in children:
            if self._is_designated_initializer(child):
                raise RuntimeError(
                    f"数组声明初始化不支持指定初始化: {array_decl.spelling}, cursor: {array_decl.location}"
                )

        if index < len(children):
            return unwrap_transparent(children[index])
        return 0

    def _is_designated_initializer(self, cursor: Cursor) -> bool:
        """判断初始化列表项是否为指定初始化。"""
        tokens = list(cursor.get_tokens())
        if not tokens:
            return False
        return tokens[0].spelling in ("[", ".")

    def _extract_array_element_assignment_value(
        self,
        assignment_cursor: Cursor,
        array_decl: Cursor,
        index: int,
    ) -> Cursor | None:
        """在 `array[index] = expr` 中提取指定数组元素对应的最终右值。"""
        children = list(assignment_cursor.get_children())
        if len(children) != 2:
            return None

        target_expr = unwrap_transparent(children[0])
        if target_expr.kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
            return None

        candidate_decl, candidate_index = ast_utils.extract_array_subscript(target_expr)
        if candidate_decl != array_decl:
            return None
        if candidate_index != index:
            return None
        return self._unwrap_assignment_chain_value(children[1])

    def _extract_kwlist(self, node: Cursor) -> list[str]:
        """解析 `PyArg_ParseTupleAndKeywords` 的静态关键字名数组。"""
        kwlist_decl = self._find_decl(node)
        if kwlist_decl.kind != CursorKind.VAR_DECL:
            raise RuntimeError(f"kwlist 必须引用 VAR_DECL, cursor: {node.location}")

        init_list_expr = var_decl_to_init_list_expr(kwlist_decl)

        result: list[str] = []
        for child in init_list_expr.get_children():
            entry = unwrap_transparent(child)
            if is_nullptr_or_zero(entry):
                break

            keyword_name = get_string_literal(entry)
            result.append(keyword_name)

        return result

    def _extract_decl_initializer(self, decl_cursor: Cursor) -> Cursor:
        """提取声明节点的初始化表达式。"""
        children = list(decl_cursor.get_children())
        if not children:
            raise RuntimeError(
                f"声明节点缺少初始化表达式: {decl_cursor.spelling}, cursor: {decl_cursor.location}"
            )
        return unwrap_transparent(children[-1])
