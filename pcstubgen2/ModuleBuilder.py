from __future__ import annotations

import ast
import inspect
import re
import types
from typing import Any

from .ErrorCollector import ErrorCollector
from .Errors import InvalidExpressionError
from .ReflectionHelpers import (
    get_generic_alias_type,
    get_module_name,
    get_doc,
    is_package,
)
from .IR import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    InvalidExpression,
    IRMethod,
    IRModule,
    QualifiedName,
    ResolvedType,
    IRValue,
)


class ModuleBuilder:
    def __init__(self, error_collector: ErrorCollector):
        self.error_collector = error_collector

    def build_module(self, path: QualifiedName, module: types.ModuleType) -> IRModule:
        self.error_collector.set_current_path(path)
        irmodule = IRModule(
            full_name=path,
            doc=get_doc(module),
            is_package=is_package(module),
        )

        for name, member in inspect.getmembers(module):
            self._handle_module_member(name, member, module, irmodule)

        return irmodule

    def _handle_module_member(
        self,
        name: str,
        member: Any,
        module: types.ModuleType,
        ilmodule: IRModule,
    ) -> None:
        path = ilmodule.full_name.concat(name)

        if self._is_imported_member(path, member, module):
            return
        if self._is_member_alias(path, member):
            return

        if inspect.isroutine(member):
            ilmodule.functions.append(self.build_function(path, member))
            return
        if inspect.isclass(member):
            ilmodule.classes.append(self.build_class(path, member))
            return
        if inspect.ismodule(member):
            ilmodule.sub_modules.append(self.build_module(path, member))

    def build_class(self, path: QualifiedName, class_: type) -> IRClass:
        self.error_collector.set_current_path(path)
        irclass = IRClass(name=path.name)
        irclass.doc = get_doc(class_)
        irclass.bases = self.build_bases(class_)

        for name, member in inspect.getmembers(class_):
            self._handle_class_member(name, member, path, class_, irclass)

        return irclass

    def _handle_class_member(
        self,
        name: str,
        member: Any,
        class_path: QualifiedName,
        class_: type,
        irclass: IRClass,
    ) -> None:
        path = class_path.concat(name)

        # 跳过从基类继承的成员（不在类自己的 __dict__ 中）
        if not hasattr(class_, "__dict__") or name not in class_.__dict__:
            return
        if self._is_member_alias(path, member):
            return

        if inspect.isroutine(member):
            irclass.methods.append(self.build_method(path, member))
            return
        if inspect.isclass(member):
            irclass.classes.append(self.build_class(path, member))

    def build_function(self, path: QualifiedName, func: Any) -> IRFunction:
        self.error_collector.set_current_path(path)
        irfunc = IRFunction(name=path.name, doc=get_doc(func))

        try:
            signature_target = func
            if inspect.ismethod(func) and inspect.isclass(getattr(func, "__self__", None)):
                signature_target = func.__func__

            sig = inspect.signature(signature_target)
            kind_map = {
                inspect.Parameter.POSITIONAL_ONLY: IRArgumentKind.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD: IRArgumentKind.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL: IRArgumentKind.VAR_POSITIONAL,
                inspect.Parameter.KEYWORD_ONLY: IRArgumentKind.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD: IRArgumentKind.VAR_KEYWORD,
            }

            for param in sig.parameters.values():
                arg = IRArgument(name=param.name, kind=kind_map[param.kind])
                if param.default is not inspect.Signature.empty:
                    arg.default = self._build_value(param.default)
                if param.annotation is not inspect.Signature.empty:
                    arg.annotation = self._build_annotation(param.annotation)
                irfunc.args.append(arg)

            if sig.return_annotation is not inspect.Signature.empty:
                irfunc.return_annotation = self._build_annotation(sig.return_annotation)
        except (TypeError, ValueError):
            # inspect.signature 失败时，回退为泛型签名，后续可由 DocString 解析修复
            irfunc.args = [
                IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
                IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
            ]
            irfunc.return_annotation = None
        return irfunc

    def build_method(self, path: QualifiedName, method: Any) -> IRMethod:
        func = self.build_function(path, method)
        return IRMethod(function=func, modifier=None)

    def build_bases(self, class_: type) -> list[QualifiedName]:
        bases = class_.__bases__
        result: list[QualifiedName] = []
        for t in bases:
            if t is object:
                continue
            base_name = self._get_type_fullname(t)
            # 在 pybind11_builtins 处停止（不包括它或随后的基类）
            if len(base_name) > 0 and base_name[0] == "pybind11_builtins":
                break
            result.append(base_name)
        return result

    def _build_annotation(self, annotation: Any) -> ResolvedType | IRValue | InvalidExpression:
        if isinstance(annotation, str):
            return self._parse_annotation_str(annotation)
        if isinstance(annotation, type):
            return ResolvedType(name=self._get_type_fullname(annotation))
        if self._is_generic_alias(annotation):
            return self._handle_generic_alias(annotation)
        return self._build_value(annotation)

    def _is_generic_alias(self, annotation: Any) -> bool:
        generic_alias = get_generic_alias_type()
        if generic_alias is not None:
            return isinstance(annotation, generic_alias)
        return False

    def _parse_annotation_str(
        self, annotation_str: str
    ) -> ResolvedType | InvalidExpression | IRValue:
        variants = self._split_type_union_str(annotation_str)
        if variants is None or len(variants) == 0:
            self.error_collector.report_error(InvalidExpressionError(annotation_str))
            return InvalidExpression(annotation_str)
        if len(variants) == 1:
            return self._parse_type_str(variants[0])
        return ResolvedType(
            name=QualifiedName.from_str("typing.Union"),
            parameters=[self._parse_type_str(variant) for variant in variants],
        )

    def _parse_type_str(
        self, annotation_str: str
    ) -> ResolvedType | InvalidExpression | IRValue:
        qname_regex = re.compile(
            r"^\s*(?P<qual_name>([_A-Za-z]\w*)?(\s*\.\s*[_A-Za-z]\w*)*)"
        )
        annotation_str = annotation_str.strip()
        match = qname_regex.match(annotation_str)
        if match is None:
            return self._parse_value_str(annotation_str)
        qual_name = QualifiedName(
            part for part in match.group("qual_name").replace(" ", "").split(".")
        )
        parameters_str = annotation_str[match.end("qual_name") :].strip()

        if len(parameters_str) == 0:
            parameters = None
        else:
            if parameters_str[0] != "[" or parameters_str[-1] != "]":
                return self._parse_value_str(annotation_str)
            split_parameters = self._split_parameters_str(parameters_str[1:-1])
            if split_parameters is None:
                return self._parse_value_str(annotation_str)
            parameters = [
                self._parse_annotation_str(param_str) for param_str in split_parameters
            ]
        return ResolvedType(name=qual_name, parameters=parameters)

    def _parse_value_str(self, value: str) -> IRValue | InvalidExpression:
        strip_expr = value.strip()
        try:
            ast.parse(strip_expr)
            print_safe = False
            try:
                ast.literal_eval(strip_expr)
                print_safe = True
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                pass
            return IRValue(strip_expr, is_print_safe=print_safe)
        except SyntaxError:
            self.error_collector.report_error(InvalidExpressionError(strip_expr))
            return InvalidExpression(strip_expr)

    def _split_type_union_str(self, type_str: str) -> list[str] | None:
        return self._split_str(type_str, delim="|")

    def _split_parameters_str(self, param_str: str) -> list[str] | None:
        return self._split_str(param_str, delim=",")

    def _split_str(self, param_str: str, delim: str) -> list[str] | None:
        result: list[str] = []
        closing = {"(": ")", "{": "}", "[": "]"}
        stack = []
        i = 0
        arg_begin = 0

        def add_arg() -> None:
            arg_end = i
            param = param_str[arg_begin:arg_end]
            result.append(param)

        while i < len(param_str):
            c = param_str[i]
            if c in "\"'":
                str_end = self._find_str_end(param_str, i)
                if str_end is None:
                    return None
                i = str_end
            elif c in closing:
                stack.append(closing[c])
            elif len(stack) == 0:
                if c == delim:
                    add_arg()
                    arg_begin = i + 1
            elif stack[-1] == c:
                stack.pop()
            i += 1
        if len(stack) != 0:
            return None
        if param_str[arg_begin:i].strip() != "":
            add_arg()
        return result

    def _find_str_end(self, s: str, start: int) -> int | None:
        for i in range(start + 1, len(s)):
            c = s[i]
            if c == "\\":
                continue
            if c == s[start]:
                return i
        return None

    def _handle_generic_alias(self, alias: Any) -> ResolvedType:
        origin = alias.__origin__
        args = alias.__args__

        parameters = [self._build_annotation(arg) for arg in args]
        return ResolvedType(name=self._get_type_fullname(origin), parameters=parameters)

    def _build_value(self, value: Any) -> IRValue:
        value_type = type(value)
        if value is Ellipsis:
            return IRValue(repr="...", is_print_safe=True)
        if value is None or value_type in (bool, int, str):
            return IRValue(repr=repr(value), is_print_safe=True)
        if value_type in (float, complex):
            try:
                repr_str = repr(value)
                eval(repr_str)
                return IRValue(repr=repr_str, is_print_safe=True)
            except (SyntaxError, NameError):
                pass
        if value_type in (list, tuple, set):
            if len(value) == 0:
                return IRValue(repr=f"{value_type.__name__}()", is_print_safe=True)
            elements = [self._build_value(el) for el in value]
            is_print_safe = all(el.is_print_safe for el in elements)
            left, right = {
                list: ("[", "]"),
                tuple: ("(", ")"),
                set: ("{", "}"),
            }[value_type]
            return IRValue(
                repr="".join([left, ", ".join(el.repr for el in elements), right]),
                is_print_safe=is_print_safe,
            )
        if value_type is dict:
            parts = []
            is_print_safe = True
            for k, v in value.items():
                k_value = self._build_value(k)
                v_value = self._build_value(v)
                parts.append(f"{k_value.repr}: {v_value.repr}")
                is_print_safe = (
                    is_print_safe and k_value.is_print_safe and v_value.is_print_safe
                )
            return IRValue(
                repr="".join(["{", ", ".join(parts), "}"]),
                is_print_safe=is_print_safe,
            )
        if inspect.isroutine(value):
            module_name = get_module_name(value)
            qual_name = getattr(value, "__qualname__", None)
            if (
                module_name is not None
                and "<" not in module_name
                and isinstance(qual_name, str)
                and "<" not in qual_name
            ):
                if module_name == "builtins":
                    repr_str = qual_name
                else:
                    repr_str = f"{module_name}.{qual_name}"
                return IRValue(repr=repr_str, is_print_safe=True)
        if inspect.isclass(value):
            return IRValue(repr=str(self._get_type_fullname(value)), is_print_safe=True)
        if inspect.ismodule(value):
            return IRValue(repr=value.__name__, is_print_safe=True)
        return IRValue(repr=repr(value), is_print_safe=False)

    def _get_type_fullname(self, type_: type) -> QualifiedName:
        module = type_.__module__
        qualname = type_.__qualname__
        if module == "builtins":
            return QualifiedName.from_str(qualname)
        return QualifiedName.from_str(f"{module}.{qualname}")

    def _get_value_parent_module_name(self, obj: Any) -> str | None:
        if inspect.ismodule(obj):
            return obj.__name__.rsplit(".", 1)[0]
        if inspect.isclass(obj) or inspect.isroutine(obj):
            return get_module_name(obj)
        return None

    def _is_imported_member(
        self, path: QualifiedName, member: Any, module: types.ModuleType
    ) -> bool:
        member_module = self._get_value_parent_module_name(member)
        return (
            (member_module is not None and member_module != module.__name__)
            or path.name == "annotations"
        )

    def _is_member_alias(self, path: QualifiedName, member: Any) -> bool:
        if (inspect.isroutine(member) or inspect.isclass(member)) and hasattr(
            member, "__name__"
        ):
            return path.name != member.__name__
        return False
