from __future__ import annotations

import ast
import importlib.machinery
import inspect
import types
from typing import Any

from .error_collector import ErrorCollector
from .reflection_helpers import (
    get_doc,
    get_module_name,
    is_package,
)
from .ir import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    InvalidExpression,
    IRMethod,
    IRModule,
    IRModuleType,
    QualifiedName,
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
            module_type=self._detect_module_type(module),
        )
        for name, member in inspect.getmembers(module):
            member_path = irmodule.full_name.concat(name)

            if self._is_imported_member(member_path, member, module):
                continue
            if self._is_member_alias(member_path, member):
                continue

            if inspect.isroutine(member):
                irmodule.functions.append(self.build_function(member_path, member))
            elif inspect.isclass(member):
                irmodule.classes.append(self.build_class(member_path, member))
            elif inspect.ismodule(member):
                irmodule.sub_modules.append(self.build_module(member_path, member))

        return irmodule

    @staticmethod
    def _detect_module_type(module: types.ModuleType) -> IRModuleType:
        spec = getattr(module, "__spec__", None)
        loader = getattr(spec, "loader", None) if spec is not None else None

        if loader is importlib.machinery.BuiltinImporter:
            return IRModuleType.BUILTIN

        if isinstance(loader, importlib.machinery.ExtensionFileLoader):
            return IRModuleType.EXTENSION

        if isinstance(
            loader,
            (
                importlib.machinery.SourcelessFileLoader,
                importlib.machinery.SourceFileLoader,
            ),
        ):
            return IRModuleType.PYTHON

        return IRModuleType.UNKNOWN

    def build_class(self, path: QualifiedName, class_: type) -> IRClass:
        self.error_collector.set_current_path(path)
        irclass = IRClass(name=path.name, doc=get_doc(class_))
        irclass.bases = self.build_bases(class_)

        for name, member in inspect.getmembers(class_):
            member_path = path.concat(name)

            # 跳过从基类继承的成员（不在类自己的 __dict__ 中）
            if not hasattr(class_, "__dict__") or name not in class_.__dict__:
                continue
            if self._is_member_alias(member_path, member):
                continue

            if inspect.isroutine(member):
                irclass.methods.append(self.build_method(member_path, member))
            elif inspect.isclass(member):
                irclass.classes.append(self.build_class(member_path, member))

        return irclass

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
        except (TypeError, ValueError) as ex:
            # try:
            #     fullargspec = inspect.getfullargspec(signature_target)
            #     print(f"fullargspec: {fullargspec}\n")
            # except (TypeError, ValueError) as ex2:
            #     print(f"getfullargspec 失败，回退为泛型签名: {path}\nEx: {ex2}\n")

            # inspect.signature 失败时，回退为泛型签名，后续可由 DocString 解析修复
            # print(f"inspect.signature 失败，回退为泛型签名: {path}\nEx: {ex}\n")
            irfunc.args = [
                IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
                IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
            ]
            irfunc.return_annotation = None
        return irfunc

    def build_method(self, path: QualifiedName, method: Any) -> IRMethod:
        func = self.build_function(path, method)
        return IRMethod(function=func, decorator=None)

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

    def _build_annotation(self, annotation: Any) -> str | None:
        if isinstance(annotation, str):
            return self._normalize_annotation_text(annotation)
        if isinstance(annotation, type):
            return str(self._get_type_fullname(annotation))
        return self._normalize_annotation_text(self._build_value(annotation).repr)

    @staticmethod
    def _normalize_annotation_text(annotation_text: str | None) -> str | None:
        if annotation_text is None:
            return None
        text = annotation_text.strip()
        return text or None

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
