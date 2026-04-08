from __future__ import annotations

import importlib
import importlib.machinery
import inspect
import pkgutil
import types
from typing import Any

from loguru import logger

from .ir_modules import (
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRModuleType,
    QualifiedName,
)

__all__ = [
    "collect_bases",
    "collect_class",
    "collect_function",
    "collect_method",
    "collect_module",
]


def get_doc(obj: Any) -> str | None:
    doc = getattr(obj, "__doc__", None)
    if isinstance(doc, str) and doc and not doc.isspace():
        return doc
    return None


def get_module_name(obj: Any) -> str | None:
    module_name = getattr(obj, "__module__", None)
    if isinstance(module_name, str):
        return module_name
    return None


def is_package(module: types.ModuleType) -> bool:
    spec = module.__spec__
    if spec is None:
        return False
    return spec.submodule_search_locations is not None


def collect_module(path: QualifiedName, module: types.ModuleType) -> IRModule:
    module_type = _detect_module_type(module)
    irmodule = IRModule(
        full_name=path,
        doc=get_doc(module),
        is_package=is_package(module),
        module_type=module_type,
    )
    for name, member in inspect.getmembers(module):
        member_path = irmodule.full_name.concat(name)

        if _is_imported_member(member_path, member, module):
            continue
        if _is_member_alias(member_path, member):
            continue

        if inspect.isbuiltin(member):
            irmodule.functions.append(
                collect_function(member_path, member, module_type=module_type)
            )
        elif inspect.isclass(member):
            irmodule.classes.append(
                collect_class(member_path, member, module_type=module_type)
            )

    if irmodule.is_package:
        for submodule_name in _iter_submodule_names(module):
            try:
                sub_module = importlib.import_module(submodule_name)
            except BaseException as ex:
                logger.error(
                    "模块导入失败, 安装来获得更完整的存根. module: {}, error: {!r}",
                    submodule_name,
                    ex,
                )
                continue
            irmodule.sub_modules.append(
                collect_module(QualifiedName.from_str(submodule_name), sub_module)
            )

    return irmodule


def _detect_module_type(module: types.ModuleType) -> IRModuleType:
    spec = module.__spec__
    loader = getattr(spec, "loader", None) if spec is not None else None

    if loader is importlib.machinery.BuiltinImporter:  # 编译进Python
        return IRModuleType.BUILTIN

    if isinstance(loader, importlib.machinery.ExtensionFileLoader):  # .pyd .so
        return IRModuleType.EXTENSION

    if isinstance(
        loader,
        (
            importlib.machinery.SourcelessFileLoader,  # .pyc
            importlib.machinery.SourceFileLoader,  # .py
        ),
    ):
        return IRModuleType.PYTHON

    return IRModuleType.UNKNOWN


def _iter_submodule_names(module: types.ModuleType) -> list[str]:
    """按 import 拓扑枚举包的直接子模块全名。"""
    spec = module.__spec__
    assert spec is not None
    assert spec.submodule_search_locations is not None

    return sorted(
        module_info.name
        for module_info in pkgutil.iter_modules(
            spec.submodule_search_locations,
            prefix=f"{module.__name__}.",
        )
    )


def collect_class(
    path: QualifiedName,
    class_: type,
    *,
    module_type: IRModuleType = IRModuleType.UNKNOWN,
) -> IRClass:
    irclass = IRClass(name=path.name, doc=get_doc(class_))
    irclass.bases = collect_bases(class_)

    for name, member in class_.__dict__.items():
        member_path = path.concat(name)

        if _is_member_alias(member_path, member):
            continue

        if _is_c_method_member(member):
            irclass.methods.append(
                collect_method(
                    member_path,
                    member,
                    module_type=module_type,
                    owner=class_,
                )
            )
        elif inspect.isclass(member):
            irclass.classes.append(
                collect_class(member_path, member, module_type=module_type)
            )

    return irclass


def collect_function(
    path: QualifiedName,
    func: Any,
    *,
    module_type: IRModuleType = IRModuleType.UNKNOWN,
) -> IRFunction:
    _ = module_type
    return IRFunction(
        name=path.name,
        doc=get_doc(func),
        runtime_handle=func,
    )


def collect_method(
    path: QualifiedName,
    method: Any,
    *,
    module_type: IRModuleType = IRModuleType.UNKNOWN,
    owner: type | None = None,
) -> IRMethod:
    func = collect_function(path, method, module_type=module_type)
    return IRMethod(
        function=func,
        decorator=None,
        runtime_owner=owner,
    )


def collect_bases(class_: type) -> list[QualifiedName]:
    bases = class_.__bases__
    result: list[QualifiedName] = []
    for t in bases:
        if t is object:
            continue
        base_name = _get_type_fullname(t)
        # 在 pybind11_builtins 处停止（不包括它或随后的基类）
        if len(base_name) > 0 and base_name[0] == "pybind11_builtins":
            break
        result.append(base_name)
    return result


def _get_type_fullname(type_: type) -> QualifiedName:
    module = type_.__module__
    qualname = type_.__qualname__
    if module == "builtins":
        return QualifiedName.from_str(qualname)
    return QualifiedName.from_str(f"{module}.{qualname}")


def _get_value_parent_module_name(obj: Any) -> str | None:
    if inspect.ismodule(obj):
        return obj.__name__.rsplit(".", 1)[0]
    if inspect.isclass(obj) or inspect.isroutine(obj):
        return get_module_name(obj)
    return None


def _is_imported_member(
    path: QualifiedName, member: Any, module: types.ModuleType
) -> bool:
    member_module = _get_value_parent_module_name(member)
    return (
        (member_module is not None and member_module != module.__name__)
        or path.name == "annotations"
    )


def _is_member_alias(path: QualifiedName, member: Any) -> bool:
    if (inspect.isroutine(member) or inspect.isclass(member)) and hasattr(
        member, "__name__"
    ):
        return path.name != member.__name__
    return False


def _is_c_method_member(member: Any) -> bool:
    return inspect.isbuiltin(member) or inspect.ismethoddescriptor(member)
