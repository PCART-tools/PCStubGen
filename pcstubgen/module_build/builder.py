from __future__ import annotations

import importlib
import importlib.machinery
import inspect
import pkgutil
import types
from typing import Any

from loguru import logger

from .reflection import (
    get_doc,
    get_module_name,
    is_package,
)
from ..ir import (
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRModuleType,
    QualifiedName,
)


def build_module(path: QualifiedName, module: types.ModuleType) -> IRModule:
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

        if inspect.isroutine(member):
            irmodule.functions.append(
                build_function(member_path, member, module_type=module_type)
            )
        elif inspect.isclass(member):
            irmodule.classes.append(
                build_class(member_path, member, module_type=module_type)
            )

    if irmodule.is_package:
        for submodule_name in _iter_submodule_names(module):
            try:
                sub_module = importlib.import_module(submodule_name)
            except (ImportError, OSError) as ex:
                missing_dependency = None
                if isinstance(ex, ModuleNotFoundError):
                    missing_dependency = ex.name
                logger.warning(
                    "跳过子模块, module: {}, error_type: {}, missing_dependency: {}, error: {}",
                    submodule_name,
                    type(ex).__name__,
                    missing_dependency,
                    ex,
                )
                continue
            irmodule.sub_modules.append(
                build_module(QualifiedName.from_str(submodule_name), sub_module)
            )

    return irmodule


def _detect_module_type(module: types.ModuleType) -> IRModuleType:
    spec = module.__spec__
    loader = getattr(spec, "loader", None) if spec is not None else None

    if loader is importlib.machinery.BuiltinImporter: # 编译进Python
        return IRModuleType.BUILTIN

    if isinstance(loader, importlib.machinery.ExtensionFileLoader): # .pyd .so
        return IRModuleType.EXTENSION

    if isinstance(
        loader,
        (
            importlib.machinery.SourcelessFileLoader, # .pyc
            importlib.machinery.SourceFileLoader, # .py
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


def build_class(
    path: QualifiedName,
    class_: type,
    *,
    module_type: IRModuleType = IRModuleType.UNKNOWN,
) -> IRClass:
    irclass = IRClass(name=path.name, doc=get_doc(class_))
    irclass.bases = build_bases(class_)

    for name, member in inspect.getmembers(class_):
        member_path = path.concat(name)

        # 跳过从基类继承的成员（不在类自己的 __dict__ 中）
        if not hasattr(class_, "__dict__") or name not in class_.__dict__:
            continue
        if _is_member_alias(member_path, member):
            continue

        if inspect.isroutine(member):
            irclass.methods.append(
                build_method(member_path, member, module_type=module_type)
            )
        elif inspect.isclass(member):
            irclass.classes.append(
                build_class(member_path, member, module_type=module_type)
            )

    return irclass


def build_function(
    path: QualifiedName,
    func: Any,
    *,
    module_type: IRModuleType = IRModuleType.UNKNOWN,
) -> IRFunction:
    _ = module_type
    return IRFunction(name=path.name, doc=get_doc(func))


def build_method(
    path: QualifiedName,
    method: Any,
    *,
    module_type: IRModuleType = IRModuleType.UNKNOWN,
) -> IRMethod:
    func = build_function(path, method, module_type=module_type)
    return IRMethod(function=func, decorator=None)


def build_bases(class_: type) -> list[QualifiedName]:
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
