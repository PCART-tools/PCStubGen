from __future__ import annotations

import importlib
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
    QualifiedName,
)

__all__ = ["ModuleCollector"]


class ModuleCollector:
    """收集模块运行时对象并构建 IRModule 树。"""

    def run(self, module_name: str) -> IRModule:
        """导入目标模块并递归收集其 IR 结构。"""
        module = importlib.import_module(module_name)
        return self._collect_module(QualifiedName.from_str(module_name), module)

    def _collect_module(
        self,
        path: QualifiedName,
        module: types.ModuleType,
    ) -> IRModule:
        """收集单个模块及其直接子模块。"""
        irmodule = IRModule(
            full_name=path,
            doc=self._get_doc(module),
            is_package=self._is_package(module),
        )
        for name, member in inspect.getmembers(module):
            member_path = irmodule.full_name.concat(name)

            if self._is_imported_member(member_path, member, module):
                continue
            if self._is_member_alias(member_path, member):
                continue

            if inspect.isbuiltin(member):
                irmodule.functions.append(self._collect_function(member_path, member))
            elif inspect.isclass(member):
                irmodule.classes.append(self._collect_class(member_path, member))

        if irmodule.is_package:
            for submodule_name in self._iter_submodule_names(module):
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
                    self._collect_module(QualifiedName.from_str(submodule_name), sub_module)
                )

        return irmodule

    def _collect_class(
        self,
        path: QualifiedName,
        class_: type,
    ) -> IRClass:
        """收集类、方法和嵌套类。"""
        irclass = IRClass(name=path.name, doc=self._get_doc(class_))
        irclass.bases = self._collect_bases(class_)

        for name, member in class_.__dict__.items():
            member_path = path.concat(name)

            if self._is_member_alias(member_path, member):
                continue

            if inspect.isbuiltin(member):
                irclass.methods.append(
                    self._collect_method(
                        member_path,
                        member,
                        owner=class_,
                    )
                )
            elif inspect.isclass(member):
                irclass.classes.append(self._collect_class(member_path, member))

        return irclass

    def _collect_function(
        self,
        path: QualifiedName,
        func: Any,
    ) -> IRFunction:
        """收集函数节点但不补全签名。"""
        return IRFunction(
            name=path.name,
            doc=self._get_doc(func),
            runtime_handle=func,
        )

    def _collect_method(
        self,
        path: QualifiedName,
        method: Any,
        *,
        owner: type | None = None,
    ) -> IRMethod:
        """收集方法节点并记录所属类型。"""
        func = self._collect_function(path, method)
        return IRMethod(
            function=func,
            decorator=None,
            runtime_owner=owner,
        )

    def _collect_bases(self, class_: type) -> list[QualifiedName]:
        """收集类基类，遇到 pybind11 builtins 时停止。"""
        bases = class_.__bases__
        result: list[QualifiedName] = []
        for type_ in bases:
            if type_ is object:
                continue
            base_name = self._get_type_fullname(type_)
            if len(base_name) > 0 and base_name[0] == "pybind11_builtins":
                break
            result.append(base_name)
        return result

    @staticmethod
    def _get_doc(obj: Any) -> str | None:
        """读取对象的非空文档字符串。"""
        doc = obj.__doc__
        if isinstance(doc, str) and doc and not doc.isspace():
            return doc
        return None

    @staticmethod
    def _is_package(module: types.ModuleType) -> bool:
        """判断模块是否为包或命名空间包。"""
        spec = module.__spec__
        assert spec is not None
        return spec.submodule_search_locations is not None

    def _iter_submodule_names(self, module: types.ModuleType) -> list[str]:
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

    @staticmethod
    def _get_type_fullname(type_: type) -> QualifiedName:
        """获取类型的完整限定名。"""
        module = type_.__module__
        qualname = type_.__qualname__
        if module == "builtins":
            return QualifiedName.from_str(qualname)
        return QualifiedName.from_str(f"{module}.{qualname}")

    def _get_value_parent_module_name(self, obj: Any) -> str | None:
        """获取成员值所属的上级模块名。"""
        if inspect.ismodule(obj):
            return obj.__name__.rsplit(".", 1)[0]
        if inspect.isclass(obj) or inspect.isroutine(obj):
            return self._get_module_name(obj)
        return None

    @staticmethod
    def _get_module_name(obj: Any) -> str | None:
        """读取对象的 `__module__`。"""
        module_name = obj.__module__
        if isinstance(module_name, str):
            return module_name
        return None

    def _is_imported_member(
        self,
        path: QualifiedName,
        member: Any,
        module: types.ModuleType,
    ) -> bool:
        """判断成员是否来自外部模块导入。"""
        member_module = self._get_value_parent_module_name(member)
        return (
            (member_module is not None and member_module != module.__name__)
            or path.name == "annotations"
        )

    @staticmethod
    def _is_member_alias(path: QualifiedName, member: Any) -> bool:
        """判断成员名是否只是原对象名称的别名。"""
        if inspect.isroutine(member) or inspect.isclass(member):
            return path.name != member.__name__
        return False
