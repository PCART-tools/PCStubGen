from __future__ import annotations

import contextlib
import importlib
import inspect
import os
import pkgutil
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from types import ModuleType

class ModuleProperties:
    def __init__(
        self,
        name: str = "",
        file: str | None = None,
        path: list[str] | None = None,
        all: list[str] | None = None,
        is_c_module: bool = False,
    ) -> None:
        self.name = name
        self.file = file
        self.path = path
        self.all = all
        self.is_c_module = is_c_module

def is_c_module(module: ModuleType) -> bool:
    if module.__dict__.get("__file__") is None:
        return True
    return os.path.splitext(module.__dict__["__file__"])[-1] in [".so", ".pyd", ".dll"]

class InspectError(Exception):
    pass

def get_package_properties(package_id: str) -> ModuleProperties:
    try:
        package = importlib.import_module(package_id)
    except BaseException as e:
        raise InspectError(str(e)) from e
    name = getattr(package, "__name__", package_id)
    file = getattr(package, "__file__", None)
    path: list[str] | None = getattr(package, "__path__", None)
    if not isinstance(path, list):
        path = None
    pkg_all = getattr(package, "__all__", None)
    if pkg_all is not None:
        try:
            pkg_all = list(pkg_all)
        except Exception:
            pkg_all = None
    is_c = is_c_module(package)

    return ModuleProperties(
        name=name, file=file, path=path, all=pkg_all, is_c_module=is_c
    )

def inspect_package_recursive(package_id: str) -> list[ModuleProperties]:
    """递归检查包及其所有子模块。"""
    root_prop = get_package_properties(package_id)
    results = [root_prop]
    
    if root_prop.path is not None:
        # 如果是包，扫描所有子模块
        all_packages = pkgutil.walk_packages(
            root_prop.path, prefix=root_prop.name + ".", onerror=lambda r: None
        )
        for importer, qualified_name, ispkg in all_packages:
            try:
                results.append(get_package_properties(qualified_name))
            except InspectError:
                continue
    elif root_prop.is_c_module:
        # 如果是 C 模块，检查它是否包含子模块（有些 C 模块会模拟包结构）
        try:
            package = importlib.import_module(package_id)
            for name, val in inspect.getmembers(package):
                if inspect.ismodule(val) and val.__name__ == package.__name__ + "." + name:
                    try:
                        results.append(get_package_properties(val.__name__))
                    except InspectError:
                        continue
        except Exception:
            pass
            
    return results

def _worker(package_id: str, sys_path: list[str]) -> list[ModuleProperties]:
    sys.path = sys_path
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return inspect_package_recursive(package_id)
    except InspectError:
        raise
    except BaseException as e:
        raise InspectError(str(e)) from e

class ModuleInspector:
    def __init__(self) -> None:
        self.counter = 0
        self._executor: ProcessPoolExecutor | None = None

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            if sys.platform == "linux":
                ctx = get_context("forkserver")
            else:
                ctx = get_context("spawn")

            # 如果 Python >= 3.11，可以使用 max_tasks_per_child=1 来确保完全隔离
            kwargs = {}
            if sys.version_info >= (3, 11):
                kwargs["max_tasks_per_child"] = 1

            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                **kwargs
            )
        return self._executor

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def inspect_package(self, package_id: str) -> list[ModuleProperties]:
        executor = self._get_executor()
        future = executor.submit(_worker, package_id, sys.path)
        try:
            res = future.result(timeout=60)  # 递归检查可能需要更长时间
        except Exception as e:
            if self.counter > 0 and not isinstance(e, InspectError):
                self.close()
                return self.inspect_package(package_id)
            if isinstance(e, InspectError):
                raise e
            raise InspectError(f"在检查包 {package_id!r} 时发生错误: {e}") from e

        self.counter += 1
        return res

    def __enter__(self) -> ModuleInspector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
