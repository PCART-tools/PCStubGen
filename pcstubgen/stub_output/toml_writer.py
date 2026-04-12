from __future__ import annotations

from pathlib import Path

import toml_rs

from .renderer import StubRenderer
from ..ir_modules import IRClass, IRFunction, IRMethod, IRModule


class TomlWriter:
    """将 IRModule 树导出为单个 TOML 文件。"""

    def write(
        self,
        module: IRModule,
        renderer: StubRenderer,
        to: Path,
    ) -> None:
        """把模块树展开并写入 TOML 文件。"""
        assert to.exists()
        assert to.is_dir()

        output_path = to / f"{module.full_name.name}.toml"
        entries = self._collect_module_entries(module, renderer)
        output_path.write_text(
            toml_rs.dumps({"entries": entries}, pretty=True),
            encoding="utf-8",
        )

    def _collect_module_entries(
        self,
        module: IRModule,
        renderer: StubRenderer,
    ) -> list[dict[str, str]]:
        """递归收集模块及其子模块的导出记录。"""
        entries: list[dict[str, str]] = []
        entries.extend(self._collect_current_module_entries(module, renderer))

        for sub_module in module.sub_modules:
            entries.extend(self._collect_module_entries(sub_module, renderer))

        return entries

    def _collect_current_module_entries(
        self,
        module: IRModule,
        renderer: StubRenderer,
    ) -> list[dict[str, str]]:
        """收集当前模块中的函数与类方法记录。"""
        entries: list[dict[str, str]] = []
        module_name = str(module.full_name)

        for func in sorted(module.functions, key=lambda current: current.name):
            entries.extend(
                self._build_function_entries(
                    module_name=module_name,
                    class_name=None,
                    func=func,
                    renderer=renderer,
                )
            )

        for irclass in sorted(module.classes, key=lambda current: current.name):
            entries.extend(
                self._collect_class_entries(
                    module_name=module_name,
                    class_path=(irclass.name,),
                    irclass=irclass,
                    renderer=renderer,
                )
            )

        return entries

    def _collect_class_entries(
        self,
        *,
        module_name: str,
        class_path: tuple[str, ...],
        irclass: IRClass,
        renderer: StubRenderer,
    ) -> list[dict[str, str]]:
        """递归收集类与嵌套类的方法记录。"""
        entries: list[dict[str, str]] = []
        class_name = ".".join(class_path)

        for method in sorted(irclass.methods, key=lambda current: current.function.name):
            entries.extend(
                self._build_method_entries(
                    module_name=module_name,
                    class_name=class_name,
                    method=method,
                    renderer=renderer,
                )
            )

        for nested_class in sorted(irclass.classes, key=lambda current: current.name):
            entries.extend(
                self._collect_class_entries(
                    module_name=module_name,
                    class_path=(*class_path, nested_class.name),
                    irclass=nested_class,
                    renderer=renderer,
                )
            )

        return entries

    def _build_method_entries(
        self,
        *,
        module_name: str,
        class_name: str,
        method: IRMethod,
        renderer: StubRenderer,
    ) -> list[dict[str, str]]:
        """将类方法转换为 TOML 记录。"""
        return self._build_function_entries(
            module_name=module_name,
            class_name=class_name,
            func=method.function,
            renderer=renderer,
        )

    def _build_function_entries(
        self,
        *,
        module_name: str,
        class_name: str | None,
        func: IRFunction,
        renderer: StubRenderer,
    ) -> list[dict[str, str]]:
        """将函数展开为一条或多条 TOML 记录。"""
        if not func.signatures:
            return [
                self._build_entry(
                    module_name=module_name,
                    class_name=class_name,
                    function_name=func.name,
                    signature=None,
                    source_comment=func.c_inferred_source_comment,
                )
            ]

        return [
            self._build_entry(
                module_name=module_name,
                class_name=class_name,
                function_name=func.name,
                signature=renderer.render_function_signature(
                    func_name=func.name,
                    signature=signature,
                ),
                source_comment=func.c_inferred_source_comment,
            )
            for signature in func.signatures
        ]

    @staticmethod
    def _build_entry(
        *,
        module_name: str,
        class_name: str | None,
        function_name: str,
        signature: str | None,
        source_comment: str | None,
    ) -> dict[str, str]:
        """构造单条 TOML 记录。"""
        entry: dict[str, str] = {
            "module_name": module_name,
            "function_name": function_name,
        }
        if class_name is not None:
            entry["class_name"] = class_name
        if signature is not None:
            entry["signature"] = signature
        if source_comment is not None:
            entry["source_comment"] = source_comment
        return entry
