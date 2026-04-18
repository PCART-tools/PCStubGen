from __future__ import annotations

from pathlib import Path

import toml_rs

from .stub_renderer import StubRenderer
from ..models import Class, Function, Module


class TomlWriter:
    """将 Module 树导出为单个 TOML 文件。"""

    def write(
        self,
        module: Module,
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
        module: Module,
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
        module: Module,
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

        for class_node in sorted(module.classes, key=lambda current: current.name):
            entries.extend(
                self._collect_class_entries(
                    module_name=module_name,
                    class_path=(class_node.name,),
                    class_node=class_node,
                    renderer=renderer,
                )
            )

        return entries

    def _collect_class_entries(
        self,
        *,
        module_name: str,
        class_path: tuple[str, ...],
        class_node: Class,
        renderer: StubRenderer,
    ) -> list[dict[str, str]]:
        """递归收集类与嵌套类的方法记录。"""
        entries: list[dict[str, str]] = []
        class_name = ".".join(class_path)

        for method in sorted(class_node.methods, key=lambda current: current.name):
            entries.extend(
                self._build_method_entries(
                    module_name=module_name,
                    class_name=class_name,
                    method=method,
                    renderer=renderer,
                )
            )

        for nested_class in sorted(class_node.classes, key=lambda current: current.name):
            entries.extend(
                self._collect_class_entries(
                    module_name=module_name,
                    class_path=(*class_path, nested_class.name),
                    class_node=nested_class,
                    renderer=renderer,
                )
            )

        return entries

    def _build_method_entries(
        self,
        *,
        module_name: str,
        class_name: str,
        method: Function,
        renderer: StubRenderer,
    ) -> list[dict[str, str]]:
        """将类方法转换为 TOML 记录。"""
        if not method.signatures:
            return [
                self._build_entry(
                    module_name=module_name,
                    class_name=class_name,
                    function_name=method.name,
                    signature=None,
                    comment=method.comment,
                )
            ]

        return [
            self._build_entry(
                module_name=module_name,
                class_name=class_name,
                function_name=method.name,
                signature="\n".join(
                    renderer.render_method_signature_lines(
                        func=method,
                        signature=signature,
                    )
                ),
                comment=method.comment,
            )
            for signature in method.signatures
        ]

    def _build_function_entries(
        self,
        *,
        module_name: str,
        class_name: str | None,
        func: Function,
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
                    comment=func.comment,
                )
            ]

        return [
            self._build_entry(
                module_name=module_name,
                class_name=class_name,
                function_name=func.name,
                signature="\n".join(
                    renderer.render_function_signature_lines(
                        func_name=func.name,
                        signature=signature,
                    )
                ),
                comment=func.comment,
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
        comment: str | None,
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
        if comment is not None:
            entry["comment"] = comment
        return entry
