from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .renderer import StubRenderer
from ..ir_modules import IRClass, IRFunction, IRMethod, IRModule


class JsonWriter:
    """将 IRModule 树导出为单个 JSON 文件。"""

    def write(
        self,
        module: IRModule,
        renderer: StubRenderer,
        to: Path,
    ) -> None:
        """把模块树展开并写入 JSON 文件。"""
        assert to.exists()
        assert to.is_dir()

        output_path = to / f"{module.full_name.name}.json"
        records = self._collect_module_records(module, renderer)
        output_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _collect_module_records(
        self,
        module: IRModule,
        renderer: StubRenderer,
    ) -> list[dict[str, Any]]:
        """递归收集模块及其子模块的导出记录。"""
        records: list[dict[str, Any]] = []
        records.extend(self._collect_current_module_records(module, renderer))

        for sub_module in module.sub_modules:
            records.extend(self._collect_module_records(sub_module, renderer))

        return records

    def _collect_current_module_records(
        self,
        module: IRModule,
        renderer: StubRenderer,
    ) -> list[dict[str, Any]]:
        """收集当前模块中的函数与直接类方法记录。"""
        records: list[dict[str, Any]] = []
        module_name = str(module.full_name)

        for func in sorted(module.functions, key=lambda current: current.name):
            records.extend(
                self._build_function_records(
                    module_name=module_name,
                    class_name=None,
                    func=func,
                    renderer=renderer,
                )
            )

        for irclass in sorted(module.classes, key=lambda current: current.name):
            records.extend(
                self._collect_class_records(
                    module_name=module_name,
                    irclass=irclass,
                    renderer=renderer,
                )
            )

        return records

    def _collect_class_records(
        self,
        *,
        module_name: str,
        irclass: IRClass,
        renderer: StubRenderer,
    ) -> list[dict[str, Any]]:
        """收集直接类方法记录，不展开嵌套类。"""
        records: list[dict[str, Any]] = []
        for method in sorted(irclass.methods, key=lambda current: current.function.name):
            records.extend(
                self._build_method_records(
                    module_name=module_name,
                    class_name=irclass.name,
                    method=method,
                    renderer=renderer,
                )
            )
        return records

    def _build_method_records(
        self,
        *,
        module_name: str,
        class_name: str,
        method: IRMethod,
        renderer: StubRenderer,
    ) -> list[dict[str, Any]]:
        """将类方法转换为 JSON 记录。"""
        return self._build_function_records(
            module_name=module_name,
            class_name=class_name,
            func=method.function,
            renderer=renderer,
        )

    def _build_function_records(
        self,
        *,
        module_name: str,
        class_name: str | None,
        func: IRFunction,
        renderer: StubRenderer,
    ) -> list[dict[str, Any]]:
        """将函数展开为一条或多条 JSON 记录。"""
        if not func.signatures:
            return [
                self._build_record(
                    module_name=module_name,
                    class_name=class_name,
                    function_name=func.name,
                    signature=None,
                    source_comment=func.c_inferred_source_comment,
                )
            ]

        return [
            self._build_record(
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
    def _build_record(
        *,
        module_name: str,
        class_name: str | None,
        function_name: str,
        signature: str | None,
        source_comment: str | None,
    ) -> dict[str, Any]:
        """构造单条 JSON 记录。"""
        return {
            "module_name": module_name,
            "class_name": class_name,
            "function_name": function_name,
            "signature": signature,
            "source_comment": source_comment,
        }
