from __future__ import annotations

from pathlib import Path

import toml_rs

from . import stub_renderer
from .stub_renderer import StubRenderer
from ..models import Class, Function, Module, Signature


def _collect_module_functions(module: Module) -> list[dict[str, object]]:
    """递归收集模块及其子模块的函数记录。"""
    functions: list[dict[str, object]] = []
    functions.extend(_collect_current_module_functions(module))

    for sub_module in module.sub_modules:
        functions.extend(_collect_module_functions(sub_module))

    return functions


def _collect_current_module_functions(module: Module) -> list[dict[str, object]]:
    """收集当前模块中的函数与类方法记录。"""
    functions: list[dict[str, object]] = []
    module_name = str(module.full_name)

    for func in sorted(module.functions, key=lambda current: current.name):
        functions.append(
            _build_function_entry(
                module_name=module_name,
                class_name=None,
                func=func,
            )
        )

    for class_node in sorted(module.classes, key=lambda current: current.name):
        functions.extend(
            _collect_class_functions(
                module_name=module_name,
                class_path=(class_node.name,),
                class_node=class_node,
            )
        )

    return functions


def _collect_class_functions(
    module_name: str,
    class_path: tuple[str, ...],
    class_node: Class,
) -> list[dict[str, object]]:
    """递归收集类与嵌套类的方法记录。"""
    functions: list[dict[str, object]] = []
    class_name = ".".join(class_path)

    for method in sorted(class_node.methods, key=lambda current: current.name):
        functions.append(
            _build_function_entry(
                module_name=module_name,
                class_name=class_name,
                func=method,
            )
        )

    for nested_class in sorted(class_node.classes, key=lambda current: current.name):
        functions.extend(
            _collect_class_functions(
                module_name=module_name,
                class_path=(*class_path, nested_class.name),
                class_node=nested_class,
            )
        )

    return functions


def _build_function_entry(
    *,
    module_name: str,
    class_name: str | None,
    func: Function,
) -> dict[str, object]:
    """将函数或方法构造为函数层 TOML 记录。"""
    if not func.signatures:
        raise RuntimeError(f"函数 {module_name}.{func.name} 缺少可导出签名。")

    entry: dict[str, object] = {
        "function_id": _build_function_id(module_name, class_name, func.name),
        "module_name": module_name,
        "function_name": func.name,
        "provider": func.provider or "",
        "mapping_status": func.mapping_status,
        "parameter_inference_status": func.parameter_inference_status,
        "return_inference_status": func.return_inference_status,
        "signatures": [
            _build_signature_entry(
                index=index,
                function_name=func.name,
                signature=signature,
            )
            for index, signature in enumerate(func.signatures)
        ],
    }
    _set_optional(entry, "class_name", class_name)
    _set_optional(entry, "decorator", func.decorator)
    _set_optional(entry, "failure_reason", func.failure_reason)
    _set_optional(entry, "source_location", func.source_location)
    _set_optional(entry, "source_text", func.source_text)
    return entry


def _build_signature_entry(
    *,
    index: int,
    function_name: str,
    signature: Signature,
) -> dict[str, object]:
    """构造签名层 TOML 记录。"""
    entry: dict[str, object] = {
        "signature_index": index,
        "signature": "\n".join(
            stub_renderer.render_function_signature(
                func_name=function_name,
                signature=signature,
            )
        ),
    }
    _set_optional(entry, "raw_signature", signature.raw_signature)
    return entry


def _build_function_id(
    module_name: str,
    class_name: str | None,
    function_name: str,
) -> str:
    """构造稳定的函数级标识。"""
    if class_name is None:
        return f"{module_name}:{function_name}"
    return f"{module_name}:{class_name}.{function_name}"


def _set_optional(
    entry: dict[str, object],
    key: str,
    value: object | None,
) -> None:
    """仅在值存在时写入 TOML 字段。"""
    if value is not None:
        entry[key] = value


class TomlWriter:
    """将 Module 树导出为单个 TOML 文件。"""

    def write(
        self,
        module: Module,
        renderer: StubRenderer,
        to: Path,
    ) -> None:
        """把模块树展开并写入函数层 TOML 文件。"""
        _ = renderer
        assert to.exists()
        assert to.is_dir()

        output_path = to / f"{module.full_name.name}.toml"
        functions = _collect_module_functions(module)
        output_path.write_text(
            toml_rs.dumps({"functions": functions}, pretty=True),
            encoding="utf-8",
        )
