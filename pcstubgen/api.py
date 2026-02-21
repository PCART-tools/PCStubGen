from __future__ import annotations

from .module_inspector import ModuleInspector, InspectError
from .inspection_tree_builder import InspectionTreeBuilder
from .models import ModuleStubData
from .utils import get_parent_name, get_short_name
from .generator import StubGenerator

def build_package_tree(
    package: str,
) -> ModuleStubData:
    """为给定包中的 C 模块构建树状结构。
    返回以该包为根的树状结构。
    """
    module_data_map: dict[str, ModuleStubData] = {}

    with ModuleInspector() as inspect:
        try:
            all_props = inspect.inspect_package(package)
        except InspectError as e:
            raise InspectError(f"无法检查包 {package!r}: {e}") from e

        all_found_names = [p.name for p in all_props]

        # 第一步：为所有发现的模块/包初始化 ModuleStubData
        # 如果是 C 模块，则生成详细的结构化数据
        for prop in all_props:
            short_name = get_short_name(prop.name)
            if prop.is_c_module:
                try:
                    builder = InspectionTreeBuilder(
                        module_name=prop.name,
                        known_modules=all_found_names,
                        _all_=prop.all,
                    )
                    builder.generate_module()
                    module_data_map[prop.name] = builder.get_structured_output()
                except (InspectError, ImportError, Exception):
                    # 如果 C 模块加载失败，至少保留一个空的结构
                    module_data_map[prop.name] = ModuleStubData(name=short_name)
            else:
                # 非 C 模块（通常是中间包层级）
                module_data_map[prop.name] = ModuleStubData(name=short_name)

        # 第二步：构建树形层级结构
        for mod_name in sorted(module_data_map):
            data = module_data_map[mod_name]
            
            # 建立父子关系
            parent_name = get_parent_name(mod_name)
            if parent_name != "":
                if parent_name in module_data_map:
                    parent_data = module_data_map[parent_name]
                    parent_data.submodules.append(data)

        if package not in module_data_map:
            raise InspectError(f"找不到包 {package!r} 的数据")

        root = module_data_map[package]
        return root

def generate_stub(
    module_data: ModuleStubData,
    include_private: bool = True,
    include_docstrings: bool = False,
) -> str:
    """将 ModuleStubData 节点转换为存根字符串。"""
    generator = StubGenerator(
        include_private=include_private, include_docstrings=include_docstrings
    )
    return generator.generate_module(module_data)

