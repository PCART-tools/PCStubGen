from __future__ import annotations

import dataclasses
import re

@dataclasses.dataclass
class StubGenerationOptions:
    """存根生成配置项。"""
    
    # 分析/解析选项
    root_suffix: str | None = None
    ignore_invalid_expressions: re.Pattern | None = None
    ignore_invalid_identifiers: re.Pattern | None = None
    ignore_unresolved_names: re.Pattern | None = None
    ignore_all_errors: bool = False
    # 由 (正则, 前缀) 组成的列表
    enum_class_locations: list[tuple[re.Pattern, str]] = dataclasses.field(default_factory=list)
    
    # 与 NumPy 相关的支持选项
    numpy_array_wrap_with_annotated: bool = False
    numpy_array_use_type_var: bool = False
    numpy_array_remove_parameters: bool = False
    
    # 输出/打印选项
    print_invalid_expressions_as_is: bool = False
    print_safe_value_reprs: re.Pattern | None = None
    
    # 通用选项
    exit_code: bool = False  # 在库模式中未使用，保留以兼容/完整
    dry_run: bool = False    # 在库模式中未使用，保留以兼容/完整
    stub_extension: str = "pyi" # 由 write_stubs 使用
    
    # 模块名由 generate_stubs 单独传入
