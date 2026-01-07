from __future__ import annotations

from .api import build_package_tree, generate_stub
from .models import ModuleStubData

# 保留此文件以实现向后兼容性
__all__ = [
    "build_package_tree",
    "generate_stub",
    "ModuleStubData"
]
