from __future__ import annotations


def check(condition: bool, message: str = "前置条件检查失败。") -> None:
    """校验内部前置条件，不满足时抛出 `RuntimeError`。"""
    if not condition:
        raise RuntimeError(message)
