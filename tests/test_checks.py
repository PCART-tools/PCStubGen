from __future__ import annotations

import pytest

from pcstubgen._checks import check


def test_check_does_not_raise_when_condition_is_true() -> None:
    check(True)


def test_check_raises_runtime_error_when_condition_is_false() -> None:
    with pytest.raises(RuntimeError, match="前置条件检查失败。"):
        check(False)


def test_check_preserves_custom_message() -> None:
    with pytest.raises(RuntimeError, match="custom message"):
        check(False, "custom message")
