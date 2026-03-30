from __future__ import annotations

import pytest

from pcstubgen.checks import check


def test_check_raises_runtime_error_when_condition_is_false() -> None:
    with pytest.raises(RuntimeError, match="前置条件检查失败。"):
        check(False)
