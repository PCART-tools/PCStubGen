from __future__ import annotations

import sys

import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def reset_loguru_logger() -> None:
    """
    在每个测试结束后恢复全局 Loguru sink，避免 CLI 测试把 logger 绑定到已关闭的捕获流。
    """
    yield
    logger.remove()
    logger.add(sys.stderr)
