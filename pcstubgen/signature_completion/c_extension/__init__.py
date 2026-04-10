from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .source import CExtensionSource

__all__ = ["CExtensionSource"]
