from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from clang.cindex import Cursor

from pcstubgen.models import Argument, ArgumentKind
from pcstubgen.types import RawType, Type, UnionType


@dataclass(frozen=True)
class _FakeCursor:
    name: str


def _cursor(name: str) -> Cursor:
    return cast(Cursor, _FakeCursor(name))


_BUFFER_TYPE = RawType("collections.abc.Buffer", imports=("collections.abc",))
_STR_OR_NONE_TYPE = UnionType((RawType("str"), RawType("None")))
_STR_OR_BUFFER_TYPE = UnionType((RawType("str"), _BUFFER_TYPE))
_STR_OR_BUFFER_OR_NONE_TYPE = UnionType((RawType("str"), _BUFFER_TYPE, RawType("None")))
_STR_OR_BYTES_OR_BYTEARRAY_TYPE = UnionType(
    (RawType("str"), RawType("bytes"), RawType("bytearray"))
)


def _arg(
    name: str,
    type_text: str | Type,
    *,
    imports: tuple[str, ...] = (),
    default_value: str | None = None,
    kind: ArgumentKind = ArgumentKind.POSITIONAL_OR_KEYWORD,
) -> Argument:
    return Argument(
        name=name,
        type=type_text if isinstance(type_text, Type) else RawType(type_text, imports=imports),
        default_value=default_value,
        kind=kind,
    )
