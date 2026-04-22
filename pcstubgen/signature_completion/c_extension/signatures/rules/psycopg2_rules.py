from __future__ import annotations

from .....type_models import RawType, Type


_CONNECTION_TYPE = RawType(
    "psycopg2.extensions.connection",
    imports=("psycopg2.extensions",),
)
_TYPECAST_TYPE = RawType(
    "psycopg2._psycopg.type",
    imports=("psycopg2._psycopg",),
)


PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "connectionType": _CONNECTION_TYPE,
    "typecastType": _TYPECAST_TYPE,
    "Text_Type": RawType.str_,
}

PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE: dict[str, Type] = {}

OBJECT_NAME_TO_TYPE: dict[str, Type] = {}

CALL_NAME_TO_TYPE: dict[str, Type] = {
    "PyInt_FromLong": RawType.int_,
    "PyInt_FromSsize_t": RawType.int_,
    "Bytes_FromString": RawType.bytes_,
    "conn_text_from_chars": RawType.str_,
}
