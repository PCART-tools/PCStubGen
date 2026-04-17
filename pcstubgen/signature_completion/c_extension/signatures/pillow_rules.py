from __future__ import annotations

from ....type_models import RawType, Type


PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "Imaging_Type": RawType("object"),
    "CmsProfile_Type": RawType(
        "PIL.ImageCms.core.CmsProfile",
        imports=("PIL.ImageCms",),
    ),
}

PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE: dict[str, Type] = {}

OBJECT_NAME_TO_TYPE: dict[str, Type] = {}

FUNCTION_NAME_TO_TYPE: dict[str, Type] = {}
