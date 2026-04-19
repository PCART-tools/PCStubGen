from __future__ import annotations

from .....type_models import RawType, Type, UnionType


_IMAGING_CORE_TYPE = RawType(
    "PIL.Image.core.ImagingCore",
    imports=("PIL.Image",),
)
_IMAGING_DECODER_TYPE = RawType(
    "PIL.Image.core.ImagingDecoder",
    imports=("PIL.Image",),
)
_IMAGING_ENCODER_TYPE = RawType(
    "PIL.Image.core.ImagingEncoder",
    imports=("PIL.Image",),
)
_IMAGING_FONT_TYPE = RawType(
    "PIL.Image.core.ImagingFont",
    imports=("PIL.Image",),
)
_IMAGING_DRAW_TYPE = RawType(
    "PIL.Image.core.ImagingDraw",
    imports=("PIL.Image",),
)
_PIXEL_ACCESS_TYPE = RawType(
    "PIL.Image.core.PixelAccess",
    imports=("PIL.Image",),
)
_OUTLINE_TYPE = RawType(
    "PIL.Image.core._Outline",
    imports=("PIL.Image",),
)
_IMAGE_PATH_TYPE = RawType(
    "PIL.ImagePath.Path",
    imports=("PIL.ImagePath",),
)
_CMS_PROFILE_TYPE = RawType(
    "PIL.ImageCms.core.CmsProfile",
    imports=("PIL.ImageCms",),
)
_CMS_TRANSFORM_TYPE = RawType(
    "PIL.ImageCms.core.CmsTransform",
    imports=("PIL.ImageCms",),
)
_IMAGING_FT_FONT_TYPE = RawType(
    "PIL._imagingft.Font",
    imports=("PIL._imagingft",),
)
_ERROR_RETURN_TYPE = UnionType(())


PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "Imaging_Type": _IMAGING_CORE_TYPE,
    "ImagingDecoder_Type": _IMAGING_DECODER_TYPE,
    "ImagingEncoder_Type": _IMAGING_ENCODER_TYPE,
    "ImagingFont_Type": _IMAGING_FONT_TYPE,
    "ImagingDraw_Type": _IMAGING_DRAW_TYPE,
    "PixelAccess_Type": _PIXEL_ACCESS_TYPE,
    "Outline_Type": _OUTLINE_TYPE,
    "CmsProfile_Type": _CMS_PROFILE_TYPE,
    "CmsTransform_Type": _CMS_TRANSFORM_TYPE,
    "Font_Type": _IMAGING_FT_FONT_TYPE,
}

PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE: dict[str, Type] = {}

OBJECT_NAME_TO_TYPE: dict[str, Type] = {}

CALL_NAME_TO_TYPE: dict[str, Type] = {
    "ImagingError_MemoryError": _ERROR_RETURN_TYPE,
    "ImagingError_ValueError": _ERROR_RETURN_TYPE,
    "HandleMuxError": _ERROR_RETURN_TYPE,
    "geterror": _ERROR_RETURN_TYPE,
    "PyImagingNew": _IMAGING_CORE_TYPE,
    "PyImaging_DecoderNew": _IMAGING_DECODER_TYPE,
    "PyImaging_EncoderNew": _IMAGING_ENCODER_TYPE,
    "cms_profile_new": _CMS_PROFILE_TYPE,
    "cms_transform_new": _CMS_TRANSFORM_TYPE,
    "_outline_new": _OUTLINE_TYPE,
    "path_new": _IMAGE_PATH_TYPE,
}
