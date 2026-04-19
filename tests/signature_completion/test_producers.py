from __future__ import annotations

from pcstubgen.models import Signature
from pcstubgen.signature_completion import producers
from tests._c_extension_test_support import _arg


def test_finalize_signatures_adds_instance_receiver() -> None:
    finalized = producers._finalize_signatures(
        [Signature(args=[_arg("value", "int")])],
        is_method=True,
        decorator=None,
    )

    assert [arg.name for arg in finalized[0].args] == ["self", "value"]


def test_finalize_signatures_rewrites_classmethod_receiver() -> None:
    finalized = producers._finalize_signatures(
        [Signature(args=[_arg("self"), _arg("value", "int")])],
        is_method=True,
        decorator="classmethod",
    )

    assert [arg.name for arg in finalized[0].args] == ["cls", "value"]


def test_finalize_signatures_strips_staticmethod_receiver() -> None:
    finalized = producers._finalize_signatures(
        [Signature(args=[_arg("self"), _arg("value", "int")])],
        is_method=True,
        decorator="staticmethod",
    )

    assert [arg.name for arg in finalized[0].args] == ["value"]
