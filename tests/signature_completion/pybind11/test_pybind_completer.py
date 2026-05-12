from __future__ import annotations

import pytest

from pcstubgen.models import QualifiedName
from pcstubgen.signature_completion.completion_models import (
    PartialSignatureCompletionError,
    SignatureCompletionContext,
)
from pcstubgen.signature_completion.pybind11.completer import Pybind11Completer


_PybindRecord = type("pybind_record", (), {})
_PybindRecord.__module__ = "pybind11_builtins"

_FakeInstanceMethod = type(
    "instancemethod",
    (),
    {"__call__": lambda self, *args, **kwargs: None},
)
_FakeInstanceMethod.__module__ = "builtins"


class _FakeBuiltinFunction:
    def __init__(self, doc: str, name: str = "build") -> None:
        self.__self__ = _PybindRecord()
        self.__doc__ = doc
        self.__name__ = name

    def __call__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs


def _make_context(
    member: object,
    *,
    func_name: str,
    owner_class: type | None = None,
) -> SignatureCompletionContext:
    return SignatureCompletionContext(
        module_name=QualifiedName.from_str("pkg.mod"),
        func_name=func_name,
        member=member,
        owner_class=owner_class,
    )


def _make_pybind11_instance_method(doc: str) -> object:
    member = _FakeInstanceMethod()
    member.__self__ = _PybindRecord()
    member.__doc__ = doc
    member.__name__ = "build"
    return member


@pytest.mark.parametrize(
    ("case_name", "owner_class", "expected"),
    [
        ("module", None, True),
        ("instance", object, True),
        ("conduit", object, False),
        ("plain", None, False),
    ],
)
def test_pybind11_completer_match_filters_members(
    case_name: str,
    owner_class: type | None,
    expected: bool,
) -> None:
    completer = Pybind11Completer()
    module_member = object()
    instance_member = _make_pybind11_instance_method("build(self: pkg.Sample, value: int) -> int")
    setattr(instance_member, "__name__", "build")
    conduit_member = _make_pybind11_instance_method("ignored(self: pkg.Sample) -> None")
    setattr(conduit_member, "__name__", "_pybind11_conduit_v1_")
    member = {
        "module": module_member,
        "instance": instance_member,
        "conduit": conduit_member,
        "plain": object(),
    }[case_name]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_module_function",
            lambda handle: handle is module_member,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_instance_method",
            lambda handle: handle is instance_member or handle is conduit_member,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_static_method",
            lambda handle: False,
        )

        assert completer.match(member, owner_class) is expected


@pytest.mark.parametrize(
    ("member_factory", "func_name", "owner_class", "expected_doc", "expected_decorator", "expected_args"),
    [
        (
            lambda: _make_pybind11_instance_method("build(self: pkg.Sample, value: int) -> int"),
            "build",
            object,
            "build(self: pkg.Sample, value: int) -> int",
            None,
            ["self", "value"],
        ),
        (
            lambda: staticmethod(_FakeBuiltinFunction("build(value: int) -> int")),
            "build",
            object,
            "build(value: int) -> int",
            "staticmethod",
            ["value"],
        ),
    ],
)
def test_pybind11_completer_get_returns_observable_result(
    member_factory,
    func_name: str,
    owner_class: type | None,
    expected_doc: str,
    expected_decorator: str | None,
    expected_args: list[str],
) -> None:
    completer = Pybind11Completer()
    member = member_factory()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_module_function",
            lambda handle: False,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_instance_method",
            lambda handle: handle is member and not isinstance(member, staticmethod),
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_static_method",
            lambda handle: handle is member and isinstance(member, staticmethod),
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.extract_pybind11_signatures",
            lambda handle: ["(self: pkg.Sample, value: int) -> int"]
            if expected_decorator is None
            else ["(value: int) -> int"],
        )
        result = completer.get(
            _make_context(member, func_name=func_name, owner_class=owner_class)
        )

    assert result.decorator == expected_decorator
    assert result.doc == expected_doc
    assert [arg.name for arg in result.signatures[0].args] == expected_args
    assert result.signatures[0].raw_signature == (
        "(self: pkg.Sample, value: int) -> int"
        if expected_decorator is None
        else "(value: int) -> int"
    )


def test_pybind11_completer_get_uses_runtime_name_for_doc_matching() -> None:
    completer = Pybind11Completer()
    member = _make_pybind11_instance_method("_mtia_exchangeDevice(arg0: typing.SupportsInt) -> int")
    setattr(member, "__name__", "_mtia_exchangeDevice")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_module_function",
            lambda handle: False,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_instance_method",
            lambda handle: handle is member,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_static_method",
            lambda handle: False,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.extract_pybind11_signatures",
            lambda handle: ["(arg0: typing.SupportsInt) -> int"],
        )
        result = completer.get(
            _make_context(member, func_name="exchange_device", owner_class=object)
        )

    assert [arg.name for arg in result.signatures[0].args] == ["arg0"]
    assert result.signatures[0].return_type is not None
    assert result.signatures[0].return_type.render() == "int"


def test_pybind11_completer_get_fails_when_one_overload_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completer = Pybind11Completer()
    member = _make_pybind11_instance_method("build(self: pkg.Sample, value: int) -> int")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_module_function",
            lambda handle: False,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_instance_method",
            lambda handle: handle is member,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_static_method",
            lambda handle: False,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.extract_pybind11_signatures",
            lambda handle: [
                "(self: pkg.Sample, value: int) -> int",
                "broken",
                "(self: pkg.Sample, value: str) -> str",
            ],
        )
        with pytest.raises(PartialSignatureCompletionError, match="overload=2"):
            completer.get(
                _make_context(member, func_name="build", owner_class=object)
            )


def test_pybind11_completer_get_does_not_parse_full_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completer = Pybind11Completer()
    member = _make_pybind11_instance_method(
        "build(*args, **kwargs)\nOverloaded function.\n1. build(value: int) -> int"
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_module_function",
            lambda handle: False,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_instance_method",
            lambda handle: handle is member,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.runtime.is_pybind11_static_method",
            lambda handle: False,
        )
        monkeypatch.setattr(
            "pcstubgen.signature_completion.pybind11.completer.extract_pybind11_signatures",
            lambda handle: ["(value: int) -> int"],
        )
        result = completer.get(
            _make_context(member, func_name="build", owner_class=object)
        )

    assert [arg.name for arg in result.signatures[0].args] == ["value"]
