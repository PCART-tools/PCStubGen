from __future__ import annotations

from pcstubgen.models import QualifiedName
from pcstubgen.signature_completion.completion_models import SignatureCompletionContext
from pcstubgen.signature_completion.pybind11_provider import Pybind11Provider


_PybindRecord = type("pybind_record", (), {})
_PybindRecord.__module__ = "pybind11_builtins"

_FakeInstanceMethod = type(
    "instancemethod",
    (),
    {"__call__": lambda self, *args, **kwargs: None},
)
_FakeInstanceMethod.__module__ = "builtins"


class _FakeBuiltinFunction:
    def __init__(self, doc: str) -> None:
        self.__self__ = _PybindRecord()
        self.__doc__ = doc

    def __call__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs


def _make_context(member: object, *, func_name: str, is_method: bool = False) -> SignatureCompletionContext:
    return SignatureCompletionContext(
        module_name=QualifiedName.from_str("pkg.mod"),
        func_name=func_name,
        member=member,
        is_method=is_method,
    )


def _make_pybind11_instance_method(doc: str) -> object:
    member = _FakeInstanceMethod()
    member.__self__ = _PybindRecord()
    member.__doc__ = doc
    return member


def test_pybind11_provider_supports_module_level_builtin(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.runtime.is_pybind11_module_function",
        lambda handle: handle is sentinel,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.runtime.is_pybind11_instance_method",
        lambda handle: False,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.runtime.is_pybind11_static_method",
        lambda handle: False,
    )

    assert Pybind11Provider.support(sentinel, False) is True
    assert Pybind11Provider.support(object(), False) is False


def test_pybind11_provider_gets_instance_method_result() -> None:
    provider = Pybind11Provider()
    member = _make_pybind11_instance_method("build(self: pkg.Sample, value: int) -> int")

    result = provider.get(_make_context(member, func_name="build", is_method=True))

    assert result.provider == "pybind11"
    assert result.decorator is None
    assert result.doc == "build(self: pkg.Sample, value: int) -> int"
    assert [arg.name for arg in result.signatures[0].args] == ["self", "value"]


def test_pybind11_provider_gets_staticmethod_result() -> None:
    provider = Pybind11Provider()
    func = _FakeBuiltinFunction("build(self: pkg.Sample, value: int) -> int")
    member = staticmethod(func)

    result = provider.get(_make_context(member, func_name="build", is_method=True))

    assert result.provider == "pybind11"
    assert result.decorator == "staticmethod"
    assert result.doc == "build(self: pkg.Sample, value: int) -> int"
    assert [arg.name for arg in result.signatures[0].args] == ["value"]


def test_pybind11_provider_rejects_non_pybind11_member() -> None:
    assert Pybind11Provider.support(object(), False) is False
    assert Pybind11Provider.support(object(), True) is False
