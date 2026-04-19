from __future__ import annotations

from pcstubgen.signature_completion.pybind11_provider import Pybind11Provider


_PybindRecord = type("pybind_record", (), {})
_PybindRecord.__module__ = "pybind11_builtins"

_FakeInstanceMethod = type("instancemethod", (), {"__call__": lambda self, *args, **kwargs: None})
_FakeInstanceMethod.__module__ = "builtins"


class _FakeBuiltinFunction:
    def __init__(self, doc: str) -> None:
        self.__self__ = _PybindRecord()
        self.__doc__ = doc

    def __call__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs


def _make_pybind11_instance_method(doc: str) -> object:
    member = _FakeInstanceMethod()
    member.__self__ = _PybindRecord()
    member.__doc__ = doc
    return member


def test_pybind11_provider_supports_module_level_builtin(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.runtime.is_pybind11_builtin",
        lambda handle: handle is sentinel,
    )

    assert Pybind11Provider.support(sentinel) is True
    assert Pybind11Provider.support(object()) is False


def test_pybind11_provider_normalizes_instance_method_doc_and_handle() -> None:
    member = _make_pybind11_instance_method("build(self: pkg.Sample) -> int")

    normalized = Pybind11Provider.normalize_class_member(member)

    assert normalized == (member, None, "build(self: pkg.Sample) -> int")


def test_pybind11_provider_normalizes_staticmethod_doc_and_decorator() -> None:
    func = _FakeBuiltinFunction("build(value: int) -> int")
    member = staticmethod(func)

    normalized = Pybind11Provider.normalize_class_member(member)

    assert normalized == (func, "staticmethod", "build(value: int) -> int")


def test_pybind11_provider_rejects_non_pybind11_member() -> None:
    class normal_method:
        def __call__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

    member = normal_method()
    member.__self__ = object()
    member.__doc__ = "build(self) -> int"

    assert Pybind11Provider.normalize_class_member(member) is None
