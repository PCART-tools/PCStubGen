from __future__ import annotations

from pcstubgen.models import Function, Module, QualifiedName
from pcstubgen.stub_output import StubRenderer, StubWriter
from tests._c_extension_test_support import _signature
from pcstubgen.type_models import RawType


def test_stub_writer_writes_regular_module_file(tmp_path) -> None:
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="foo",
                runtime_handle=object(),
                signatures=[
                    _signature(
                        return_type=RawType("int"),
                    )
                ],
            )
        ],
    )

    StubWriter().write(
        module,
        StubRenderer(include_docstrings=False),
        to=tmp_path,
    )

    stub_path = tmp_path / "mod.pyi"
    assert stub_path.exists()
    assert stub_path.read_text(encoding="utf-8") == "def foo() -> int:\n    ...\n"
