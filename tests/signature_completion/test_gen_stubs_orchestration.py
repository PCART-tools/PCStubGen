from __future__ import annotations

from pcstubgen.models import Argument, Function, Module, QualifiedName
from pcstubgen.type_models import RawType
from tests._c_extension_test_support import _signature


def test_gen_stubs_writes_stub_from_collected_module(
    monkeypatch,
    tmp_path,
) -> None:
    import pcstubgen.api as stubgen_module

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.ClangParser",
        lambda compilation_database: object(),
    )
    (tmp_path / "compile_commands.json").write_text("[]", encoding="utf-8")

    module_node = Module(
        full_name=QualifiedName.from_str("math"),
        functions=[
            Function(
                name="foo",
                handle=object(),
                signatures=[
                    _signature(
                        args=[Argument(name="value", type=RawType("str"))],
                        return_type=RawType("bool"),
                    )
                ],
            )
        ],
    )

    monkeypatch.setattr(
        stubgen_module.ModuleCollector,
        "run",
        lambda self, module_name: module_node,
    )

    stubgen_module.gen_stubs(
        "math",
        tmp_path,
        tmp_path / "compile_commands.json",
    )

    stub_path = tmp_path / "math.pyi"
    assert stub_path.exists()
    assert stub_path.read_text(encoding="utf-8") == "def foo(value: str) -> bool:\n    ...\n"
