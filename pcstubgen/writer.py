from __future__ import annotations

from pathlib import Path

from .stub_printer import StubPrinter
from .ir import IRModule
from .ir.ir_module import IRModuleType


class Writer:
    def write(
        self,
        module: IRModule,
        printer: StubPrinter,
        to: Path,
    ) -> None:
        assert to.exists()
        assert to.is_dir()

        if module.sub_modules or module.is_package:
            module_dir = to / module.full_name.name
            module_dir.mkdir(exist_ok=True)
            module_file = module_dir / "__init__.pyi"
        else:
            module_dir = to
            module_file = to / f"{module.full_name.name}.pyi"

        if module.module_type == IRModuleType.EXTENSION:
            with open(module_file, "w", encoding="utf-8") as f:
                for line in printer.print_module(module):
                    f.write(line + "\n")

        for sub_module in module.sub_modules:
            self.write(sub_module, printer, module_dir)
