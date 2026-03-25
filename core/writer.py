from __future__ import annotations

from pathlib import Path

from .printer_visitor import PrinterVisitor
from .ir import IRModule


class Writer:
    def write(
        self,
        module: IRModule,
        printer: PrinterVisitor,
        to: Path,
    ) -> None:
        assert to.exists()
        assert to.is_dir()

        if module.sub_modules or module.is_package:
            module_dir = to / module.Name
            module_dir.mkdir(exist_ok=True)
            module_file = module_dir / "__init__.pyi"
        else:
            module_dir = to
            module_file = to / f"{module.Name}.pyi"

        with open(module_file, "w", encoding="utf-8") as f:
            for line in printer.visit_module(module):
                f.write(line + "\n")

        for sub_module in module.sub_modules:
            self.write(sub_module, printer, module_dir)
