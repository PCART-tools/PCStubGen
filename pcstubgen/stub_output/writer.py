from __future__ import annotations

from pathlib import Path

from .renderer import StubRenderer
from ..ir_modules import IRModule


class StubWriter:
    def write(
        self,
        module: IRModule,
        renderer: StubRenderer,
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

        with open(module_file, "w", encoding="utf-8") as f:
            for line in renderer.print_module(module):
                f.write(line + "\n")

        for sub_module in module.sub_modules:
            self.write(sub_module, renderer, module_dir)
