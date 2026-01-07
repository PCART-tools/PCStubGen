from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

class ImportTracker:
    """在存根（stub）生成过程中记录必要的导入。"""

    def __init__(self) -> None:
        self.module_for: dict[str, str | None] = {}
        self.direct_imports: dict[str, str] = {}
        self.reverse_alias: dict[str, str] = {}
        self.required_names: set[str] = set()
        self.reexports: set[str] = set()

    def add_import_from(
        self, module: str, names: list[tuple[str, str | None]], require: bool = False
    ) -> None:
        for name, alias in names:
            if alias:
                self.module_for[alias] = module
                self.reverse_alias[alias] = name
            else:
                self.module_for[name] = module
                self.reverse_alias.pop(name, None)
            if require:
                self.require_name(alias or name)
            self.direct_imports.pop(alias or name, None)

    def add_import(self, module: str, alias: str | None = None, require: bool = False) -> None:
        if alias:
            self.module_for[alias] = None
            self.reverse_alias[alias] = module
            if require:
                self.required_names.add(alias)
        else:
            name = module
            if require:
                self.required_names.add(name)
            while name:
                self.module_for[name] = None
                self.direct_imports[name] = module
                self.reverse_alias.pop(name, None)
                name = name.rpartition(".")[0]

    def require_name(self, name: str) -> None:
        while name not in self.direct_imports and "." in name:
            name = name.rsplit(".", 1)[0]
        self.required_names.add(name)

    def reexport(self, name: str) -> None:
        self.require_name(name)
        self.reexports.add(name)

    def import_lines(self) -> list[str]:
        import_statements = []
        from_import_map: Mapping[str, list[str]] = defaultdict(list)

        # 收集所有导入
        for name in sorted(
            self.required_names,
            key=lambda n: (self.reverse_alias[n], n) if n in self.reverse_alias else (n, ""),
        ):
            if name not in self.module_for:
                continue

            m = self.module_for[name]
            if m is not None:
                # from module import name [as alias]
                if name in self.reverse_alias:
                    import_name = f"{self.reverse_alias[name]} as {name}"
                elif name in self.reexports:
                    import_name = f"{name} as {name}"
                else:
                    import_name = name
                from_import_map[m].append(import_name)
            else:
                # import module [as alias]
                if name in self.reverse_alias:
                    source = self.reverse_alias[name]
                    import_statements.append(f"import {source} as {name}")
                elif name in self.reexports:
                    import_statements.append(f"import {name} as {name}")
                else:
                    import_statements.append(f"import {name}")

        # 官方 stubgen 通常将 import 放在 from ... import 之前
        result = sorted(import_statements)

        # 格式化 from ... import ...
        # 官方 stubgen 通常将 typing 放在最后
        sorted_modules = sorted(from_import_map.keys(), key=lambda m: (m == "typing", m))
        for module in sorted_modules:
            names = sorted(list(set(from_import_map[module])))
            result.append(f"from {module} import {', '.join(names)}")
        return result
