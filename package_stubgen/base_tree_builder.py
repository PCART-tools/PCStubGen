from __future__ import annotations

from typing import Final

from .stubdoc import FunctionSig
from .utils import get_short_name, is_private_name, is_not_in_all
from .import_tracker import ImportTracker
from .signatures import SignatureGenerator
from .models import ClassStubData, ModuleStubData, VariableInfo

class ClassInfo:
    def __init__(
        self,
        name: str,
        self_var: str,
        docstring: str | None = None,
        cls: type | None = None,
        parent: ClassInfo | None = None,
    ) -> None:
        self.name = name
        self.self_var = self_var
        self.docstring = docstring
        self.cls = cls
        self.parent = parent

class FunctionContext:
    def __init__(
        self,
        module_name: str,
        name: str,
        docstring: str | None = None,
        is_abstract: bool = False,
        class_info: ClassInfo | None = None,
    ) -> None:
        self.module_name = module_name
        self.name = name
        self.docstring = docstring
        self.is_abstract = is_abstract
        self.class_info = class_info
        self._fullname: str | None = None

    @property
    def fullname(self) -> str:
        if self._fullname is None:
            if self.class_info:
                parents = []
                class_info: ClassInfo | None = self.class_info
                while class_info is not None:
                    parents.append(class_info.name)
                    class_info = class_info.parent
                namespace = ".".join(reversed(parents))
                self._fullname = f"{self.module_name}.{namespace}.{self.name}"
            else:
                self._fullname = f"{self.module_name}.{self.name}"
        return self._fullname

class BaseTreeBuilder:
    def __init__(
        self,
        _all_: list[str] | None = None,
    ) -> None:
        self._all_ = _all_
        self.import_tracker = ImportTracker()
        self.defined_names: set[str] = set()
        self.sig_generators = self.get_sig_generators()
        self.module_name: str = ""
        self.docstring: str | None = None
        self._structured_variables: list[VariableInfo] = []
        self._structured_functions: list[FunctionSig] = []
        self._structured_classes: list[ClassStubData] = []
        self._recorded_names: set[str] = set()
        self.known_imports = {
            "_typeshed": ["Incomplete"],
            "typing": ["Any", "TypeVar", "NamedTuple", "TypedDict", "overload"],
            "collections.abc": ["Generator"],
            "typing_extensions": ["ParamSpec", "TypeVarTuple"],
        }

    def get_sig_generators(self) -> list[SignatureGenerator]:
        return []

    @property
    def short_name(self) -> str:
        """获取模块的短名称（不含包前缀）。"""
        return get_short_name(self.module_name)

    def resolve_name(self, name: str) -> str:
        if "." not in name:
            real_module = self.import_tracker.module_for.get(name)
            real_short = self.import_tracker.reverse_alias.get(name, name)
            if real_module is None and real_short not in self.defined_names:
                real_module = "builtins"
        else:
            name_module, real_short = name.split(".", 1)
            real_module = self.import_tracker.reverse_alias.get(name_module, name_module)
        resolved_name = real_short if real_module is None else f"{real_module}.{real_short}"
        return resolved_name

    def add_name(self, fullname: str, require: bool = True) -> str:
        if "." not in fullname:
            return fullname
        module, name = fullname.rsplit(".", 1)
        if module == "builtins":
            return name
        
        # 只有当名称与当前模块定义的顶级名称冲突时才使用别名
        alias = None
        if name in self.defined_names:
            alias = "_" + name
            while alias in self.defined_names:
                alias = "_" + alias
        
        self.import_tracker.add_import_from(module, [(name, alias)], require=require)
        return alias or name

    def record_name(self, name: str) -> None:
        self._recorded_names.add(name)

    def is_recorded_name(self, name: str) -> bool:
        return name in self._recorded_names

    def get_structured_output(self) -> ModuleStubData:
        # 用户请求在树结构中使用短名称
        return ModuleStubData(
            name=self.short_name,
            docstring=self.docstring,
            _all_=self._all_,
            imports=self.import_tracker.import_lines(),
            variables=self._structured_variables,
            functions=self._structured_functions,
            classes=self._structured_classes,
        )

    def set_defined_names(self, defined_names: set[str]) -> None:
        self.defined_names = defined_names
        for name in self._all_ or ():
            self.import_tracker.reexport(name)
        for pkg, imports in self.known_imports.items():
            for t in imports:
                self.add_name(f"{pkg}.{t}", require=False)

    def get_signatures(
        self,
        default_signature: FunctionSig,
        sig_generators: list[SignatureGenerator],
        func_ctx: FunctionContext,
    ) -> list[FunctionSig]:
        for sig_gen in sig_generators:
            inferred = sig_gen.get_function_sig(default_signature, func_ctx)
            if inferred:
                return inferred
        return [default_signature]

    def get_property_type(
        self,
        default_type: str | None,
        sig_generators: list[SignatureGenerator],
        func_ctx: FunctionContext,
    ) -> str | None:
        for sig_gen in sig_generators:
            inferred = sig_gen.get_property_type(default_type, func_ctx)
            if inferred:
                return inferred
        return default_type

    def is_not_in_all(self, name: str, is_top_level: bool = True) -> bool:
        return is_not_in_all(name, self._all_, is_top_level)

    def is_private_name(self, name: str, fullname: str | None = None) -> bool:
        # 在构造树时，我们默认不进行私有过滤，以便“完整保留数据”
        return is_private_name(name, self._all_, include_private=True)
