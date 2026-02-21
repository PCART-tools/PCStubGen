from __future__ import annotations

import importlib
import importlib.abc
import inspect
import keyword
from types import FunctionType, ModuleType
from typing import Any, Callable

from .module_inspector import is_c_module
from .utils import method_name_sort_key, IGNORED_DUNDERS
from .stubdoc import (
    ArgSig,
    FunctionSig,
    infer_c_method_args,
    infer_method_arg_types,
    infer_method_ret_type,
)
from .base_tree_builder import BaseTreeBuilder, ClassInfo, FunctionContext
from .signatures import DocstringSignatureGenerator, SignatureGenerator
from .models import ClassStubData, PropertyInfo, VariableInfo

class _Missing(importlib.abc.MetaPathFinder):
    VALUE = 1

class InspectionTreeBuilder(BaseTreeBuilder):
    def __init__(
        self,
        module_name: str,
        known_modules: list[str],
        _all_: list[str] | None = None,
        module: ModuleType | None = None,
    ) -> None:
        if module is None:
            self.module = importlib.import_module(module_name)
        else:
            self.module = module
        self.is_c_module = is_c_module(self.module)
        self.known_modules = known_modules
        self.resort_members = self.is_c_module
        super().__init__(_all_)
        self.module_name = module_name
        if self.is_c_module:
            self.known_imports.update(
                {
                    "typing": [
                        "Any", "Callable", "ClassVar", "Dict", "Iterable",
                        "Iterator", "List", "Literal", "NamedTuple",
                        "Optional", "Tuple", "Union",
                    ]
                }
            )

    def get_sig_generators(self) -> list[SignatureGenerator]:
        return [DocstringSignatureGenerator()]

    def get_default_function_sig(self, func: object, ctx: FunctionContext) -> FunctionSig:
        argspec = None
        if not self.is_c_module:
            try:
                argspec = inspect.getfullargspec(func)
            except TypeError:
                pass
        if argspec is None:
            if ctx.class_info is not None:
                return FunctionSig(
                    name=ctx.name,
                    args=infer_c_method_args(ctx.name, ctx.class_info.self_var),
                    ret_type=infer_method_ret_type(ctx.name),
                )
            else:
                return FunctionSig(
                    name=ctx.name,
                    args=[ArgSig(name="*args"), ArgSig(name="**kwargs")],
                    ret_type=None,
                )

        args = argspec.args
        defaults = argspec.defaults
        varargs = argspec.varargs
        kwargs = argspec.varkw
        annotations = argspec.annotations
        kwonlyargs = argspec.kwonlyargs
        kwonlydefaults = argspec.kwonlydefaults

        def get_annotation(key: str) -> str | None:
            if key not in annotations:
                return None
            argtype = annotations[key]
            if argtype is None:
                return "None"
            if not isinstance(argtype, str):
                return self.get_type_fullname(argtype)
            return argtype

        arglist: list[ArgSig] = []

        def add_args(
            args: list[str], get_default_value: Callable[[int, str], Any]
        ) -> None:
            for i, arg in enumerate(args):
                default_value = get_default_value(i, arg)
                if default_value is not _Missing.VALUE:
                    if arg in annotations:
                        argtype = get_annotation(arg)
                    else:
                        argtype = self.get_type_annotation(default_value)
                        if argtype == "None":
                            incomplete = self.add_name("_typeshed.Incomplete")
                            argtype = f"{incomplete} | None"
                    arglist.append(ArgSig(arg, argtype, default=True))
                else:
                    arglist.append(ArgSig(arg, get_annotation(arg), default=False))

        def get_pos_default(i: int, _arg: str) -> Any:
            if defaults and i >= len(args) - len(defaults):
                return defaults[i - (len(args) - len(defaults))]
            else:
                return _Missing.VALUE

        add_args(args, get_pos_default)

        if varargs:
            arglist.append(ArgSig(f"*{varargs}", get_annotation(varargs)))
        elif kwonlyargs:
            arglist.append(ArgSig("*"))

        def get_kw_default(_i: int, arg: str) -> Any:
            if kwonlydefaults and arg in kwonlydefaults:
                return kwonlydefaults[arg]
            else:
                return _Missing.VALUE

        add_args(kwonlyargs, get_kw_default)

        if kwargs:
            arglist.append(ArgSig(f"**{kwargs}", get_annotation(kwargs)))

        if ctx.class_info is not None and all(
            arg.type is None and arg.default is False for arg in arglist
        ):
            new_args = infer_method_arg_types(
                ctx.name, ctx.class_info.self_var, [arg.name for arg in arglist if arg.name]
            )
            if new_args is not None:
                arglist = new_args

        ret_type = get_annotation("return") or infer_method_ret_type(ctx.name)
        return FunctionSig(ctx.name, arglist, ret_type)

    def get_type_fullname(self, typ: type) -> str:
        if typ is Any:
            return self.add_name("typing.Any")
        typename = getattr(typ, "__qualname__", typ.__name__)
        module_name = getattr(typ, "__module__", None)
        if module_name is None:
            return self.add_name("_typeshed.Incomplete")
        if module_name != "builtins":
            # 确保模块被导入
            self.import_tracker.add_import(module_name, require=True)
            typename = f"{module_name}.{typename}"
        return typename

    def get_type_annotation(self, obj: object) -> str:
        if obj is None or obj is type(None):
            return "None"
        elif inspect.isclass(obj):
            return f"type[{self.get_type_fullname(obj)}]"
        elif isinstance(obj, FunctionType):
            return self.add_name("typing.Callable")
        elif isinstance(obj, ModuleType):
            return self.add_name("types.ModuleType")
        else:
            return self.get_type_fullname(type(obj))

    def generate_module(self) -> None:
        self.docstring = getattr(self.module, "__doc__", None)
        all_items = self.get_members(self.module)
        if self.resort_members:
            all_items = sorted(all_items, key=lambda x: x[0])
        items = []
        for name, obj in all_items:
            if inspect.ismodule(obj) and obj.__name__ in self.known_modules:
                module_name = obj.__name__
                self.import_tracker.add_import(module_name, name)
                self.import_tracker.reexport(name)
            elif self.is_defined_in_module(obj) and not inspect.ismodule(obj):
                items.append((name, obj))
            else:
                if not self.is_private_name(name):
                    obj_module_name = getattr(obj, "__module__", None)
                    if obj_module_name and obj_module_name != "builtins":
                        self.import_tracker.add_import_from(obj_module_name, [(name, None)])
                        self.import_tracker.reexport(name)

        self.set_defined_names({name for name, obj in all_items if not inspect.ismodule(obj)})

        for name, obj in items:
            if self.is_function(obj):
                self.generate_function_stub(name, obj)
            elif inspect.isclass(obj):
                # Official stubgen seems to include all classes defined in the module
                self.record_name(name)
                self.generate_class_stub(name, obj)
            else:
                self.record_name(name)
                self.generate_variable_stub(name, obj)

    def is_defined_in_module(self, obj: object) -> bool:
        module = getattr(obj, "__module__", None)
        return module is None or module == self.module_name

    def get_members(self, obj: object) -> list[tuple[str, Any]]:
        results = []
        for name, value in inspect.getmembers(obj):
            if self.is_skipped_attribute(name):
                continue
            results.append((name, value))
        return results

    def is_skipped_attribute(self, attr: str) -> bool:
        return (
            attr in (
                "__class__", "__getattribute__", "__str__", "__repr__",
                "__doc__", "__dict__", "__module__", "__weakref__",
                "__annotations__", "__firstlineno__", "__static_attributes__",
                "__annotate__",
            )
            or attr in IGNORED_DUNDERS
            or keyword.iskeyword(attr)
        )

    def is_function(self, obj: object) -> bool:
        if type(obj).__name__ == "ufunc":
            return False
        if self.is_c_module:
            return inspect.isbuiltin(obj) or (
                callable(obj) and not inspect.isclass(obj) and not hasattr(obj, "__mro__")
            )
        else:
            return inspect.isfunction(obj)

    def generate_function_stub(
        self, name: str, obj: object, *, class_info: ClassInfo | None = None
    ) -> list[FunctionSig]:
        docstring = getattr(obj, "__doc__", None)
        if not isinstance(docstring, str):
            docstring = None

        ctx = FunctionContext(
            self.module_name, name, docstring=docstring,
            is_abstract=getattr(obj, "__abstractmethod__", False),
            class_info=class_info,
        )

        if class_info is None:
            if self.is_recorded_name(name):
                return []
            self.record_name(name)

        default_sig = self.get_default_function_sig(obj, ctx)
        inferred = self.get_signatures(default_sig, self.sig_generators, ctx)

        # 处理装饰器和 self/cls 参数
        overload_name = self.add_name("typing.overload") if len(inferred) > 1 else None
        
        processed_sigs = []
        for sig in inferred:
            decorators = []
            if overload_name:
                decorators.append(f"@{overload_name}")
            
            if class_info is not None:
                if self.is_staticmethod(class_info, name, obj):
                    decorators.append("@staticmethod")
                elif self.is_classmethod(class_info, name, obj):
                    decorators.append("@classmethod")
                
                if not self.is_staticmethod(class_info, name, obj):
                    if not sig.args or sig.args[0].name not in ("self", "cls"):
                        sig.args.insert(0, ArgSig(name=class_info.self_var))
            
            if ctx.docstring:
                sig = sig._replace(docstring=ctx.docstring)
            
            if decorators:
                sig = sig._replace(decorators=decorators)
            processed_sigs.append(sig)

        if class_info is None:
            self._structured_functions.extend(processed_sigs)
        
        # 注册并简化签名中使用的类型
        for i, sig in enumerate(processed_sigs):
            new_args = []
            for arg in sig.args:
                if arg.type:
                    arg.type = self.strip_or_import(arg.type)
                new_args.append(arg)
            
            ret_type = sig.ret_type
            if ret_type:
                ret_type = self.strip_or_import(ret_type)
            
            processed_sigs[i] = sig._replace(args=new_args, ret_type=ret_type)
                
        return processed_sigs

    def strip_or_import(self, type_str: str) -> str:
        """解析类型字符串，注册导入，并返回简化后的名称。"""
        if type_str == "Any" or type_str == "typing.Any":
            return self.add_name("typing.Any")
        if type_str == "Incomplete" or type_str == "_typeshed.Incomplete":
            return self.add_name("_typeshed.Incomplete")

        # 处理 Union 类型
        if "|" in type_str:
            parts = [self.strip_or_import(p.strip()) for p in type_str.split("|")]
            return " | ".join(parts)
        
        # 处理泛型
        if "[" in type_str and type_str.endswith("]"):
            idx = type_str.find("[")
            base = type_str[:idx]
            inner = type_str[idx+1:-1]
            return f"{self.strip_or_import(base)}[{self.strip_or_import(inner)}]"

        if "." in type_str:
            if type_str.startswith("typing."):
                return self.add_name(type_str)
            if type_str.startswith("_typeshed."):
                return self.add_name(type_str)
            if type_str.startswith("numpy."):
                self.import_tracker.add_import("numpy", require=True)
                # 官方输出中，函数签名中的 numpy 类型通常被简化（如果已导入）
                # 但变量类型保留了 numpy. 前缀。
                # 为了简单起见，我们统一简化，除非是变量。
                return self.add_name(type_str)
        
        return type_str

    def is_staticmethod(self, class_info: ClassInfo | None, name: str, obj: object) -> bool:
        if class_info is None:
            return False
        if self.is_c_module:
            raw_lookup = getattr(class_info.cls, "__dict__", {})
            raw_value = raw_lookup.get(name, obj)
            return isinstance(raw_value, staticmethod)
        else:
            return isinstance(inspect.getattr_static(class_info.cls, name), staticmethod)

    def generate_class_stub(
        self, class_name: str, cls: type, parent_class: ClassInfo | None = None
    ) -> ClassStubData:
        raw_lookup = getattr(cls, "__dict__", {})
        items = self.get_members(cls)
        if self.resort_members:
            items = sorted(items, key=lambda x: method_name_sort_key(x[0]))
        names = {x[0] for x in items}
        attrs = []

        structured_methods: list[FunctionSig] = []
        structured_properties: list[PropertyInfo] = []
        structured_variables: list[VariableInfo] = []
        structured_classes: list[ClassStubData] = []

        class_info = ClassInfo(
            class_name, "", getattr(cls, "__doc__", None), cls, parent=parent_class
        )

        for attr, value in items:
            raw_value = raw_lookup.get(attr, value)
            if self.is_method(class_info, attr, value) or self.is_classmethod(class_info, attr, value) or attr == "__new__":
                is_new = attr == "__new__"
                if is_new:
                    if "__init__" in names:
                        continue
                    attr = "__init__"
                
                if is_new:
                    class_info.self_var = "cls"
                elif self.is_staticmethod(class_info, attr, value):
                    class_info.self_var = ""
                elif self.is_classmethod(class_info, attr, value):
                    class_info.self_var = "cls"
                else:
                    class_info.self_var = "self"

                # 捕获结构化方法
                sigs = self.generate_function_stub(attr, value, class_info=class_info)
                structured_methods.extend(sigs)

            elif self.is_property(class_info, attr, raw_value):
                self.generate_property_stub(
                    attr, raw_value, value, class_info, structured_properties
                )
            elif inspect.isclass(value) and self.is_defined_in_module(value):
                child_class_data = self.generate_class_stub(attr, value, parent_class=class_info)
                structured_classes.append(child_class_data)
            else:
                attrs.append((attr, value))

        for attr, value in attrs:
            if attr == "__hash__" and value is None:
                continue
            prop_type_name = self.get_type_annotation(value)
            classvar = self.add_name("typing.ClassVar")
            structured_variables.append(VariableInfo(name=attr, type=f"{classvar}[{prop_type_name}]"))

        bases = self.get_base_types(cls)

        class_data = ClassStubData(
            name=class_name,
            docstring=class_info.docstring,
            bases=bases,
            methods=structured_methods,
            properties=structured_properties,
            variables=structured_variables,
            classes=structured_classes,
        )
        if parent_class is None:
            self._structured_classes.append(class_data)
        return class_data

    def is_method(self, class_info: ClassInfo, name: str, obj: object) -> bool:
        if self.is_c_module:
            return inspect.ismethoddescriptor(obj) or type(obj) in (
                type(str.index), type(str.__add__), type(str.__new__),
            )
        else:
            return inspect.isfunction(obj)

    def is_classmethod(self, class_info: ClassInfo, name: str, obj: object) -> bool:
        # __new__ is implicitly a class method
        if getattr(obj, "__name__", None) == "__new__":
            return True
        if self.is_c_module:
            return inspect.isbuiltin(obj) or type(obj).__name__ in (
                "classmethod", "classmethod_descriptor",
            )
        else:
            return inspect.ismethod(obj)

    def is_property(self, class_info: ClassInfo, name: str, obj: object) -> bool:
        return inspect.isdatadescriptor(obj) or hasattr(obj, "fget")

    def generate_property_stub(
        self, name: str, raw_obj: object, obj: object,
        class_info: ClassInfo | None = None,
        structured_properties: list[PropertyInfo] | None = None,
    ) -> None:
        docstring = getattr(raw_obj, "__doc__", None)
        ctx = FunctionContext(self.module_name, name, docstring=docstring, class_info=class_info)
        static = type(raw_obj).__name__ in ("pybind11_static_property", "StaticProperty")
        readonly = hasattr(raw_obj, "fset") and getattr(raw_obj, "fset") is None

        if static:
            ret_type = self.get_type_annotation(obj)
        else:
            default_sig = self.get_default_function_sig(raw_obj, ctx)
            ret_type = default_sig.ret_type

        inferred_type = self.get_property_type(ret_type, self.sig_generators, ctx)
        if static:
            classvar = self.add_name("typing.ClassVar")
            if inferred_type is None:
                inferred_type = self.add_name("_typeshed.Incomplete")
            if structured_properties is not None:
                structured_properties.append(PropertyInfo(name=name, type=f"{classvar}[{inferred_type}]", readonly=False))
        else:
            if readonly:
                if structured_properties is not None:
                    structured_properties.append(PropertyInfo(name=name, type=inferred_type or "Any", readonly=True))
            else:
                if inferred_type is None:
                    inferred_type = self.add_name("_typeshed.Incomplete")
                if structured_properties is not None:
                    structured_properties.append(PropertyInfo(name=name, type=inferred_type, readonly=False))

    def get_base_types(self, obj: type) -> list[str]:
        all_bases = type.mro(obj)
        if all_bases[-1] is object:
            del all_bases[-1]
        if all_bases and all_bases[-1].__name__ == "pybind11_object":
            del all_bases[-1]
        all_bases = all_bases[1:]
        bases = []
        for base in all_bases:
            if not any(issubclass(b, base) for b in bases):
                bases.append(base)
        return [self.get_type_fullname(base) for base in bases]

    def generate_variable_stub(self, name: str, obj: object) -> None:
        type_str = self.get_type_annotation(obj)
        self._structured_variables.append(VariableInfo(name=name, type=type_str))
