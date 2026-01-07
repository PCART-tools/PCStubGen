from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stubdoc import FunctionSig

@dataclass
class VariableInfo:
    name: str
    type: str

@dataclass
class PropertyInfo:
    name: str
    type: str
    readonly: bool

@dataclass
class ClassStubData:
    name: str
    docstring: str | None = None
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionSig] = field(default_factory=list)
    properties: list[PropertyInfo] = field(default_factory=list)
    variables: list[VariableInfo] = field(default_factory=list)
    classes: list[ClassStubData] = field(default_factory=list)

@dataclass
class ModuleStubData:
    name: str
    docstring: str | None = None
    _all_: list[str] | None = None
    imports: list[str] = field(default_factory=list)
    variables: list[VariableInfo] = field(default_factory=list)
    functions: list[FunctionSig] = field(default_factory=list)
    classes: list[ClassStubData] = field(default_factory=list)
    submodules: list[ModuleStubData] = field(default_factory=list)
