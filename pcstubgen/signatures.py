from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from .stubdoc import (
    infer_prop_type_from_docstring,
    infer_ret_type_sig_from_anon_docstring,
    infer_ret_type_sig_from_docstring,
    infer_sig_from_docstring,
)

if TYPE_CHECKING:
    from .stubdoc import FunctionSig
    from .base_stub_generator import FunctionContext

class SignatureGenerator(metaclass=abc.ABCMeta):
    def remove_self_type(
        self, inferred: list[FunctionSig] | None, self_var: str
    ) -> list[FunctionSig] | None:
        if inferred:
            for signature in inferred:
                if signature.args:
                    if signature.args[0].name == self_var:
                        signature.args[0].type = None
        return inferred

    @abc.abstractmethod
    def get_function_sig(
        self, default_sig: FunctionSig, ctx: FunctionContext
    ) -> list[FunctionSig] | None:
        pass

    @abc.abstractmethod
    def get_property_type(self, default_type: str | None, ctx: FunctionContext) -> str | None:
        pass

class DocstringSignatureGenerator(SignatureGenerator):
    def get_function_sig(
        self, default_sig: FunctionSig, ctx: FunctionContext
    ) -> list[FunctionSig] | None:
        inferred = infer_sig_from_docstring(ctx.docstring, ctx.name)
        if ctx.class_info:
            if not inferred and ctx.name == "__init__":
                inferred = infer_sig_from_docstring(ctx.class_info.docstring, ctx.class_info.name)
                if inferred:
                    inferred = [sig._replace(name="__init__") for sig in inferred]
            return self.remove_self_type(inferred, ctx.class_info.self_var)
        else:
            return inferred

    def get_property_type(self, default_type: str | None, ctx: FunctionContext) -> str | None:
        if ctx.docstring is not None:
            inferred = infer_ret_type_sig_from_anon_docstring(ctx.docstring)
            if inferred:
                return inferred
            inferred = infer_ret_type_sig_from_docstring(ctx.docstring, ctx.name)
            if inferred:
                return inferred
            inferred = infer_prop_type_from_docstring(ctx.docstring)
            return inferred
        else:
            return None
