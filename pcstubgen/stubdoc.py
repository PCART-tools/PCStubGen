from __future__ import annotations

import contextlib
import io
import keyword
import re
import tokenize
from collections.abc import Sequence
from typing import Any, Final, NamedTuple, TypeAlias as _TypeAlias

from .utils import quote_docstring

# 用于格式为 ('func_name', '(arg, opt_arg=False)') 的签名字符串的类型别名。
Sig: _TypeAlias = tuple[str, str]

_TYPE_RE: Final = re.compile(r"^[a-zA-Z_][\w\[\], .\"\'|]*(\.[a-zA-Z_][\w\[\], ]*)*$")
_ARG_NAME_RE: Final = re.compile(r"\**[A-Za-z_][A-Za-z0-9_]*$")

def is_valid_type(s: str) -> bool:
    """尝试判断字符串是否可能是有效的类型注解。"""
    if s in ("True", "False", "retval"):
        return False
    if "," in s and "[" not in s:
        return False
    return _TYPE_RE.match(s) is not None

class ArgSig:
    """单个参数的签名信息。"""

    def __init__(
        self,
        name: str,
        type: str | None = None,
        *,
        default: bool = False,
        default_value: str = "...",
    ) -> None:
        self.name = name
        self.type = type
        # 该参数是否有默认值？
        self.default = default
        self.default_value = default_value

    def is_star_arg(self) -> bool:
        return self.name.startswith("*") and not self.name.startswith("**")

    def is_star_kwarg(self) -> bool:
        return self.name.startswith("**")

    def __repr__(self) -> str:
        return "ArgSig(name={}, type={}, default={})".format(
            repr(self.name), repr(self.type), repr(self.default)
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ArgSig):
            return (
                self.name == other.name
                and self.type == other.type
                and self.default == other.default
                and self.default_value == other.default_value
            )
        return False

class FunctionSig(NamedTuple):
    name: str
    args: list[ArgSig]
    ret_type: str | None
    type_args: str = ""
    docstring: str | None = None
    decorators: list[str] = []

    def is_special_method(self) -> bool:
        return bool(
            self.name.startswith("__")
            and self.name.endswith("__")
            and self.args
            and self.args[0].name in ("self", "cls")
        )

    def format_sig(
        self,
        indent: str = "",
        is_async: bool = False,
        any_val: str | None = None,
        docstring: str | None = None,
        include_docstrings: bool = False,
    ) -> str:
        args: list[str] = []
        for arg in self.args:
            arg_def = arg.name

            if arg_def in keyword.kwlist:
                arg_def = "_" + arg_def

            if (
                arg.type is None
                and any_val is not None
                and arg.name not in ("self", "cls")
                and not arg.name.startswith("*")
            ):
                arg_type: str | None = any_val
            else:
                arg_type = arg.type
            if arg_type:
                arg_def += ": " + arg_type
                if arg.default:
                    arg_def += f" = {arg.default_value}"

            elif arg.default:
                arg_def += f"={arg.default_value}"

            args.append(arg_def)

        retfield = ""
        ret_type = self.ret_type if self.ret_type else any_val
        if ret_type is not None:
            retfield = " -> " + ret_type

        prefix = "async " if is_async else ""
        
        decorator_lines = ""
        for dec in self.decorators:
            decorator_lines += f"{indent}{dec}\n"
            
        sig = f"{indent}{prefix}def {self.name}{self.type_args}({', '.join(args)}){retfield}:"
        doc = (self.docstring or docstring) if include_docstrings else None
        if doc:
            suffix = f"\n{indent}    {quote_docstring(doc)}"
        else:
            suffix = " ..."
        return f"{decorator_lines}{sig}{suffix}"

# 文档字符串解析器的状态。
STATE_INIT: Final = 1
STATE_FUNCTION_NAME: Final = 2
STATE_ARGUMENT_LIST: Final = 3
STATE_ARGUMENT_TYPE: Final = 4
STATE_ARGUMENT_DEFAULT: Final = 5
STATE_RETURN_VALUE: Final = 6
STATE_OPEN_BRACKET: Final = 7

class DocStringParser:
    """解析文档中的函数签名。"""

    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        self.state = [STATE_INIT]
        self.accumulator = ""
        self.arg_type: str | None = None
        self.arg_name = ""
        self.arg_default: str | None = None
        self.ret_type = "Any"
        self.found = False
        self.args: list[ArgSig] = []
        self.pos_only: int | None = None
        self.keyword_only: int | None = None
        self.signatures: list[FunctionSig] = []

    def add_token(self, token: tokenize.TokenInfo) -> None:
        if (
            token.type == tokenize.NAME
            and token.string == self.function_name
            and self.state[-1] == STATE_INIT
        ):
            self.state.append(STATE_FUNCTION_NAME)

        elif (
            token.type == tokenize.OP
            and token.string == "("
            and self.state[-1] == STATE_FUNCTION_NAME
        ):
            self.state.pop()
            self.accumulator = ""
            self.found = True
            self.state.append(STATE_ARGUMENT_LIST)

        elif self.state[-1] == STATE_FUNCTION_NAME:
            self.state.pop()

        elif (
            token.type == tokenize.OP
            and token.string in ("[", "(", "{")
            and self.state[-1] != STATE_INIT
        ):
            self.accumulator += token.string
            self.state.append(STATE_OPEN_BRACKET)

        elif (
            token.type == tokenize.OP
            and token.string in ("]", ")", "}")
            and self.state[-1] == STATE_OPEN_BRACKET
        ):
            self.accumulator += token.string
            self.state.pop()

        elif (
            token.type == tokenize.OP
            and token.string == ":"
            and self.state[-1] == STATE_ARGUMENT_LIST
        ):
            self.arg_name = self.accumulator
            self.accumulator = ""
            self.state.append(STATE_ARGUMENT_TYPE)

        elif (
            token.type == tokenize.OP
            and token.string == ":"
            and self.state[-1] == STATE_ARGUMENT_TYPE
            and self.accumulator == ""
        ):
            self.reset()

        elif (
            token.type == tokenize.OP
            and token.string == "="
            and self.state[-1] in (STATE_ARGUMENT_LIST, STATE_ARGUMENT_TYPE)
        ):
            if self.state[-1] == STATE_ARGUMENT_TYPE:
                self.arg_type = self.accumulator
                self.state.pop()
            else:
                self.arg_name = self.accumulator
            self.accumulator = ""
            self.state.append(STATE_ARGUMENT_DEFAULT)

        elif (
            token.type == tokenize.OP
            and token.string in (",", ")")
            and self.state[-1]
            in (STATE_ARGUMENT_LIST, STATE_ARGUMENT_DEFAULT, STATE_ARGUMENT_TYPE)
        ):
            if self.state[-1] == STATE_ARGUMENT_DEFAULT:
                self.arg_default = self.accumulator
                self.state.pop()
            elif self.state[-1] == STATE_ARGUMENT_TYPE:
                self.arg_type = self.accumulator
                self.state.pop()
            elif self.state[-1] == STATE_ARGUMENT_LIST:
                if self.accumulator == "*":
                    if self.keyword_only is not None:
                        self.reset()
                        return
                    self.keyword_only = len(self.args)
                    self.accumulator = ""
                else:
                    if self.accumulator.startswith("*"):
                        self.keyword_only = len(self.args) + 1
                    self.arg_name = self.accumulator
                    if not (
                        token.string == ")" and self.accumulator.strip() == ""
                    ) and not _ARG_NAME_RE.match(self.arg_name):
                        self.reset()
                        return

            if token.string == ")":
                if (
                    self.state[-1] == STATE_ARGUMENT_LIST
                    and self.keyword_only is not None
                    and self.keyword_only == len(self.args)
                    and not self.arg_name
                ):
                    self.reset()
                    return
                self.state.pop()

            if self.arg_name:
                if self.arg_type and not is_valid_type(self.arg_type):
                    self.args.append(
                        ArgSig(name=self.arg_name, type=None, default=bool(self.arg_default))
                    )
                else:
                    self.args.append(
                        ArgSig(
                            name=self.arg_name, type=self.arg_type, default=bool(self.arg_default)
                        )
                    )
            self.arg_name = ""
            self.arg_type = None
            self.arg_default = None
            self.accumulator = ""
        elif (
            token.type == tokenize.OP
            and token.string == "/"
            and self.state[-1] == STATE_ARGUMENT_LIST
        ):
            if self.pos_only is not None or self.keyword_only is not None or not self.args:
                self.reset()
                return
            self.pos_only = len(self.args)
            self.state.append(STATE_ARGUMENT_TYPE)
            self.accumulator = ""

        elif token.type == tokenize.OP and token.string == "->" and self.state[-1] == STATE_INIT:
            self.accumulator = ""
            self.state.append(STATE_RETURN_VALUE)

        elif token.type in (tokenize.NEWLINE, tokenize.ENDMARKER) and self.state[-1] in (
            STATE_INIT,
            STATE_RETURN_VALUE,
        ):
            if self.state[-1] == STATE_RETURN_VALUE:
                if not is_valid_type(self.accumulator):
                    self.reset()
                    return
                self.ret_type = self.accumulator
                self.accumulator = ""
                self.state.pop()

            if self.found:
                self.signatures.append(
                    FunctionSig(name=self.function_name, args=self.args, ret_type=self.ret_type)
                )
                self.found = False
            self.args = []
            self.ret_type = "Any"
        else:
            self.accumulator += token.string

    def reset(self) -> None:
        self.state = [STATE_INIT]
        self.args = []
        self.found = False
        self.accumulator = ""

    def get_signatures(self) -> list[FunctionSig]:
        def has_arg(name: str, signature: FunctionSig) -> bool:
            return any(x.name == name for x in signature.args)

        def args_kwargs(signature: FunctionSig) -> bool:
            return has_arg("*args", signature) and has_arg("**kwargs", signature)

        return sorted(self.signatures, key=lambda x: 1 if args_kwargs(x) else 0)

def infer_sig_from_docstring(docstr: str | None, name: str) -> list[FunctionSig] | None:
    if not (isinstance(docstr, str) and docstr):
        return None

    state = DocStringParser(name)
    with contextlib.suppress(tokenize.TokenError):
        try:
            tokens = tokenize.tokenize(io.BytesIO(docstr.encode("utf-8")).readline)
            for token in tokens:
                state.add_token(token)
        except IndentationError:
            return None
    sigs = state.get_signatures()

    def is_unique_args(sig: FunctionSig) -> bool:
        return len(sig.args) == len({arg.name for arg in sig.args})

    return [sig for sig in sigs if is_unique_args(sig)]

def infer_ret_type_sig_from_anon_docstring(docstr: str) -> str | None:
    lines = ["stub" + line.strip() for line in docstr.splitlines() if line.strip().startswith("(")]
    return infer_ret_type_sig_from_docstring("".join(lines), "stub")

def infer_ret_type_sig_from_docstring(docstr: str, name: str) -> str | None:
    ret = infer_sig_from_docstring(docstr, name)
    if ret:
        return ret[0].ret_type
    return None

def infer_prop_type_from_docstring(docstr: str | None) -> str | None:
    if not docstr:
        return None
    test_str = r"^([a-zA-Z0-9_, \.\[\]]*): "
    m = re.match(test_str, docstr)
    return m.group(1) if m else None

def infer_arg_sig_from_anon_docstring(docstr: str) -> list[ArgSig]:
    ret = infer_sig_from_docstring("stub" + docstr, "stub")
    if ret:
        return ret[0].args
    return []

def infer_method_ret_type(name: str) -> str | None:
    if name.startswith("__") and name.endswith("__"):
        name = name[2:-2]
        if name in ("float", "bool", "bytes", "int", "complex", "str"):
            return name
        elif name in ("eq", "ne", "lt", "le", "gt", "ge", "contains"):
            return "bool"
        elif name in ("len", "length_hint", "index", "hash", "sizeof", "trunc", "floor", "ceil"):
            return "int"
        elif name in ("format", "repr"):
            return "str"
        elif name in ("init", "setitem", "del", "delitem"):
            return "None"
    return None

def infer_method_arg_types(
    name: str, self_var: str = "self", arg_names: list[str] | None = None
) -> list[ArgSig] | None:
    args: list[ArgSig] | None = None
    if name.startswith("__") and name.endswith("__"):
        if arg_names and len(arg_names) >= 1 and arg_names[0] == "self":
            arg_names = arg_names[1:]

        name = name[2:-2]
        if name == "exit":
            if arg_names is None:
                arg_names = ["type", "value", "traceback"]
            if len(arg_names) == 3:
                arg_types = [
                    "type[BaseException] | None",
                    "BaseException | None",
                    "types.TracebackType | None",
                ]
                args = [
                    ArgSig(name=arg_name, type=arg_type)
                    for arg_name, arg_type in zip(arg_names, arg_types)
                ]
    if args is not None:
        return [ArgSig(name=self_var)] + args
    return None

def infer_c_method_args(name: str, self_var: str = "self") -> list[ArgSig]:
    args = None
    if name.startswith("__") and name.endswith("__"):
        name = name[2:-2]
        if name in ("hash", "iter", "next", "sizeof", "copy", "deepcopy", "reduce", "getinitargs", "int", "float", "trunc", "complex", "bool", "abs", "bytes", "dir", "len", "reversed", "round", "index", "enter"):
            args = []
        elif name == "getitem":
            args = [ArgSig(name="index")]
        elif name == "setitem":
            args = [ArgSig(name="index"), ArgSig(name="object")]
        elif name in ("delattr", "getattr"):
            args = [ArgSig(name="name")]
        elif name == "setattr":
            args = [ArgSig(name="name"), ArgSig(name="value")]
        elif name in ("eq", "ne", "lt", "le", "gt", "ge"):
            args = [ArgSig(name="other", type="object")]
        elif name in ("add", "radd", "sub", "rsub", "mul", "rmul", "mod", "rmod", "floordiv", "rfloordiv", "truediv", "rtruediv", "divmod", "rdivmod", "pow", "rpow", "xor", "rxor", "or", "ror", "and", "rand", "lshift", "rlshift", "rshift", "rrshift", "contains", "delitem", "iadd", "iand", "ifloordiv", "ilshift", "imod", "imul", "ior", "ipow", "irshift", "isub", "itruediv", "ixor"):
            args = [ArgSig(name="other")]
        elif name in ("neg", "pos", "invert"):
            args = []
        elif name == "exit":
            args = [
                ArgSig(name="type", type="type[BaseException] | None"),
                ArgSig(name="value", type="BaseException | None"),
                ArgSig(name="traceback", type="types.TracebackType | None"),
            ]
    if args is None:
        args = [ArgSig(name="*args"), ArgSig(name="**kwargs")]
    else:
        args = [ArgSig(name=self_var)] + args
    return args
