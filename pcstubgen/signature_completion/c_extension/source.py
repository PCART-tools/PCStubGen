from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from clang.cindex import Cursor, CursorKind, Index, TranslationUnit, TranslationUnitLoadError
from loguru import logger

from ...ir_modules import IRArgument, IRFunction, IRModule, IRModuleType, IRSignature
from ...types import AnyType, Type
from .clang import parser as clang_parser
from .clang.cursor_utils import source_range_get_text, walk_cursor
from .models import CArgument, CFunction, CSignature
from .modules.method_flags import METH_FASTCALL, METH_KEYWORDS, METH_NOARGS, METH_O, METH_VARARGS
from .runtime import resolve_runtime_pymethoddef
from .signatures import inference
from .address_resolver import SymbolizedAddressLocation, resolve_symbolized_address
from ...ir_modules import IRArgumentKind

ResolvedCExtensionFunction: TypeAlias = tuple[list[IRSignature], str | None]


class CExtensionSource:
    def __init__(
        self,
        *,
        compilation_database: Path | None = None,
    ) -> None:
        self._compilation_database = compilation_database
        self._compilation_commands: list[clang_parser.CompilationCommand] | None = None
        self._translation_units: dict[Path, TranslationUnit] = {}
        self._index: Index | None = None

    def resolve_function(
        self,
        irmodule: IRModule,
        irfunction: IRFunction,
    ) -> ResolvedCExtensionFunction:
        """按函数懒解析 C 扩展签名。"""
        if irmodule.module_type is not IRModuleType.EXTENSION:
            raise RuntimeError(f"模块 {irmodule.full_name} 不是扩展模块。")
        if irfunction.runtime_handle is None:
            raise RuntimeError(f"函数 {irfunction.name} 缺少运行时对象引用。")

        runtime_method = resolve_runtime_pymethoddef(irfunction.runtime_handle)
        location = resolve_symbolized_address(runtime_method.method_address)
        c_function = CFunction(
            ml_name=runtime_method.name,
            ml_flags=runtime_method.flags,
            ml_meth_address=runtime_method.method_address,
            library_path=location.binary_path,
            source_path=location.function_start_path,
            source_line=location.function_start_line,
            symbol_name=location.function_name,
        )

        function_cursor = self._resolve_function_cursor(location=location, c_function=c_function)
        if function_cursor is not None:
            c_function.function_cursor = function_cursor
            signatures = inference.infer_signature(c_function)
        else:
            signatures = self._infer_minimal_signatures(c_function)
        if function_cursor is not None and not signatures:
            signatures = self._infer_minimal_signatures(c_function)

        if not signatures:
            raise RuntimeError(f"C函数 {irmodule.full_name}.{irfunction.name} 没有可用签名")

        return (
            [self._build_ir_signature(signature.arguments, signature.return_type) for signature in signatures],
            self._get_source_comment(c_function),
        )

    def _resolve_function_cursor(
        self,
        *,
        location: SymbolizedAddressLocation,
        c_function: CFunction,
    ) -> Cursor | None:
        """按需 parse 已定位到的源码文件，并找到对应的函数 cursor。"""
        if self._compilation_database is None:
            return None

        source_path_candidates = self._source_path_candidates(location)
        if len(source_path_candidates) == 0:
            return None

        compilation_command = self._match_compilation_command(source_path_candidates)
        if compilation_command is None:
            logger.info(
                "未在编译数据库中找到源码文件, source_paths: {}",
                ", ".join(str(path) for path in source_path_candidates),
            )
            return None

        translation_unit = self._load_translation_unit(compilation_command)
        if translation_unit is None:
            return None

        matched = self._find_function_cursor(
            translation_unit=translation_unit,
            source_path=compilation_command.file_path,
            line_candidates=[
                location.function_start_line,
                location.resolved_line,
            ],
            symbol_candidates=[c_function.symbol_name, c_function.ml_name],
        )
        if matched is None:
            logger.info(
                "未在 translation unit 中定位到函数定义, source_path: {}, ml_name: {}, symbol_name: {}",
                compilation_command.file_path,
                c_function.ml_name,
                c_function.symbol_name,
            )
        return matched

    def _match_compilation_command(
        self,
        source_paths: list[Path],
    ) -> clang_parser.CompilationCommand | None:
        commands = self._get_compilation_commands()
        resolved_source_paths = {source_path.resolve() for source_path in source_paths}
        for command in commands:
            if command.file_path in resolved_source_paths:
                return command
        return None

    def _get_compilation_commands(self) -> list[clang_parser.CompilationCommand]:
        if self._compilation_commands is None:
            compilation_database = self._compilation_database
            if compilation_database is None:
                raise RuntimeError("缺少 compile_commands.json，无法加载编译数据库。")
            self._compilation_commands = clang_parser.list_compilation_commands(
                compilation_database
            )
        compilation_commands = self._compilation_commands
        if compilation_commands is None:
            raise RuntimeError("编译数据库加载失败。")
        return compilation_commands

    def _load_translation_unit(
        self,
        compilation_command: clang_parser.CompilationCommand,
    ) -> TranslationUnit | None:
        cached = self._translation_units.get(compilation_command.file_path)
        if cached is not None:
            return cached

        if self._index is None:
            self._index = Index.create()
        index = self._index
        if index is None:
            raise RuntimeError("libclang Index 初始化失败。")

        effective_parse_args = clang_parser.build_effective_parse_args(compilation_command)
        try:
            translation_unit = clang_parser.parse(
                index,
                compilation_command,
                effective_parse_args=effective_parse_args,
            )
        except TranslationUnitLoadError:
            logger.warning(
                "按需Parse失败, 文件路径: {}, 工作目录: {}, 解析参数: {}",
                compilation_command.file_path,
                compilation_command.working_directory,
                " ".join(effective_parse_args),
            )
            return None

        diagnostics = translation_unit.diagnostics
        if clang_parser.has_error_diagnostics(diagnostics):
            logger.warning(
                "按需Parse诊断, 文件路径: {}, 诊断: {}",
                compilation_command.file_path,
                "\n".join(
                    clang_parser.diagnostic_to_str(diagnostic)
                    for diagnostic in diagnostics
                ),
            )

        self._translation_units[compilation_command.file_path] = translation_unit
        return translation_unit

    @staticmethod
    def _find_function_cursor(
        *,
        translation_unit: TranslationUnit,
        source_path: Path,
        line_candidates: list[int],
        symbol_candidates: list[str],
    ) -> Cursor | None:
        normalized_source_path = source_path.resolve()
        available_symbols = {
            symbol
            for symbol in symbol_candidates
            if symbol != ""
        }

        line_matches: list[Cursor] = []
        symbol_matches: list[Cursor] = []
        for cursor in walk_cursor(translation_unit.cursor):
            if cursor.kind != CursorKind.FUNCTION_DECL:
                continue
            location_file = cursor.location.file
            if location_file is None or Path(location_file.name).resolve() != normalized_source_path:
                continue

            if _cursor_covers_any_line(cursor, line_candidates):
                line_matches.append(cursor)

            cursor_names = {
                name
                for name in (
                    cursor.spelling,
                    cursor.mangled_name,
                    cursor.displayname,
                )
                if name
            }
            if available_symbols and cursor_names & available_symbols:
                symbol_matches.append(cursor)

        if len(line_matches) == 1:
            return line_matches[0]
        if len(symbol_matches) == 1:
            return symbol_matches[0]
        if line_matches:
            return line_matches[0]
        if symbol_matches:
            return symbol_matches[0]
        return None

    @staticmethod
    def _source_path_candidates(location: SymbolizedAddressLocation) -> list[Path]:
        result: list[Path] = []
        for path in (
            location.function_start_path,
            location.resolved_path,
        ):
            if path in result:
                continue
            result.append(path)
        return result

    @staticmethod
    def _infer_minimal_signatures(c_function: CFunction) -> list[CSignature]:
        """无法进入 AST 深推断时，根据 `ml_flags` 构造最小签名。"""
        flags = c_function.ml_flags
        if flags & METH_NOARGS:
            return [inference.CSignature(arguments=[])]

        if flags & METH_O:
            return [
                inference.CSignature(
                    arguments=[
                        CArgument(
                            name="arg",
                            type=AnyType(),
                            kind=IRArgumentKind.POSITIONAL_ONLY,
                        )
                    ]
                )
            ]

        if flags & (METH_VARARGS | METH_FASTCALL):
            arguments = [
                CArgument(
                    name="args",
                    type=AnyType(),
                    kind=IRArgumentKind.VAR_POSITIONAL,
                )
            ]
            if flags & METH_KEYWORDS:
                arguments.append(
                    CArgument(
                        name="kwargs",
                        type=AnyType(),
                        kind=IRArgumentKind.VAR_KEYWORD,
                    )
                )
            return [inference.CSignature(arguments=arguments)]

        return []

    @staticmethod
    def _build_ir_signature(
        arguments: list[CArgument],
        return_type: Type | None,
    ) -> IRSignature:
        return IRSignature(
            args=[
                IRArgument(
                    name=argument.name,
                    type=argument.type,
                    default_value=argument.default_value,
                    has_default=argument.has_default,
                    kind=argument.kind,
                )
                for argument in arguments
            ],
            return_type=return_type,
        )

    @staticmethod
    def _get_source_comment(c_function: CFunction) -> str | None:
        function_cursor = c_function.function_cursor
        if function_cursor is None or function_cursor.extent is None:
            return None
        source_text = source_range_get_text(function_cursor.extent)
        if not source_text:
            return None
        return source_text


def _cursor_covers_any_line(cursor: Cursor, lines: list[int]) -> bool:
    extent = cursor.extent
    if extent is None:
        return False
    start_line = int(extent.start.line)
    end_line = int(extent.end.line)
    return any(start_line <= line <= end_line for line in lines)
