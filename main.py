import dataclasses
import tomllib

import clang
import click
from clang.cindex import CursorKind, TypeKind
import os

@dataclasses.dataclass()
class Config:
    clang_library_path: str
    clang_parse_args: list[str]
    c_filename: str

@dataclasses.dataclass()
class ParameterInfo:
    name: str
    type: str

@dataclasses.dataclass()
class FieldInfo:
    name: str
    type: str

@dataclasses.dataclass()
class FunctionInfo:
    name: str
    return_type: str
    params: list[ParameterInfo]
    location: tuple[int, int]
    file: str | None

@dataclasses.dataclass()
class StructInfo:
    name: str
    fields: list[FieldInfo]
    location: tuple[int, int]
    file: str | None

@dataclasses.dataclass()
class VariableInfo:
    name: str
    type: str
    location: tuple[int, int]
    file: str | None


# 递归打印AST节点
def print_node(cursor, indent=0):
    """递归打印AST节点"""
    kind = cursor.kind
    type_name = cursor.type.spelling if cursor.type else ""

    # 打印当前节点信息
    indent_str = "  " * indent
    str = f"{indent_str}[{kind.name}]"
    if cursor.spelling != "" or type_name != "":
        str += f" {cursor.spelling}: {type_name}"
    click.echo(str)

    # # 如果是函数声明，打印参数信息
    # if kind == CursorKind.FUNCTION_DECL:
    #     for arg in cursor.get_arguments():
    #         click.echo(f"{indent_str}  参数: {arg.spelling} : {arg.type.spelling}")
    #
    # # 如果是结构体或联合体，打印字段信息
    # elif kind in [CursorKind.STRUCT_DECL, CursorKind.UNION_DECL]:
    #     for child in cursor.get_children():
    #         if child.kind == CursorKind.FIELD_DECL:
    #             click.echo(f"{indent_str}  字段: {child.spelling} : {child.type.spelling}")

    # 递归处理子节点
    for child in cursor.get_children():
        print_node(child, indent + 1)


def extract_functions(cursor, functions=None):
    """提取所有函数信息"""
    if functions is None:
        functions = []

    if cursor.kind == CursorKind.FUNCTION_DECL:
        func_info = FunctionInfo(
            name=cursor.spelling,
            return_type=cursor.result_type.spelling,
            params=[],
            location=(cursor.location.line, cursor.location.column),
            file=cursor.location.file.name if cursor.location.file else None
        )

        for arg in cursor.get_arguments():
            param_info = ParameterInfo(
                name=arg.spelling,
                type=arg.type.spelling
            )
            func_info.params.append(param_info)

        functions.append(func_info)

    for child in cursor.get_children():
        extract_functions(child, functions)

    return functions


def extract_structs(cursor, structs=None):
    """提取所有结构体信息"""
    if structs is None:
        structs = []

    if cursor.kind == CursorKind.STRUCT_DECL:
        struct_info = StructInfo(
            name=cursor.spelling,
            fields=[],
            location=(cursor.location.line, cursor.location.column),
            file=cursor.location.file.name if cursor.location.file else None
        )

        for child in cursor.get_children():
            if child.kind == CursorKind.FIELD_DECL:
                field_info = FieldInfo(
                    name=child.spelling,
                    type=child.type.spelling
                )
                struct_info.fields.append(field_info)

        structs.append(struct_info)

    for child in cursor.get_children():
        extract_structs(child, structs)

    return structs


def extract_variables(cursor, variables=None):
    """提取全局变量信息"""
    if variables is None:
        variables = []

    if cursor.kind == CursorKind.VAR_DECL:
        # 检查是否为全局变量（在文件作用域中）
        if cursor.semantic_parent.kind == CursorKind.TRANSLATION_UNIT:
            var_info = VariableInfo(
                name=cursor.spelling,
                type=cursor.type.spelling,
                location=(cursor.location.line, cursor.location.column),
                file=cursor.location.file.name if cursor.location.file else None
            )
            variables.append(var_info)

    for child in cursor.get_children():
        extract_variables(child, variables)

    return variables


def parse_c_file(config: Config) -> tuple[clang.cindex.TranslationUnit, clang.cindex.Cursor]:
    # 创建索引
    index = clang.cindex.Index.create()

    # 解析翻译单元
    translation_unit = index.parse(
        config.c_filename,
        args=config.clang_parse_args
    )

    if not translation_unit:
        click.echo(f"错误：无法解析文件 '{config.c_filename}'", err=True)
        return None

    # 检查解析错误
    diagnostics = list(translation_unit.diagnostics)
    if diagnostics:
        click.echo("解析警告/错误:", err=True)
        for diag in diagnostics:
            click.echo(f"  {diag.spelling}", err=True)

    # 获取根游标
    cursor = translation_unit.cursor

    return translation_unit, cursor


@click.command()
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
def main(config_path):
    with open(config_path, 'rb') as f:
        data = tomllib.load(f)
        config = Config(**data)

    # 设置 libclang 库路径
    clang.cindex.Config.set_library_file(config.clang_library_path)

    click.echo(f"正在解析文件: {config.c_filename}")

    # 解析 C 文件
    result = parse_c_file(config)
    if not result:
        return

    translation_unit, cursor = result

    # 输出AST
    click.echo("\n完整AST结构:")
    print_node(cursor)

    # 输出函数列表
    click.echo("\n函数列表:")
    functions_list = extract_functions(cursor)
    if functions_list:
        for func in functions_list:
            params = ', '.join([f"{p.type} {p.name}" for p in func.params])
            click.echo(f"{func.return_type} {func.name}({params})")
    else:
        click.echo("未找到函数")

    # 输出结构体列表
    click.echo("\n结构体列表:")
    structs_list = extract_structs(cursor)
    if structs_list:
        for struct in structs_list:
            click.echo(f"struct {struct.name}")
            for field in struct.fields:
                click.echo(f"    {field.type} {field.name}")
    else:
        click.echo("未找到结构体")

    # 输出全局变量列表
    click.echo("\n全局变量列表:")
    variables_list = extract_variables(cursor)
    if variables_list:
        for var in variables_list:
            click.echo(f"{var.type} {var.name}")
    else:
        click.echo("未找到全局变量")

    # 统计信息
    functions_count = len(functions_list)
    structs_count = len(structs_list)
    variables_count = len(variables_list)
    click.echo(f"\n统计: {functions_count} 个函数, {structs_count} 个结构体, {variables_count} 个全局变量")


if __name__ == "__main__":
    main()