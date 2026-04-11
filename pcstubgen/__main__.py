from __future__ import annotations

import typer

from ._build_command import BUILD_COMMAND_HELP, _build_command
from ._gen_command import _gen_command

app = typer.Typer(add_completion=False)
app.command("gen", help="使用 pcstubgen 为模块生成 Python stub。")(_gen_command)
app.command("build", help=BUILD_COMMAND_HELP)(_build_command)


if __name__ == "__main__":
    app()
