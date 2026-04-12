from __future__ import annotations

import typer

from ._build_command import _build_command
from ._gen_command import _gen_command

app = typer.Typer(add_completion=False)
app.command("build")(_build_command)
app.command("gen")(_gen_command)


if __name__ == "__main__":
    app()
