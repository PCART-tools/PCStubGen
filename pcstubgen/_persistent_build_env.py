from __future__ import annotations

import os
import sys
from pathlib import Path

from build import env as build_env
from build.env import DefaultIsolatedEnv


class PersistentIsolatedEnv(DefaultIsolatedEnv):
    """在项目目录下维护可复用的持久隔离构建环境。"""

    BUILD_ENV_DIRNAME = ".pcstubgen-build-venv"

    def __init__(
        self,
        srcdir: Path,
        *,
        installer: build_env.Installer = "pip",
    ) -> None:
        super().__init__(installer=installer)
        self._srcdir = srcdir

    @staticmethod
    def get_build_env_path(srcdir: Path) -> Path:
        return srcdir / PersistentIsolatedEnv.BUILD_ENV_DIRNAME

    def __enter__(self) -> PersistentIsolatedEnv:
        try:
            path = self.get_build_env_path(self._srcdir).resolve()
            # 与 DefaultIsolatedEnv 保持一致，统一真实路径表示。
            self._path = os.path.realpath(path)

            self._env_backend: build_env._EnvBackend

            if self.installer == "uv":
                self._env_backend = build_env._UvBackend()
            else:
                self._env_backend = build_env._PipBackend()

            if os.path.exists(self._path):
                try:
                    python_executable, scripts_dir, _ = build_env._find_executable_and_scripts(
                        self._path
                    )
                except Exception as ex:
                    raise RuntimeError(
                        f"无效持久构建环境: {self._path}。可使用 --clean-env 重新创建。"
                    ) from ex
                self._env_backend.python_executable = python_executable
                self._env_backend.scripts_dir = scripts_dir
            else:
                build_env._ctx.log(
                    f"Creating isolated environment: {self._env_backend.display_name}..."
                )
                self._env_backend.create(self._path)
        except Exception:
            self.__exit__(*sys.exc_info())
            raise

        return self

    def __exit__(self, *args: object) -> None:
        _ = args
