from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from build import env as build_env
from build.env import DefaultIsolatedEnv


class PersistentIsolatedEnv(DefaultIsolatedEnv):
    """在项目目录下维护固定路径、每次重建的隔离构建环境。"""

    BUILD_ENV_DIRNAME = ".pcstubgen-build-env"
    BUILD_ENV_GITIGNORE_CONTENT = "*\n"

    def __init__(
        self,
        srcdir: Path,
        *,
        installer: build_env.Installer = "pip",
    ) -> None:
        """初始化与项目目录绑定的隔离构建环境。"""
        super().__init__(installer=installer)
        self._srcdir = srcdir

    @staticmethod
    def get_build_env_path(srcdir: Path) -> Path:
        """返回项目内固定的构建环境目录。"""
        return srcdir / PersistentIsolatedEnv.BUILD_ENV_DIRNAME

    @staticmethod
    def write_gitignore(build_env_path: Path) -> None:
        """写入构建环境目录的 Git 忽略规则。"""
        gitignore_path = build_env_path / ".gitignore"
        gitignore_path.write_text(
            PersistentIsolatedEnv.BUILD_ENV_GITIGNORE_CONTENT,
            encoding="utf-8",
        )

    def __enter__(self) -> PersistentIsolatedEnv:
        """删除旧环境后，在固定路径创建新的隔离构建环境。"""
        try:
            path = self.get_build_env_path(self._srcdir).resolve()
            self._path = os.path.realpath(path)

            self._env_backend: build_env._EnvBackend
            if self.installer == "uv":
                self._env_backend = build_env._UvBackend()
            else:
                self._env_backend = build_env._PipBackend()

            if os.path.exists(self._path):
                if os.path.islink(self._path) or not os.path.isdir(self._path):
                    raise RuntimeError(
                        f"构建环境路径存在但不是可清理目录: {self._path}"
                    )
                shutil.rmtree(self._path)

            build_env._ctx.log(
                f"Creating isolated environment: {self._env_backend.display_name}...",
                kind=("step",),
            )
            self._env_backend.create(self._path)
            self.write_gitignore(path)
        except Exception:
            self.__exit__(*sys.exc_info())
            raise

        return self

    def __exit__(self, *args: object) -> None:
        """退出时保留环境目录，供后续排查使用。"""
        _ = args
