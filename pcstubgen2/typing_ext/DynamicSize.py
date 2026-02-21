from __future__ import annotations


class DynamicSize:
    def __init__(self, *dim: int | str):
        self.dim: tuple[int | str, ...] = dim

    def __repr__(self):
        return (
            f"{self.__module__}."
            f"{self.__class__.__qualname__}"
            f"({', '.join(repr(d) for d in self.dim)})"
        )
