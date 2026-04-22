from __future__ import annotations

from pcstubgen.type_models import AnyType, DictType, ListType, RawType, UnionType


def test_collect_imports_returns_recursive_dependency_set() -> None:
    node = DictType(
        RawType("numpy.ndarray", imports=("numpy",)),
        UnionType((AnyType(), RawType("collections.abc.Buffer", imports=("collections.abc",)))),
    )

    assert node.collect_imports() == {"numpy", "collections.abc", "typing"}


def test_union_canonicalize_short_circuits_any() -> None:
    canonical = UnionType(
        (
            RawType.str_,
            UnionType((RawType.int_, AnyType())),
        )
    ).canonicalize()

    assert canonical == AnyType()


def test_union_canonicalize_flattens_deduplicates_and_sorts_members() -> None:
    canonical = UnionType(
        (
            RawType.str_,
            UnionType((RawType.float_, RawType.int_)),
            RawType.bool_,
            UnionType(
                (
                    RawType.int_,
                    UnionType((RawType.bool_, UnionType(()))),
                )
            ),
        )
    ).canonicalize()

    assert canonical == UnionType(
        (
            RawType.bool_,
            RawType.float_,
            RawType.int_,
            RawType.str_,
        )
    )


def test_container_canonicalize_recurses_into_nested_union_members() -> None:
    canonical = ListType(
        UnionType(
            (
                RawType.str_,
                UnionType((RawType.int_, RawType.bool_)),
                RawType.bool_,
            )
        )
    ).canonicalize()

    assert canonical == ListType(
        UnionType((RawType.bool_, RawType.int_, RawType.str_))
    )


def test_container_canonicalize_folds_single_union_members() -> None:
    canonical = DictType(
        UnionType((RawType.str_,)),
        UnionType((RawType.int_,)),
    ).canonicalize()

    assert canonical == DictType(RawType.str_, RawType.int_)
