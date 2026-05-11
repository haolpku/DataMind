"""Registry mechanics.

We don't test the built-in registries' *contents* (they're empty in Phase 1).
We test that the registry machinery itself works, using throwaway fixtures —
this is the contract every future provider will rely on.
"""
from __future__ import annotations

import pytest

from datamind.core.errors import ConfigError
from datamind.core.registry import Registry


class _Face:
    """Minimal base for fake providers used in these tests."""


def test_register_and_create_round_trip():
    reg: Registry = Registry("fake")

    @reg.register("x")
    class X(_Face):
        def __init__(self, value: int) -> None:
            self.value = value

    inst = reg.create("x", value=42)
    assert isinstance(inst, X)
    assert inst.value == 42
    assert reg.known() == ["x"]
    assert "x" in reg
    assert len(reg) == 1


def test_duplicate_name_raises():
    reg: Registry = Registry("fake")

    @reg.register("dup")
    class A(_Face):
        pass

    with pytest.raises(ConfigError) as exc:

        @reg.register("dup")
        class B(_Face):  # noqa: F841
            pass

    assert "already registered" in str(exc.value)


def test_unknown_name_hint_lists_registered():
    reg: Registry = Registry("database")

    @reg.register("sqlite")
    class SQL(_Face):
        pass

    @reg.register("mysql")
    class MySQL(_Face):
        pass

    with pytest.raises(ConfigError) as exc:
        reg.create("oracle")

    msg = str(exc.value)
    assert "Unknown database provider 'oracle'" in msg
    assert "mysql" in msg
    assert "sqlite" in msg


def test_empty_name_rejected():
    reg: Registry = Registry("fake")
    with pytest.raises(ConfigError):

        @reg.register("")  # type: ignore[arg-type]
        class _Bad(_Face):  # noqa: F841
            pass


def test_get_class_returns_same_ref():
    reg: Registry = Registry("fake")

    @reg.register("k")
    class K(_Face):
        pass

    assert reg.get_class("k") is K


def test_global_registries_exist_and_are_registries():
    """The five global registries are `Registry` instances and non-None.

    Populated entries depend on which provider packages have been imported.
    """
    from datamind.core.registry import (
        db_registry,
        embedding_registry,
        graph_registry,
        memory_registry,
        retriever_registry,
        vector_store_registry,
    )

    for reg in (
        embedding_registry,
        retriever_registry,
        graph_registry,
        db_registry,
        memory_registry,
        vector_store_registry,
    ):
        assert isinstance(reg, Registry)
