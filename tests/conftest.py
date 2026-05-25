"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_rust_backend_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Rust is opt-in; only test_rust_parity enables it via its fixture."""
    if request.node.fspath.basename == "test_rust_parity.py":
        return
    monkeypatch.delenv("WELL_ARRAY_SIM_USE_RUST", raising=False)
