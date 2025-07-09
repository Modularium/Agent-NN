"""Test configuration and fixtures."""

import importlib
import sys
import types
import pytest

try:  # check for heavy dependency
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    sys.modules.setdefault("torch", types.ModuleType("torch"))

try:
    from typer.testing import CliRunner  # type: ignore  # noqa
except ImportError:  # pragma: no cover - optional dependency
    typer_stub = types.ModuleType("typer")
    typer_stub.testing = types.SimpleNamespace(CliRunner=object)
    sys.modules.setdefault("typer", typer_stub)
    sys.modules.setdefault("typer.testing", typer_stub.testing)

try:
    import pydantic  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pydantic = None


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line(
        "markers", "heavy: requires optional dependencies like torch"
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--skip-heavy",
        action="store_true",
        default=False,
        help="skip tests requiring heavy dependencies",
    )


def pytest_collection_modifyitems(config, items):
    """Handle test markers."""
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    skip_heavy = pytest.mark.skip(reason="heavy dependency missing or skipped")

    for item in items:
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
        if "heavy" in item.keywords:
            if config.getoption("--skip-heavy") or not TORCH_AVAILABLE:
                item.add_marker(skip_heavy)
        if "unit" not in item.keywords and "integration" not in item.keywords:
            item.add_marker(pytest.mark.skip(reason="missing unit marker"))
