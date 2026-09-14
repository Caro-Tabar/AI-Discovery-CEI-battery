"""Smoke tests for the installed package."""

import cei_scout


def test_package_import() -> None:
    """Verify that the top-level package is importable."""
    assert cei_scout.__name__ == "cei_scout"
