"""Shared pytest fixtures and configuration.

Each unit test should be hermetic: no real API calls, no real disk writes
outside `tmp_path`. We enforce that by clearing the settings cache between
tests (so env-var changes inside a test take effect) and by skipping the
integration suite unless `-m integration` is passed.
"""

from __future__ import annotations

import pytest

from src.config import settings as settings_module


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    """Clear the memoized `get_settings()` so each test sees fresh env vars."""
    settings_module.get_settings.cache_clear()
    yield
    settings_module.get_settings.cache_clear()
