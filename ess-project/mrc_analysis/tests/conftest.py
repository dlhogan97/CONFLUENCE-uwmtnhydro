"""Shared test fixtures and path setup.

Adds the package parent directory to ``sys.path`` so tests import ``mrc``
without an install step (this is a plain module folder, not a pip package).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrc.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()
