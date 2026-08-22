"""Pytest configuration.

Puts the repository root on sys.path so `import yield_screening` works without an
install step, and registers the markers the suite uses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    """Register custom markers so they do not raise PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "slow: end to end checks that run the whole pipeline, roughly two minutes",
    )
