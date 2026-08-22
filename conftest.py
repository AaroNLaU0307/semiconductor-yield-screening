"""Put the repository root on sys.path so `import yield_screening` works from pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
