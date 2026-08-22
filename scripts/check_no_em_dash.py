"""Enforce non-negotiable rule number one of this project: no em dash, no en dash.

This project forbids the Unicode em dash character (U+2014) and the Unicode
en dash character (U+2013) anywhere in tracked source files, including code,
comments, docstrings, YAML, and markdown. Use a comma, colon, semicolon, full
stop, or parentheses instead. Plain ASCII hyphen-minus is fine.

This script walks the repository tree, starting from its own location's
parent directory (the repository root), and fails with exit code 1 if it
finds either forbidden character in any scanned file. It is invoked by the
CI workflow on every push and pull request, so a violation is caught
automatically rather than relying on manual review.

Usage:
    python scripts/check_no_em_dash.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)
FORBIDDEN_CHARACTERS = (EM_DASH, EN_DASH)

SCANNED_EXTENSIONS = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".txt",
    ".html",
    ".json",
    ".cfg",
    ".toml",
}

# Files that carry no extension but are still project text, so they must be
# scanned. LICENSE in particular is prose and a plausible place for an em dash.
SCANNED_EXTENSIONLESS_NAMES = {
    "LICENSE",
    ".gitignore",
    ".gitattributes",
}

SKIPPED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "data",
}


def get_repo_root() -> Path:
    """Return the repository root: the parent directory of this script's own location."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


def iter_scanned_files(repo_root: Path):
    """Yield every scannable file under repo_root.

    A file is scannable if its extension is in SCANNED_EXTENSIONS or its exact
    name is in SCANNED_EXTENSIONLESS_NAMES.

    Directories named in SKIPPED_DIR_NAMES are pruned from the walk at any
    depth, so their contents are never visited or yielded.
    """
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIPPED_DIR_NAMES)
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            if (
                file_path.suffix in SCANNED_EXTENSIONS
                or file_path.name in SCANNED_EXTENSIONLESS_NAMES
            ):
                yield file_path


def find_violations(file_path: Path) -> list[tuple[int, str]]:
    """Return a list of (line_number, stripped_line_text) pairs for a file.

    Only lines containing at least one forbidden character are included.
    The file is read as UTF-8 with decoding errors replaced rather than
    raised, so a stray non-UTF-8 byte cannot crash the scan.
    """
    violations: list[tuple[int, str]] = []
    text = file_path.read_text(encoding="utf-8", errors="replace")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(character in line for character in FORBIDDEN_CHARACTERS):
            violations.append((line_number, line.strip()))
    return violations


def main() -> int:
    """Scan the repository for em dash or en dash violations.

    Prints one line per offending source line and returns 1 if any are
    found, otherwise prints a summary line and returns 0.
    """
    repo_root = get_repo_root()
    files_scanned = 0
    found_violation = False

    for file_path in iter_scanned_files(repo_root):
        files_scanned += 1
        relative_path = file_path.relative_to(repo_root)
        for line_number, line_text in find_violations(file_path):
            found_violation = True
            print(f"{relative_path}:{line_number}: {line_text}")

    if found_violation:
        return 1

    print(f"no em dash or en dash found in {files_scanned} files scanned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
