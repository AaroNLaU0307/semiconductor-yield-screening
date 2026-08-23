"""Enforce non-negotiable rule number one of this project: no em dash, no en dash.

This project forbids the Unicode em dash character (U+2014) and the Unicode
en dash character (U+2013) anywhere in tracked source files, including code,
comments, docstrings, YAML, and markdown. Use a comma, colon, semicolon, full
stop, or parentheses instead. Plain ASCII hyphen-minus is fine.

The scan is slightly wider than that one rule: it also rejects the typographic
look-alikes that cause the same class of problem, listed in FORBIDDEN_TYPOGRAPHY.
A U+2212 minus sign is visually identical to a hyphen and silently stops being a
number the moment anyone copies it into code; two of them reached this repository's
README before this check existed. Curly quotes break the same way.

It is a denylist, not an ASCII-only allowlist. Rejecting every non-ASCII character
would also redden the build over a diacritic in an author's name or a citation,
which is legitimate text and not the problem this rule exists to prevent.

This script walks the repository tree, starting from its own location's
parent directory (the repository root), and fails with exit code 1 if it
finds a violation in any scanned file. It is invoked by the CI workflow on
every push and pull request, so a violation is caught automatically rather
than relying on manual review.

Usage:
    python scripts/check_no_em_dash.py
"""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

#: The characters this check actually rejects: the em dash forbidden by
#: non-negotiable 1, plus the typographic look-alikes that cause the same class of
#: problem. A U+2212 minus sign is visually identical to a hyphen and silently stops
#: being a number the moment anyone copies it into code; two of them reached this
#: repository's README before this check existed. Curly quotes break the same way.
#:
#: This is deliberately a denylist rather than "reject everything non-ASCII". An
#: allowlist of ASCII would redden the build over a diacritic in an author's name or a
#: citation, which is a legitimate thing to write and not the problem this rule exists
#: to prevent.
FORBIDDEN_TYPOGRAPHY = {
    chr(0x2014): "em dash",
    chr(0x2013): "en dash",
    chr(0x2012): "figure dash",
    chr(0x2015): "horizontal bar",
    chr(0x2212): "minus sign",
    chr(0x2018): "left single quotation mark",
    chr(0x2019): "right single quotation mark",
    chr(0x201C): "left double quotation mark",
    chr(0x201D): "right double quotation mark",
    chr(0x2026): "horizontal ellipsis",
    chr(0x00A0): "no-break space",
}

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

#: Longest offending line printed. A minified bundle on one line would otherwise
#: flood the console and bury the actual violations.
MAX_REPORTED_LINE_LENGTH = 120

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


def describe_character(character: str) -> str:
    """Name a character for the violation message."""
    if character in FORBIDDEN_TYPOGRAPHY:
        return FORBIDDEN_TYPOGRAPHY[character]
    try:
        return unicodedata.name(character).lower()
    except ValueError:
        return "unnamed character"


def find_violations(file_path: Path) -> list[tuple[int, str, str]]:
    """Return (line_number, reason, stripped_line_text) triples for a file.

    A line is a violation if it contains any character in FORBIDDEN_TYPOGRAPHY: the
    em dash and en dash forbidden by non-negotiable 1, plus the look-alikes that fail
    the same way. Other non-ASCII characters, such as a diacritic in a cited author's
    name, are permitted.

    The file is read as UTF-8 with decoding errors replaced rather than raised, so
    a stray non-UTF-8 byte cannot crash the scan.
    """
    violations: list[tuple[int, str, str]] = []
    text = file_path.read_text(encoding="utf-8", errors="replace")
    for line_number, line in enumerate(text.splitlines(), start=1):
        offenders = {c for c in line if c in FORBIDDEN_TYPOGRAPHY}
        if not offenders:
            continue
        named = ", ".join(
            f"U+{ord(c):04X} {describe_character(c)}" for c in sorted(offenders)
        )
        # The offending line contains, by definition, characters the console may not
        # be able to encode. A Windows cp1252 stdout raises UnicodeEncodeError on
        # U+2212, so the guard would crash instead of reporting. Escaping to ASCII
        # keeps the output printable everywhere and shows the culprit explicitly.
        safe = line.strip().encode("ascii", "backslashreplace").decode("ascii")
        if len(safe) > MAX_REPORTED_LINE_LENGTH:
            safe = safe[:MAX_REPORTED_LINE_LENGTH] + " ... (line truncated)"
        violations.append((line_number, named, safe))
    return violations


def main() -> int:
    """Scan the repository for em dashes and other forbidden typography.

    Prints one line per offending source line and returns 1 if any are
    found, otherwise prints a summary line and returns 0.
    """
    repo_root = get_repo_root()
    files_scanned = 0
    found_violation = False

    for file_path in iter_scanned_files(repo_root):
        files_scanned += 1
        relative_path = file_path.relative_to(repo_root)
        for line_number, reason, line_text in find_violations(file_path):
            found_violation = True
            print(f"{relative_path}:{line_number}: [{reason}] {line_text}")

    if found_violation:
        print(
            "\nNon-negotiable 1 forbids the em dash. This check also rejects the "
            "typographic look-alikes that fail the same way. Use a comma, colon, "
            "semicolon, full stop, or parentheses; a plain ASCII hyphen for a minus "
            "sign; and straight quotes. Other non-ASCII characters, such as a "
            "diacritic in a cited name, are allowed."
        )
        return 1

    print(f"no forbidden typography found in {files_scanned} files scanned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
