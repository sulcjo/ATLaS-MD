#!/usr/bin/env python3
"""Check local Markdown/HTML destinations used by the README and MkDocs pages."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


INLINE_LINK = re.compile(r"!?(?:\[[^\]]*\])\(\s*(?:<([^>]+)>|([^\s)]+))")
REFERENCE = re.compile(r"^\s*\[[^\]]+\]:\s*(?:<([^>]+)>|([^\s]+))", re.MULTILINE)
HTML_ATTR = re.compile(r"\b(?:href|src)\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)
FENCED = re.compile(r"(?ms)^\s*(?:```|~~~).*?^\s*(?:```|~~~)\s*$")
INLINE_CODE = re.compile(r"(?s)(`+).*?\1")


def destinations(text: str):
    text = FENCED.sub("", text)
    text = INLINE_CODE.sub("", text)
    for pattern in (INLINE_LINK, REFERENCE):
        for match in pattern.finditer(text):
            yield match.start(), match.group(1) or match.group(2)
    for match in HTML_ATTR.finditer(text):
        yield match.start(), match.group(2).strip()


def is_local(destination: str) -> bool:
    if not destination or destination.startswith("#") or destination.startswith("//"):
        return False
    parsed = urlsplit(destination)
    return not parsed.scheme and not parsed.netloc


def target_path(source: Path, destination: str, docs_root: Path) -> Path | None:
    if not is_local(destination):
        return None
    raw_path = unquote(urlsplit(destination).path)
    if not raw_path:
        return None
    if raw_path.startswith("/"):
        candidate = docs_root / raw_path.lstrip("/")
    else:
        candidate = source.parent / raw_path
    if candidate.is_dir():
        candidate = candidate / "index.md"
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Markdown files or directories to inspect")
    parser.add_argument("--docs-root", type=Path, default=Path("docs/atlas-md"))
    args = parser.parse_args()
    root = Path.cwd()
    docs_root = (root / args.docs_root).resolve()
    sources: set[Path] = set()
    for path in args.paths:
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            sources.update(p.resolve() for p in path.rglob("*.md"))
        elif path.suffix.lower() == ".md":
            sources.add(path.resolve())
        else:
            parser.error(f"not a Markdown file or directory: {path}")

    failures = []
    checked = 0
    for source in sorted(sources):
        text = source.read_text(encoding="utf-8")
        for offset, destination in destinations(text):
            candidate = target_path(source, destination, docs_root)
            if candidate is None:
                continue
            checked += 1
            if not candidate.exists():
                line = text.count("\n", 0, offset) + 1
                failures.append(f"{source.relative_to(root)}:{line}: missing local target {destination!r}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Checked {checked} local link and asset targets in {len(sources)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
