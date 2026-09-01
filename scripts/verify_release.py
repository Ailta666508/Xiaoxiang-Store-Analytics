#!/usr/bin/env python3
"""Check provenance metadata and public-release boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "source-manifest.json"
FORBIDDEN_SUFFIXES = {".doc", ".docx", ".joblib", ".pdf", ".pkl", ".ppt", ".pptx", ".xls", ".xlsx", ".zip"}
REQUIRED_DISCLOSURES = (
    "March–June 2025",
    "without a commit-by-commit version history",
    "curated release",
)


def main() -> None:
    errors: list[str] = []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    hashes = [source["sha256"] for source in manifest["sources"]]
    if len(hashes) != len(set(hashes)):
        errors.append("source manifest contains duplicate script hashes")
    for source in manifest["sources"]:
        if not (ROOT / source["release_path"]).is_file():
            errors.append(f"missing mapped entry point: {source['release_path']}")

    python_hashes: dict[str, list[str]] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = str(path.relative_to(ROOT))
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden private/generated file: {relative}")
        if path.suffix == ".py":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            python_hashes.setdefault(digest, []).append(relative)
            source = path.read_text(encoding="utf-8")
            if "C:" + "\\Users\\" in source:
                errors.append(f"machine-specific Windows path: {relative}")
            if "debug" + "=True" in source:
                errors.append(f"unconditional Flask debug mode: {relative}")

    for paths in python_hashes.values():
        if len(paths) > 1:
            errors.append("duplicate Python files: " + ", ".join(paths))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in REQUIRED_DISCLOSURES:
        if phrase not in readme:
            errors.append(f"missing README disclosure: {phrase}")

    if errors:
        raise SystemExit("Release verification failed:\n- " + "\n- ".join(errors))
    print(
        f"Release verification passed ({len(manifest['sources'])} source mappings; "
        f"{sum(len(paths) for paths in python_hashes.values())} Python files)."
    )


if __name__ == "__main__":
    main()
