#!/usr/bin/env python3
"""Basic release hygiene checks for the public navatala_gpu tree."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".h", ".hpp", ".mm", ".cu", ".hip", ".cl",
    ".comp", ".metal", ".cmake", ".txt", ".md", ".json", ".yaml", ".yml",
    ".py", ".pyi", ".sh", ".toml", ".rst", ".H", ".C",
}

PRIVATE_REPO_NAME = "codegen" + "_gpu"

PATTERNS = [
    ("absolute project path", re.compile(r"/opt/data/projects/")),
    ("home directory path", re.compile(r"/home/[A-Za-z0-9_.-]+/")),
    ("private specs path", re.compile(r"\bnavatala_specs/")),
    ("lean source path", re.compile(r"\.lean\b")),
    ("internal pipeline script", re.compile(r"run_navatala_pipeline\.sh")),
    ("private generator repo name", re.compile(r"\b" + re.escape(PRIVATE_REPO_NAME) + r"\b", re.IGNORECASE)),
    ("private generator namespace", re.compile(r"CodeGenGPU\.")),
    ("private GPU IR type namespace", re.compile(r"GPU\.GPUType")),
    ("legacy generated kernel path fragment", re.compile(r"Calculations_GPU")),
    (
        "credential-looking assignment",
        re.compile(
            r"(?i)(password|secret|api[_-]?key|(?:access|refresh|auth|bearer|session|id)[_-]?token)\s*[:=]"
        ),
    ),
]


def is_text(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", help="release tree root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_text(path):
            continue
        rel = path.relative_to(root).as_posix()
        if rel in {
            "tools/check_release_hygiene.py",
            "tools/check_generated_content_tells.sh",
            ".github/workflows/wheel-hygiene.yml",
            ".github/workflows/publish-pypi.yml",
            ".github/workflows/rocm-validation.yml",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(root)
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append((str(rel), line, label))

    if findings:
        for rel, line, label in findings:
            print(f"{rel}:{line}: {label}")
        return 1

    print("release hygiene check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
