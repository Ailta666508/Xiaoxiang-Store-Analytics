from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_machine_specific_windows_path() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ROOT.rglob("*.py")
        if ".git" not in path.parts
    )
    assert "C:" + "\\Users\\" not in source
    assert "debug" + "=True" not in source


def test_private_inputs_and_generated_models_are_not_versioned() -> None:
    forbidden = {".xlsx", ".xls", ".pdf", ".pkl", ".joblib"}
    assert not [
        path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden
    ]
