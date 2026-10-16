"""Inspect the latest user revision without replacing previous snapshots."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / "docs/audits/2026-10-15_created_deck_review/inspect_and_render.py"
spec = importlib.util.spec_from_file_location("deck_inspector", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = Path(__file__).resolve().parent

if __name__ == "__main__":
    module.main()
