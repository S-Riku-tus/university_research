"""Inventory the retired strategy in saved results and research documents."""
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PATTERN = r"prediction[_ -]max|ensemble__max|予測最大値"


def long_path(path):
    path = os.path.abspath(path)
    return Path(path if path.startswith("\\\\?\\") else "\\\\?\\" + path)


def rg_files(*args):
    result = subprocess.run(["rg", *args], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8")
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return [Path(p) for p in result.stdout.splitlines()]


def main():
    paths = rg_files("--hidden", "--no-ignore", "-i", "-l", "-g", "!.git/**",
                     "-g", "!experiments/2026-09-14_remove_prediction_max/**",
                     "-g", "*.{py,md,txt,json,csv,yaml,yml,ipynb,log,xml}", PATTERN, ".")
    items = [{"path": p.as_posix(), "bytes": long_path(ROOT / p).stat().st_size} for p in paths]
    (OUT / "text_inventory.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"matching_files": len(items),
                      "by_extension": dict(collections.Counter(p.suffix for p in paths)),
                      "by_top_directory": dict(collections.Counter(p.parts[0] for p in paths)),
                      "total_bytes": sum(i["bytes"] for i in items)}, ensure_ascii=False))
    for p in paths:
        if p.parts[0] not in ("Pool_boiling", "trush_box") and p.suffix not in (".csv", ".json", ".txt", ".log"):
            print(p.as_posix())
    office = rg_files("--files", "--hidden", "-g", "!.git/**", "-g", "*.{docx,pptx,xlsx}")
    findings = []
    for path in office:
        try:
            with zipfile.ZipFile(long_path(ROOT / path)) as z:
                for entry in z.namelist():
                    if not entry.endswith(".xml"):
                        continue
                    try:
                        content = "".join(ET.fromstring(z.read(entry)).itertext())
                    except ET.ParseError:
                        continue
                    match = re.search(PATTERN, content, re.I)
                    if match:
                        findings.append({"path": path.as_posix(), "member": entry,
                                         "excerpt": content[max(0, match.start()-80):match.end()+180]})
        except zipfile.BadZipFile:
            continue
    (OUT / "office_inventory.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"office_checked": len(office), "office_matches": findings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
