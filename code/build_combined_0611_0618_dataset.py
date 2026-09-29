"""6/11・6/18の既存NPYとWAVを、出典を保持して1実験フォルダへ複製する。

既存ファイルは変更しない。NPY名は先頭の熱流束値を維持し、その後へ実験日tagを
加える。これにより既存loaderのラベル読取りを保ちつつ、同名衝突を防ぐ。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "Pool_boiling" / "Subcooling_20_degrees" / "0.3"
TARGET_NAME = "2025.06.11_0.3_2_6.18_0.3_3"
DATASET_NAME = "waterflow_20260817_1s"
AUDIO_DIRS = ("録音データ", "録音データ_熱流束")
SOURCES = (
    ("2025.06.11_0.3_2", "20250611", 221505.1102),
    ("2025.06.18_0.3_3", "20250618", 271677.6816),
)
EXTRA_MANIFEST_FIELDS = (
    "source_experiment_name",
    "original_sample_filename",
    "original_source_wav_id",
    "original_source_wav_name",
)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _copy_verified(source: Path, target: Path, dry_run: bool) -> bool:
    """未作成ならcopyし、既存ならbyte数一致を確認する。copyしたときTrue。"""
    if target.exists():
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"既存出力がコピー元と一致しません: {target}")
        return False
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.stat().st_size != source.stat().st_size:
            raise IOError(f"コピー後のbyte数が一致しません: {target}")
    return True


def _renamed_sample(filename: str, source_tag: str) -> str:
    label, separator, rest = filename.partition("_")
    if not separator or not rest:
        raise ValueError(f"熱流束prefixを持たないNPY名です: {filename}")
    float(label)
    return f"{label}_exp-{source_tag}_{rest}"


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"manifest headerがありません: {path}")
        return list(reader.fieldnames), list(reader)


def _write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]], dry_run: bool):
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _condition_manifests(source_root: Path) -> dict[Path, Path]:
    return {
        path.parent.relative_to(source_root): path
        for path in source_root.rglob("chunk_manifest.csv")
    }


def build(dry_run: bool = False) -> dict:
    target_root = EXPERIMENT_ROOT / TARGET_NAME
    if not _within(target_root, EXPERIMENT_ROOT) or target_root == EXPERIMENT_ROOT:
        raise ValueError(f"出力先が実験root直下ではありません: {target_root}")

    source_roots = {
        name: EXPERIMENT_ROOT / name / "data" / "npy" / DATASET_NAME
        for name, _, _ in SOURCES
    }
    for name, source_root in source_roots.items():
        if not source_root.is_dir():
            raise FileNotFoundError(f"NPY sourceがありません: {name}: {source_root}")

    condition_maps = {
        name: _condition_manifests(source_root)
        for name, source_root in source_roots.items()
    }
    expected_conditions = set(next(iter(condition_maps.values())))
    if not expected_conditions:
        raise ValueError("結合対象のcondition manifestがありません。")
    for name, conditions in condition_maps.items():
        if set(conditions) != expected_conditions:
            missing = sorted(str(item) for item in expected_conditions - set(conditions))
            extra = sorted(str(item) for item in set(conditions) - expected_conditions)
            raise ValueError(f"condition集合が一致しません: {name}; missing={missing}; extra={extra}")

    existing_summary = target_root / "combined_dataset_manifest.json"
    if existing_summary.is_file():
        saved = json.loads(existing_summary.read_text(encoding="utf-8"))
        if saved.get("source_experiments") != [item[0] for item in SOURCES]:
            raise ValueError(f"既存の統合folderは別のsourceを示しています: {existing_summary}")

    total_rows = total_new_files = total_existing_files = 0
    target_data_root = target_root / "data" / "npy" / DATASET_NAME
    condition_summaries = []
    for relative in sorted(expected_conditions, key=lambda value: value.as_posix()):
        merged_rows: list[dict[str, str]] = []
        merged_fields: list[str] = []
        seen_names: set[str] = set()
        source_counts = {}
        for source_name, source_tag, _ in SOURCES:
            manifest_path = condition_maps[source_name][relative]
            fields, rows = _read_manifest(manifest_path)
            for field in [*fields, *EXTRA_MANIFEST_FIELDS]:
                if field not in merged_fields:
                    merged_fields.append(field)
            source_condition = manifest_path.parent
            manifest_names = {row.get("sample_filename", "") for row in rows}
            file_names = {path.name for path in source_condition.glob("*.npy")}
            if "" in manifest_names or manifest_names != file_names:
                raise ValueError(
                    f"NPYとmanifestのsample集合が一致しません: {manifest_path}"
                )
            for row in rows:
                original_name = row["sample_filename"]
                renamed = _renamed_sample(original_name, source_tag)
                if renamed in seen_names:
                    raise ValueError(f"統合後のNPY名が重複します: {renamed}")
                seen_names.add(renamed)
                original_wav_id = row.get("source_wav_id", "")
                original_wav_name = row.get("source_wav_name", "")
                merged = dict(row)
                merged.update({
                    "sample_filename": renamed,
                    "experiment_name": TARGET_NAME,
                    "source_wav_id": f"{source_tag}::{original_wav_id}",
                    "source_wav_name": f"{source_tag}::{original_wav_name}",
                    "source_experiment_name": source_name,
                    "original_sample_filename": original_name,
                    "original_source_wav_id": original_wav_id,
                    "original_source_wav_name": original_wav_name,
                })
                merged_rows.append(merged)
                copied = _copy_verified(
                    source_condition / original_name,
                    target_data_root / relative / renamed,
                    dry_run,
                )
                total_new_files += int(copied)
                total_existing_files += int(not copied)
            source_counts[source_name] = len(rows)
        merged_rows.sort(key=lambda row: row["sample_filename"])
        _write_manifest(
            target_data_root / relative / "chunk_manifest.csv",
            merged_fields,
            merged_rows,
            dry_run,
        )
        total_rows += len(merged_rows)
        condition_summaries.append({
            "condition": relative.as_posix(),
            "rows": len(merged_rows),
            "source_rows": source_counts,
        })

    copied_audio = 0
    audio_counts = {directory: 0 for directory in AUDIO_DIRS}
    copied_metadata = 0
    for source_name, _, _ in SOURCES:
        source_experiment = EXPERIMENT_ROOT / source_name
        for audio_directory in AUDIO_DIRS:
            audio_root = source_experiment / audio_directory
            if not audio_root.is_dir():
                raise FileNotFoundError(f"音響folderがありません: {audio_root}")
            for source in sorted(audio_root.glob("*.wav")):
                audio_counts[audio_directory] += 1
                copied_audio += int(_copy_verified(
                    source,
                    target_root / audio_directory / source_name / source.name,
                    dry_run,
                ))
        for pattern in ("*.csv", "*.xlsx"):
            for source in sorted(source_experiment.glob(pattern)):
                copied_metadata += int(_copy_verified(
                    source,
                    target_root / "source_metadata" / source_name / source.name,
                    dry_run,
                ))

    threshold_values = [item[2] for item in SOURCES]
    summary = {
        "schema_version": 1,
        "combined_experiment": TARGET_NAME,
        "source_experiments": [item[0] for item in SOURCES],
        "dataset_name": DATASET_NAME,
        "copy_mode": "independent_file_copy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "onb_threshold_w_m2": sum(threshold_values) / len(threshold_values),
        "onb_aggregation": "arithmetic_mean_of_confirmed_source_experiment_thresholds",
        "source_onb_thresholds_w_m2": {
            name: threshold for name, _, threshold in SOURCES
        },
        "manifest_rows_total": total_rows,
        "condition_count": len(condition_summaries),
        "conditions": condition_summaries,
        "audio_layouts": [
            f"{directory}/<source_experiment>/*.wav" for directory in AUDIO_DIRS
        ],
        "audio_file_counts": audio_counts,
        "provenance_fields_added": list(EXTRA_MANIFEST_FIELDS),
    }
    if not dry_run:
        target_root.mkdir(parents=True, exist_ok=True)
        (target_root / "regression_result").mkdir(exist_ok=True)
        existing_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({
        "target": str(target_root),
        "dry_run": dry_run,
        "conditions": len(condition_summaries),
        "manifest_rows": total_rows,
        "new_npy_files": total_new_files,
        "existing_npy_files": total_existing_files,
        "new_audio_files": copied_audio,
        "new_metadata_files": copied_metadata,
        "onb_threshold_w_m2": summary["onb_threshold_w_m2"],
    }, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
